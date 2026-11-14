"""Paired fixed-noise validation for CPU video checkpoints."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from video_foundation.training.train_cpu_video import ClipDataset
from video_foundation.models.unet import UNet
from video_foundation.models.cpu_video_unet import CPUVideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--image_checkpoint", required=True)
    parser.add_argument("--video_checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    torch.set_num_threads(4)
    image_ckpt = torch.load(args.image_checkpoint, map_location="cpu", weights_only=False)
    video_ckpt = torch.load(args.video_checkpoint, map_location="cpu", weights_only=False)
    cfg = image_ckpt["config"]
    mc = cfg["model"]
    spatial = UNet(in_channels=3, model_channels=mc["model_channels"], channel_mult=mc["channel_mult"],
                   num_res_blocks=mc["num_res_blocks"], num_classes=mc["num_classes"],
                   num_color_classes=mc["num_color_classes"])
    spatial.load_state_dict(image_ckpt["model_state_dict"])
    baseline = CPUVideoUNet(spatial, args.frames, freeze_spatial=True, use_motion=False)
    model = CPUVideoUNet(UNet(in_channels=3, model_channels=mc["model_channels"], channel_mult=mc["channel_mult"],
                              num_res_blocks=mc["num_res_blocks"], num_classes=mc["num_classes"],
                              num_color_classes=mc["num_color_classes"]), args.frames, freeze_spatial=True)
    model.load_state_dict(video_ckpt["model_state_dict"])
    dataset = ClipDataset(args.data_dir, args.frames, args.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    process = DiffusionProcess(GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear"))
    generator = torch.Generator().manual_seed(args.seed)
    baseline_losses, video_losses, trajectory_losses = [], [], []
    baseline_temporal_errors, video_temporal_errors = [], []
    baseline_foreground_losses, video_foreground_losses = [], []
    temporal_by_t = [[[], []] for _ in range(3)]
    temporal_by_region = [[[], []] for _ in range(2)]
    baseline.eval(); model.eval()
    with torch.no_grad():
        for clean, shapes, colors, motions, speeds, offsets in loader:
            if len(baseline_losses) >= args.samples:
                break
            t = torch.randint(0, 1000, (1,), generator=generator)
            noise = torch.randn(clean.shape, generator=generator)
            noisy, _ = process.q_sample(clean, t, noise)
            _, alpha, one_minus_alpha = process.schedule.get(t, clean.device)
            alpha_5d = alpha.view(-1, 1, 1, 1, 1)
            one_minus_5d = one_minus_alpha.view(-1, 1, 1, 1, 1)
            target = alpha_5d.sqrt() * noise - one_minus_5d.sqrt() * clean
            base_prediction = baseline(noisy, t, shapes, colors)
            video_prediction = model(noisy, t, shapes, colors, motions, speeds)
            predicted_offsets = model.predict_offsets(motions, speeds)
            trajectory_losses.append(((predicted_offsets - offsets) * args.image_size).square().mean().sqrt().item())
            baseline_losses.append((base_prediction - target).square().mean().item())
            video_losses.append((video_prediction - target).square().mean().item())
            foreground = (clean > -0.95).any(dim=1, keepdim=True).to(clean.dtype)
            weights = 1.0 + 4.0 * foreground
            temporal_foreground = torch.maximum(foreground[:, :, 1:], foreground[:, :, :-1])
            for model_index, (prediction, foreground_results, temporal_results) in enumerate((
                (base_prediction, baseline_foreground_losses, baseline_temporal_errors),
                (video_prediction, video_foreground_losses, video_temporal_errors),
            )):
                pixel_error = (prediction - target).square().mean(dim=1, keepdim=True)
                foreground_results.append(((pixel_error * weights).flatten(1).sum(1) / weights.flatten(1).sum(1)).mean().item())
                pred_x0 = alpha_5d.sqrt() * noisy - one_minus_5d.sqrt() * prediction
                pred_delta = pred_x0[:, :, 1:] - pred_x0[:, :, :-1]
                clean_delta = clean[:, :, 1:] - clean[:, :, :-1]
                temporal_error = (pred_delta - clean_delta).square().mean(dim=1, keepdim=True)
                temporal_results.append(temporal_error.mean().sqrt().item())
                fg_mask = temporal_foreground
                bg_mask = 1.0 - fg_mask
                temporal_by_region[0][model_index].append(
                    ((temporal_error * fg_mask).sum() / fg_mask.sum().clamp_min(1)).sqrt().item()
                )
                temporal_by_region[1][model_index].append(
                    ((temporal_error * bg_mask).sum() / bg_mask.sum().clamp_min(1)).sqrt().item()
                )
                timestep = int(t.item())
                bucket = 0 if timestep < 333 else (1 if timestep < 666 else 2)
                temporal_by_t[bucket][model_index].append(temporal_error.mean().sqrt().item())
    b, v = sum(baseline_losses) / len(baseline_losses), sum(video_losses) / len(video_losses)
    trajectory = sum(trajectory_losses) / len(trajectory_losses)
    bfg = sum(baseline_foreground_losses) / len(baseline_foreground_losses)
    vfg = sum(video_foreground_losses) / len(video_foreground_losses)
    bt = sum(baseline_temporal_errors) / len(baseline_temporal_errors)
    vt = sum(video_temporal_errors) / len(video_temporal_errors)
    temporal_bucket_summary = ",".join(
        f"{name}:{sum(values[0]) / max(len(values[0]), 1):.4f}->{sum(values[1]) / max(len(values[1]), 1):.4f}"
        for name, values in zip(("low_t", "mid_t", "high_t"), temporal_by_t)
    )
    temporal_region_summary = ",".join(
        f"{name}:{sum(values[0]) / len(values[0]):.4f}->{sum(values[1]) / len(values[1]):.4f}"
        for name, values in zip(("fg", "bg"), temporal_by_region)
    )
    print(
        f"clips={len(baseline_losses)} v_mse={b:.6f}->{v:.6f} "
        f"foreground_v_mse={bfg:.6f}->{vfg:.6f} "
        f"temporal_gradient_rmse={bt:.6f}->{vt:.6f} "
        f"trajectory_pixel_rmse={trajectory:.3f} "
        f"temporal_by_t=[{temporal_bucket_summary}] temporal_by_region=[{temporal_region_summary}]"
    )


if __name__ == "__main__":
    main()
