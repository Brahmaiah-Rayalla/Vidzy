"""Paired fixed-noise validation for CPU video checkpoints."""

import argparse
import copy
import json
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
    parser.add_argument("--trajectory_reference_dir", default=None,
                        help="Training dataset used to estimate the expected velocity for each motion/speed label")
    parser.add_argument("--velocity_conditioning", choices=("metadata", "table"), default="metadata",
                        help="Use each clip's continuous velocity, or the categorical direction/speed table only")
    parser.add_argument("--noise_correlation", type=float, default=0.0,
                        help="Shared frame-noise fraction; 0 matches iid training noise, 1 matches the default sampler")
    parser.add_argument("--noise_alignment", choices=("canonical", "video"), default=None,
                        help="Coordinate system for noise; canonical matches canonical-noise training")
    args = parser.parse_args()
    if not 0.0 <= args.noise_correlation <= 1.0:
        parser.error("--noise_correlation must be in [0, 1]")
    torch.set_num_threads(4)
    image_ckpt = torch.load(args.image_checkpoint, map_location="cpu", weights_only=False)
    video_ckpt = torch.load(args.video_checkpoint, map_location="cpu", weights_only=False)
    cfg = image_ckpt["config"]
    mc = cfg["model"]
    spatial = UNet(in_channels=3, model_channels=mc["model_channels"], channel_mult=mc["channel_mult"],
                   num_res_blocks=mc["num_res_blocks"], num_classes=mc["num_classes"],
                   num_color_classes=mc["num_color_classes"],
                   num_size_classes=mc.get("num_size_classes", 0),
                   spatial_size_conditioning=mc.get("spatial_size_conditioning", False),
                   high_resolution_refinement=mc.get("high_resolution_refinement", False),
                   size_embedding_scale=mc.get("size_embedding_scale", 1.0),
                   middle_dilation=int(mc.get("middle_dilation", 1)))
    spatial.load_state_dict(image_ckpt["model_state_dict"])
    baseline = CPUVideoUNet(spatial, args.frames, freeze_spatial=True, use_motion=False)
    # The image-only reference must be the spatial prior alone. Its temporal
    # head is otherwise randomly initialized and can add arbitrary v-residuals,
    # making baseline scores seed/process dependent and invalidating ablations.
    with torch.no_grad():
        baseline.temporal[-1].weight.zero_()
        baseline.temporal[-1].bias.zero_()
    canonicalize_motion = not video_ckpt.get("video", {}).get("no_canonicalize_motion", False)
    if args.noise_alignment is None:
        args.noise_alignment = "canonical" if canonicalize_motion else "video"
    model = CPUVideoUNet(UNet(in_channels=3, model_channels=mc["model_channels"], channel_mult=mc["channel_mult"],
                              num_res_blocks=mc["num_res_blocks"], num_classes=mc["num_classes"],
                              num_color_classes=mc["num_color_classes"],
                              num_size_classes=mc.get("num_size_classes", 0),
                              spatial_size_conditioning=mc.get("spatial_size_conditioning", False),
                              high_resolution_refinement=mc.get("high_resolution_refinement", False),
                              size_embedding_scale=mc.get("size_embedding_scale", 1.0),
                              middle_dilation=int(mc.get("middle_dilation", 1))),
                          args.frames, freeze_spatial=True,
                          canonicalize_motion=canonicalize_motion,
                          gate_temporal_foreground=video_ckpt.get("video", {}).get("gate_temporal_foreground", False))
    # The explicit image checkpoint is the spatial prior under evaluation.
    # Older code silently overwrote it with the spatial weights embedded in the
    # video checkpoint, which made image-prior comparisons ineffective.
    model.spatial_unet.load_state_dict(image_ckpt["model_state_dict"])
    video_state = {
        key: value for key, value in video_ckpt["model_state_dict"].items()
        if not key.startswith("spatial_unet.")
    }
    incompatible = model.load_state_dict(video_state, strict=False)
    unexpected = incompatible.unexpected_keys
    missing = [key for key in incompatible.missing_keys if not key.startswith("spatial_unet.")]
    if unexpected or missing:
        raise RuntimeError(f"Incompatible video checkpoint: missing={missing}, unexpected={unexpected}")
    aligned_spatial_model = copy.deepcopy(model)
    with torch.no_grad():
        aligned_spatial_model.temporal[-1].weight.zero_()
        aligned_spatial_model.temporal[-1].bias.zero_()
    dataset = ClipDataset(args.data_dir, args.frames, args.image_size)
    reference_dir = args.trajectory_reference_dir or os.path.join(os.path.dirname(os.path.normpath(args.data_dir)), "train")
    reference_metadata_path = os.path.join(reference_dir, "all_metadata.json")
    if not os.path.isfile(reference_metadata_path):
        raise FileNotFoundError(
            "Trajectory label-only accuracy needs training metadata; pass --trajectory_reference_dir"
        )
    with open(reference_metadata_path, encoding="utf-8") as handle:
        reference_metadata = json.load(handle)
    velocity_values = {}
    for item in reference_metadata:
        key = (item["motion"], item["speed_adjective"])
        velocity_values.setdefault(key, []).append(float(item["velocity"]))
    mean_velocity = {key: sum(values) / len(values) for key, values in velocity_values.items()}
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    process = DiffusionProcess(GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear"))
    generator = torch.Generator().manual_seed(args.seed)
    baseline_losses, aligned_losses, video_losses, trajectory_losses, table_trajectory_losses = [], [], [], [], []
    baseline_temporal_errors, video_temporal_errors = [], []
    baseline_foreground_losses, video_foreground_losses = [], []
    baseline_background_losses, video_background_losses = [], []
    baseline_weighted_losses, video_weighted_losses = [], []
    aligned_foreground_losses, aligned_background_losses = [], []
    temporal_by_t = [[[], []] for _ in range(3)]
    temporal_by_region = [[[], []] for _ in range(2)]
    motion_names = ("stationary", "left_to_right", "right_to_left", "top_to_bottom",
                    "bottom_to_top", "diagonal_down_right", "diagonal_down_left")
    speed_names = ("slowly", "quickly")
    trajectory_by_condition = {}
    condition_mean_trajectory_errors = []
    velocity_variance_errors = []
    direction_vectors = {
        "stationary": (0.0, 0.0), "left_to_right": (1.0, 0.0), "right_to_left": (-1.0, 0.0),
        "top_to_bottom": (0.0, 1.0), "bottom_to_top": (0.0, -1.0),
        "diagonal_down_right": (1.0, 1.0), "diagonal_down_left": (-1.0, 1.0),
    }
    baseline.eval(); aligned_spatial_model.eval(); model.eval()
    with torch.no_grad():
        for clean, shapes, colors, sizes, motions, speeds, velocities, offsets in loader:
            if len(baseline_losses) >= args.samples:
                break
            size_condition = sizes if mc.get("num_size_classes", 0) > 0 else None
            t = torch.randint(0, 1000, (1,), generator=generator)
            shared_noise = torch.randn((1, clean.shape[1], 1, clean.shape[3], clean.shape[4]), generator=generator)
            independent_noise = torch.randn(clean.shape, generator=generator)
            noise = (args.noise_correlation ** 0.5) * shared_noise
            noise = noise.expand_as(clean) + ((1.0 - args.noise_correlation) ** 0.5) * independent_noise
            table_offsets = model.predict_offsets(motions, speeds)
            conditioned_offsets = model.predict_offsets(
                motions, speeds, velocity_px_per_frame=velocities,
                image_size=args.image_size, reference_frames=args.frames,
            )
            selected_offsets = conditioned_offsets if args.velocity_conditioning == "metadata" else table_offsets
            if args.noise_alignment == "canonical":
                noise_offsets = torch.round(selected_offsets * args.image_size)
                noise = model._translate_video(
                    noise.permute(0, 2, 1, 3, 4), noise_offsets,
                    to_canonical=False, mode="nearest",
                ).permute(0, 2, 1, 3, 4).contiguous()
            noisy, _ = process.q_sample(clean, t, noise)
            _, alpha, one_minus_alpha = process.schedule.get(t, clean.device)
            alpha_5d = alpha.view(-1, 1, 1, 1, 1)
            one_minus_5d = one_minus_alpha.view(-1, 1, 1, 1, 1)
            target = alpha_5d.sqrt() * noise - one_minus_5d.sqrt() * clean
            base_prediction = baseline(noisy, t, shapes, colors, size_labels=size_condition)
            video_prediction = model(
                noisy, t, shapes, colors, motions, speeds,
                velocity_px_per_frame=velocities if args.velocity_conditioning == "metadata" else None,
                reference_frames=args.frames,
                size_labels=size_condition,
            )
            aligned_prediction = aligned_spatial_model(
                noisy, t, shapes, colors, motions, speeds,
                velocity_px_per_frame=velocities if args.velocity_conditioning == "metadata" else None,
                reference_frames=args.frames,
                size_labels=size_condition,
            )
            predicted_offsets = selected_offsets
            trajectory_rmse = ((predicted_offsets - offsets) * args.image_size).square().mean().sqrt().item()
            table_trajectory_rmse = ((table_offsets - offsets) * args.image_size).square().mean().sqrt().item()
            trajectory_losses.append(trajectory_rmse)
            table_trajectory_losses.append(table_trajectory_rmse)
            condition = (motion_names[int(motions.item())], speed_names[int(speeds.item())])
            trajectory_by_condition.setdefault(condition, []).append(trajectory_rmse)
            mean_speed = mean_velocity[condition]
            direction = offsets.new_tensor(direction_vectors[condition[0]])
            frame_positions = torch.arange(args.frames, dtype=offsets.dtype) - (args.frames - 1) / 2
            expected_condition_offsets = (
                frame_positions.view(1, args.frames, 1)
                * mean_speed * direction.view(1, 1, 2) / args.image_size
            )
            predicted_offsets_px = table_offsets * args.image_size
            expected_offsets_px = expected_condition_offsets * args.image_size
            actual_offsets_px = offsets * args.image_size
            condition_mean_trajectory_errors.append(
                (predicted_offsets_px - expected_offsets_px).square().mean().sqrt().item()
            )
            velocity_variance_errors.append(
                (actual_offsets_px - expected_offsets_px).square().mean().sqrt().item()
            )
            baseline_losses.append((base_prediction - target).square().mean().item())
            aligned_losses.append((aligned_prediction - target).square().mean().item())
            video_losses.append((video_prediction - target).square().mean().item())
            foreground = (clean > -0.95).any(dim=1, keepdim=True).to(clean.dtype)
            weights = 1.0 + 4.0 * foreground
            temporal_foreground = torch.maximum(foreground[:, :, 1:], foreground[:, :, :-1])
            for model_index, (prediction, foreground_results, temporal_results) in enumerate((
                (base_prediction, baseline_foreground_losses, baseline_temporal_errors),
                (video_prediction, video_foreground_losses, video_temporal_errors),
            )):
                pixel_error = (prediction - target).square().mean(dim=1, keepdim=True)
                foreground_results.append(
                    (pixel_error * foreground).sum().div(foreground.sum().clamp_min(1)).item()
                )
                background_results = (
                    baseline_background_losses if model_index == 0 else video_background_losses
                )
                background = 1.0 - foreground
                background_results.append(
                    (pixel_error * background).sum().div(background.sum().clamp_min(1)).item()
                )
                weighted_results = baseline_weighted_losses if model_index == 0 else video_weighted_losses
                weighted_results.append(
                    ((pixel_error * weights).flatten(1).sum(1) / weights.flatten(1).sum(1)).mean().item()
                )
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
            aligned_pixel_error = (aligned_prediction - target).square().mean(dim=1, keepdim=True)
            aligned_foreground_losses.append(
                (aligned_pixel_error * foreground).sum().div(foreground.sum().clamp_min(1)).item()
            )
            background = 1.0 - foreground
            aligned_background_losses.append(
                (aligned_pixel_error * background).sum().div(background.sum().clamp_min(1)).item()
            )
    b, v = sum(baseline_losses) / len(baseline_losses), sum(video_losses) / len(video_losses)
    aligned = sum(aligned_losses) / len(aligned_losses)
    trajectory = sum(trajectory_losses) / len(trajectory_losses)
    table_trajectory = sum(table_trajectory_losses) / len(table_trajectory_losses)
    bfg = sum(baseline_foreground_losses) / len(baseline_foreground_losses)
    vfg = sum(video_foreground_losses) / len(video_foreground_losses)
    bbg = sum(baseline_background_losses) / len(baseline_background_losses)
    vbg = sum(video_background_losses) / len(video_background_losses)
    afg = sum(aligned_foreground_losses) / len(aligned_foreground_losses)
    abg = sum(aligned_background_losses) / len(aligned_background_losses)
    bfw = sum(baseline_weighted_losses) / len(baseline_weighted_losses)
    vfw = sum(video_weighted_losses) / len(video_weighted_losses)
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
    trajectory_condition_summary = ",".join(
        f"{motion}/{speed}:{sum(values) / len(values):.3f}px"
        for (motion, speed), values in sorted(trajectory_by_condition.items())
    )
    print(
        f"clips={len(baseline_losses)} v_mse={b:.6f}->{v:.6f} "
        f"aligned_no_temporal_v_mse={aligned:.6f} "
        f"foreground_region_v_mse={bfg:.6f}->{afg:.6f}->{vfg:.6f} "
        f"background_region_v_mse={bbg:.6f}->{abg:.6f}->{vbg:.6f} "
        f"foreground_weighted_v_mse={bfw:.6f}->{vfw:.6f} "
        f"temporal_gradient_rmse={bt:.6f}->{vt:.6f} "
        f"trajectory_conditioning={args.velocity_conditioning} "
        f"trajectory_vs_clip_metadata_rmse={trajectory:.3f}px "
        f"categorical_table_vs_clip_metadata_rmse={table_trajectory:.3f}px "
        f"trajectory_vs_training_condition_mean_rmse={sum(condition_mean_trajectory_errors) / len(condition_mean_trajectory_errors):.3f}px "
        f"clip_velocity_variation_floor_rmse={sum(velocity_variance_errors) / len(velocity_variance_errors):.3f}px "
        f"noise_correlation={args.noise_correlation:.2f} noise_alignment={args.noise_alignment} "
        f"trajectory_by_condition=[{trajectory_condition_summary}] "
        f"temporal_by_t=[{temporal_bucket_summary}] temporal_by_region=[{temporal_region_summary}]"
    )


if __name__ == "__main__":
    main()
