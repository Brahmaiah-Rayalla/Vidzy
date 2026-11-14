"""Train a compact class/color conditioned video diffuser on CPU."""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from video_foundation.models.unet import UNet
from video_foundation.models.cpu_video_unet import CPUVideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


class ClipDataset(Dataset):
    """Loads clips with consistent palette and geometry conditions."""

    def __init__(self, root, frames=8, size=32):
        self.root, self.frames, self.size = root, frames, size
        self.videos = sorted(os.path.join(root, name) for name in os.listdir(root)
                             if name.startswith("video_") and os.path.isdir(os.path.join(root, name)))
        if not self.videos:
            raise ValueError(f"No video_* clips found in {root}")
        with open(os.path.join(root, "all_metadata.json"), encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        if len(self.metadata) != len(self.videos):
            raise ValueError(f"Metadata/video count mismatch: {len(self.metadata)} records for {len(self.videos)} clips")
        self.shape_ids = {"circle": 0, "square": 1, "triangle": 2}
        self.color_ids = {"red": 0, "green": 1, "blue": 2, "yellow": 3}
        self.motion_ids = {
            "stationary": 0, "left_to_right": 1, "right_to_left": 2,
            "top_to_bottom": 3, "bottom_to_top": 4,
            "diagonal_down_right": 5, "diagonal_down_left": 6,
        }
        self.speed_ids = {"slowly": 0, "quickly": 1}
        source_colors = {
            "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0),
            "yellow": (255, 255, 0), "white": (255, 255, 255), "black": (0, 0, 0),
            "orange": (255, 165, 0), "purple": (128, 0, 128), "cyan": (0, 255, 255),
            "pink": (255, 192, 203),
        }
        self.source_colors = {name: np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0 for name, rgb in source_colors.items()}
        self.foreground_rgb = {
            "red": (1.0, -1.0, -1.0),
            "green": (-1.0, 1.0, -1.0),
            "blue": (-1.0, -1.0, 1.0),
            "yellow": (1.0, 1.0, -1.0),
        }
        for item in self.metadata:
            if (item["shape"] not in self.shape_ids or item["color"] not in self.color_ids
                    or item["motion"] not in self.motion_ids or item["speed_adjective"] not in self.speed_ids):
                raise ValueError("Unsupported shape, color, or motion in CPU clip curriculum")
        self.cache = {}

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        if index not in self.cache:
            directory = self.videos[index]
            paths = sorted(name for name in os.listdir(directory) if name.lower().endswith(".png"))[:self.frames]
            if len(paths) != self.frames:
                raise ValueError(f"{directory} has {len(paths)} frames; requested {self.frames}")
            frames = []
            for name in paths:
                with Image.open(os.path.join(directory, name)) as im:
                    im = im.convert("RGB").resize((self.size, self.size), Image.Resampling.NEAREST)
                    frames.append(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)
            meta = self.metadata[index]
            # Remap legacy colors/backgrounds into the exact image-model palette.
            bg = (-1.0, -1.0, -1.0)
            fg = self.foreground_rgb[meta["color"]]
            rgb = np.stack(frames)
            source_bg = self.source_colors[meta["background_color"]]
            mask = np.max(np.abs(rgb - source_bg[None, None, None, :]), axis=-1) > 0.1
            rgb[:] = np.asarray(bg, dtype=np.float32)
            rgb[mask] = np.asarray(fg, dtype=np.float32)
            video = torch.from_numpy(rgb.transpose(3, 0, 1, 2).copy())
            positions = np.asarray(meta["trajectory"], dtype=np.float32)[:self.frames]
            offsets = (positions - positions.mean(axis=0, keepdims=True)) / float(meta["image_size"])
            offsets = torch.from_numpy(offsets.copy())
            self.cache[index] = (
                video,
                self.shape_ids[meta["shape"]],
                self.color_ids[meta["color"]],
                self.motion_ids[meta["motion"]],
                self.speed_ids[meta["speed_adjective"]],
                offsets,
            )
        return self.cache[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_video_checkpoint", default=None, help="Continue a compatible CPU video checkpoint")
    parser.add_argument("--reset_temporal", action="store_true",
                        help="Keep resumed motion embeddings/table but reinitialize the temporal filter")
    parser.add_argument("--trajectory_lr", type=float, default=0.01, help="Learning rate for the small trajectory table")
    parser.add_argument("--trajectory_only", action="store_true", help="Fit the learned motion path while freezing all denoising weights")
    parser.add_argument("--foreground_weight", type=float, default=4.0, help="Extra weight for shape pixels in v-prediction loss")
    parser.add_argument("--temporal_foreground_weight", type=float, default=None,
                        help="Extra temporal-loss weight on object pixels (defaults to --foreground_weight)")
    parser.add_argument("--temporal_weight", type=float, default=0.2, help="Weight for x0 temporal-gradient matching loss")
    parser.add_argument("--temporal_snr_power", type=float, default=1.0,
                        help="Downweight x0 temporal loss at noisy steps by alpha_bar raised to this power")
    args = parser.parse_args()
    if args.temporal_foreground_weight is None:
        args.temporal_foreground_weight = args.foreground_weight

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("diffusion", {}).get("prediction_type") != "v_prediction":
        raise ValueError("CPU video fine-tuning expects a v_prediction image checkpoint")
    model_cfg = config["model"]
    spatial = UNet(in_channels=3, model_channels=model_cfg["model_channels"],
                   channel_mult=model_cfg["channel_mult"], num_res_blocks=model_cfg["num_res_blocks"],
                   num_classes=model_cfg.get("num_classes", 2),
                   num_color_classes=model_cfg.get("num_color_classes", 4))
    spatial.load_state_dict(checkpoint["model_state_dict"])
    model = CPUVideoUNet(spatial, args.frames, freeze_spatial=True)
    if args.resume_video_checkpoint:
        video_checkpoint = torch.load(args.resume_video_checkpoint, map_location="cpu", weights_only=False)
        # Keep the image prior supplied by --init_checkpoint; only resume the
        # compatible compact motion modules from a video checkpoint. Temporal
        # layers are skipped when reset because their input semantics may have
        # changed with the motion-equivariant denoiser.
        video_state = {
            key: value for key, value in video_checkpoint["model_state_dict"].items()
            if not key.startswith("spatial_unet.")
            and not (args.reset_temporal and key.startswith("temporal."))
        }
        incompatible = model.load_state_dict(video_state, strict=False)
        allowed_prefixes = ("spatial_unet.", "temporal.") if args.reset_temporal else ("spatial_unet.",)
        allowed_missing = {key for key in incompatible.missing_keys if key.startswith(allowed_prefixes)}
        if "motion_velocity.weight" in incompatible.missing_keys:
            allowed_missing.add("motion_velocity.weight")
        allowed_unexpected = {key for key in incompatible.unexpected_keys if key.startswith("trajectory_head.")}
        exact_match = not incompatible.missing_keys and not incompatible.unexpected_keys
        known_migration = (
            set(incompatible.missing_keys) == allowed_missing
            and set(incompatible.unexpected_keys) == allowed_unexpected
        )
        if not exact_match and not known_migration:
            raise RuntimeError(f"Incompatible video checkpoint: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
        if args.reset_temporal:
            for layer in model.temporal:
                if isinstance(layer, torch.nn.Conv1d):
                    layer.reset_parameters()
            torch.nn.init.zeros_(model.temporal[-1].weight)
            torch.nn.init.zeros_(model.temporal[-1].bias)
            print("Reinitialized temporal filter for motion-canonical v-prediction semantics")
        print(f"Resumed temporal and motion weights from {args.resume_video_checkpoint}; kept spatial prior from {args.init_checkpoint}")
    trainable = (list(model.temporal.parameters()) + list(model.motion_embed.parameters())
                 + list(model.speed_embed.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr) if not args.trajectory_only else None
    trajectory_optimizer = torch.optim.AdamW(model.motion_velocity.parameters(), lr=args.trajectory_lr)
    data = ClipDataset(args.data_dir, args.frames, args.image_size)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    process = DiffusionProcess(GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear"))
    model.train()
    losses = []
    iterator = iter(loader)
    progress = tqdm(range(args.steps), desc="CPU video training")

    def save_video_checkpoint(path, step):
        torch.save({"model_state_dict": model.state_dict(), "step": step, "config": config,
                    "video": vars(args), "model_variant": "motion_canonical_v_prediction_v1",
                    "loss": float(np.mean(losses[-50:]))}, path)

    for step in progress:
        try:
            clean, shapes, colors, motions, speeds, offsets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            clean, shapes, colors, motions, speeds, offsets = next(iterator)
        predicted_offsets = model.predict_offsets(motions, speeds)
        # Train path parameters separately from denoising so pixel-scale motion
        # gradients cannot dominate or be clipped by the diffusion objective.
        trajectory_loss = ((predicted_offsets - offsets) * args.image_size).square().mean()
        if not args.trajectory_only:
            timesteps = torch.randint(0, 1000, (clean.shape[0],))
            noisy, noise = process.q_sample(clean, timesteps)
            _, alpha, one_minus_alpha = process.schedule.get(timesteps, clean.device)
            alpha = alpha.view(-1, 1, 1, 1, 1)
            one_minus_alpha = one_minus_alpha.view(-1, 1, 1, 1, 1)
            target_v = alpha.sqrt() * noise - one_minus_alpha.sqrt() * clean
            predicted_v = model(noisy, timesteps, shapes, colors, motions, speeds)
            foreground = (clean > -0.95).any(dim=1, keepdim=True).to(clean.dtype)
            spatial_weights = 1.0 + args.foreground_weight * foreground
            v_pixel_error = (predicted_v - target_v).square().mean(dim=1, keepdim=True)
            v_loss_per_clip = (v_pixel_error * spatial_weights).flatten(1).sum(1) / spatial_weights.flatten(1).sum(1)
            v_loss = v_loss_per_clip.mean()

            predicted_x0 = alpha.sqrt() * noisy - one_minus_alpha.sqrt() * predicted_v
            pred_delta = predicted_x0[:, :, 1:] - predicted_x0[:, :, :-1]
            clean_delta = clean[:, :, 1:] - clean[:, :, :-1]
            temporal_error = (pred_delta - clean_delta).square().mean(dim=1, keepdim=True)
            temporal_mask = torch.maximum(foreground[:, :, 1:], foreground[:, :, :-1])
            temporal_weights = 1.0 + args.temporal_foreground_weight * temporal_mask
            temporal_loss_per_clip = (
                (temporal_error * temporal_weights).flatten(1).sum(1)
                / temporal_weights.flatten(1).sum(1)
            )
            signal_weight = alpha.flatten(1).mean(1).clamp_min(0.0).pow(args.temporal_snr_power)
            temporal_loss = (temporal_loss_per_clip * signal_weight).mean()
            diffusion_loss = v_loss + args.temporal_weight * temporal_loss
            optimizer.zero_grad(set_to_none=True)
            diffusion_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        else:
            diffusion_loss = torch.zeros((), dtype=trajectory_loss.dtype)
        trajectory_optimizer.zero_grad(set_to_none=True)
        trajectory_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.motion_velocity.parameters(), 5.0)
        trajectory_optimizer.step()
        loss = diffusion_loss.detach() + trajectory_loss.detach() / (args.image_size ** 2)
        losses.append(loss.item())
        progress.set_postfix(loss=f"{loss.item():.5f}", denoise=f"{diffusion_loss.item():.4f}", traj_px=f"{trajectory_loss.item():.3f}", avg=f"{np.mean(losses[-50:]):.5f}")
        if (step + 1) % 500 == 0:
            save_video_checkpoint(os.path.join(args.output_dir, f"checkpoint_step_{step + 1:06d}.pt"), step + 1)
    save_video_checkpoint(os.path.join(args.output_dir, "checkpoint_final.pt"), args.steps)
    print(f"Saved checkpoint; final 50-step mean loss={np.mean(losses[-50:]):.6f}")


if __name__ == "__main__":
    main()
