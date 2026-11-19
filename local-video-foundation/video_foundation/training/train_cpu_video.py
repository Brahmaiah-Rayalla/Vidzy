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
                    or item["motion"] not in self.motion_ids or item["speed_adjective"] not in self.speed_ids
                    or "object_size" not in item):
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
                0 if int(meta["object_size"]) <= 20 else (1 if int(meta["object_size"]) <= 35 else 2),
                self.motion_ids[meta["motion"]],
                self.speed_ids[meta["speed_adjective"]],
                torch.tensor(float(meta["velocity"]), dtype=torch.float32),
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
    parser.add_argument("--finetune_spatial_output", action="store_true",
                        help="Unfreeze only the pretrained image U-Net output convolution during video training")
    parser.add_argument("--finetune_spatial_decoder", action="store_true",
                        help="Unfreeze the final full-resolution decoder blocks and output convolution")
    parser.add_argument("--spatial_lr", type=float, default=1e-6,
                        help="Learning rate for the optional spatial output convolution")
    parser.add_argument("--fit_trajectory_table", action="store_true",
                        help="Fit the constant-velocity motion table by closed-form least squares on the training trajectories")
    parser.add_argument("--foreground_weight", type=float, default=4.0, help="Extra weight for shape pixels in v-prediction loss")
    parser.add_argument("--temporal_foreground_weight", type=float, default=None,
                        help="Extra temporal-loss weight on object pixels (defaults to --foreground_weight)")
    parser.add_argument("--video_structure_weight", type=float, default=0.0,
                        help="Low-noise foreground-mask and area loss for each denoised video frame")
    parser.add_argument("--foreground_x0_weight", type=float, default=0.0,
                        help="SNR-weighted clean-pixel reconstruction loss over foreground pixels")
    parser.add_argument("--temporal_weight", type=float, default=0.2, help="Weight for x0 temporal-gradient matching loss")
    parser.add_argument("--canonical_temporal_weight", type=float, default=0.25,
                        help="Weight for appearance consistency after aligning frames to object-centered coordinates")
    parser.add_argument("--temporal_snr_power", type=float, default=1.0,
                        help="Downweight x0 temporal loss at noisy steps by alpha_bar raised to this power")
    parser.add_argument("--temporal_high_t_boost", type=float, default=0.0,
                        help="Extra bounded temporal-loss weight at high noise levels; quadratic in normalized timestep")
    parser.add_argument("--screen_trajectory_weight", type=float, default=0.0,
                        help="Weight for low-noise foreground-centroid matching in screen-space denoising")
    parser.add_argument("--canonical_noise", action="store_true",
                        help="Sample one canonical noise field and transport it along the clip trajectory")
    parser.add_argument("--no_canonicalize_motion", action="store_true",
                        help="Denoise frames in screen coordinates instead of aligning them to object-centered coordinates")
    parser.add_argument("--gate_temporal_foreground", action="store_true",
                        help="Gate the temporal residual by a palette-based foreground estimate in screen-space mode")
    parser.add_argument("--noise_correlation", type=float, default=0.0,
                        help="Shared fraction of canonical training noise (0 iid, 1 identical across frames)")
    args = parser.parse_args()
    if not 0.0 <= args.noise_correlation <= 1.0:
        parser.error("--noise_correlation must be between 0 and 1")
    if args.no_canonicalize_motion and args.canonical_noise:
        parser.error("--no_canonicalize_motion cannot be combined with --canonical_noise")
    if args.gate_temporal_foreground and not args.no_canonicalize_motion:
        parser.error("--gate_temporal_foreground requires --no_canonicalize_motion")
    if args.temporal_high_t_boost < 0:
        parser.error("--temporal_high_t_boost must be nonnegative")
    if args.video_structure_weight < 0:
        parser.error("--video_structure_weight must be nonnegative")
    if args.foreground_x0_weight < 0:
        parser.error("--foreground_x0_weight must be nonnegative")
    if args.spatial_lr <= 0:
        parser.error("--spatial_lr must be positive")
    if args.finetune_spatial_output and args.finetune_spatial_decoder:
        parser.error("Choose either --finetune_spatial_output or --finetune_spatial_decoder")
    if (args.finetune_spatial_output or args.finetune_spatial_decoder) and args.trajectory_only:
        parser.error("Spatial fine-tuning cannot be combined with --trajectory_only")
    if args.screen_trajectory_weight < 0:
        parser.error("--screen_trajectory_weight must be nonnegative")
    if args.screen_trajectory_weight > 0 and not args.no_canonicalize_motion:
        parser.error("--screen_trajectory_weight requires --no_canonicalize_motion")
    if args.temporal_foreground_weight is None:
        args.temporal_foreground_weight = args.foreground_weight
    if args.no_canonicalize_motion and args.canonical_temporal_weight > 0:
        print("Disabling canonical temporal loss because motion canonicalization is off")
        args.canonical_temporal_weight = 0.0

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
                   num_color_classes=model_cfg.get("num_color_classes", 4),
                   num_size_classes=model_cfg.get("num_size_classes", 0),
                   spatial_size_conditioning=model_cfg.get("spatial_size_conditioning", False),
                   high_resolution_refinement=model_cfg.get("high_resolution_refinement", False),
                   size_embedding_scale=model_cfg.get("size_embedding_scale", 1.0),
                   middle_dilation=int(model_cfg.get("middle_dilation", 1)))
    spatial.load_state_dict(checkpoint["model_state_dict"])
    model = CPUVideoUNet(
        spatial, args.frames, freeze_spatial=True,
        canonicalize_motion=not args.no_canonicalize_motion,
        gate_temporal_foreground=args.gate_temporal_foreground,
    )
    decoder_indices = range(max(0, len(model.spatial_unet.up_blocks) - 2),
                            len(model.spatial_unet.up_blocks))
    resumed_spatial_prefixes = ("spatial_unet.conv_out.",)
    if args.finetune_spatial_decoder:
        resumed_spatial_prefixes += tuple(
            f"spatial_unet.up_blocks.{index}." for index in decoder_indices
        )
    base_step = 0
    if args.resume_video_checkpoint:
        video_checkpoint = torch.load(args.resume_video_checkpoint, map_location="cpu", weights_only=False)
        base_step = int(video_checkpoint.get("step", 0))
        # Keep the image prior supplied by --init_checkpoint; only resume the
        # compatible compact motion modules from a video checkpoint. Temporal
        # layers are skipped when reset because their input semantics may have
        # changed with the motion-equivariant denoiser.
        video_state = {
            key: value for key, value in video_checkpoint["model_state_dict"].items()
            if (not key.startswith("spatial_unet.")
                or (args.finetune_spatial_output and key.startswith("spatial_unet.conv_out."))
                or (args.finetune_spatial_decoder and key.startswith(resumed_spatial_prefixes)))
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
            alignment = "screen-space" if args.no_canonicalize_motion else "motion-canonical"
            print(f"Reinitialized temporal filter for {alignment} v-prediction semantics")
        print(f"Resumed temporal and motion weights from {args.resume_video_checkpoint}; kept spatial prior from {args.init_checkpoint}")
    spatial_trainable = []
    spatial_modules = []
    if args.finetune_spatial_output:
        spatial_modules = [model.spatial_unet.conv_out]
    elif args.finetune_spatial_decoder:
        spatial_modules = [model.spatial_unet.up_blocks[index] for index in decoder_indices]
        spatial_modules.append(model.spatial_unet.conv_out)
    if spatial_modules:
        for module in spatial_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                spatial_trainable.append(parameter)
        print(f"Fine-tuning {sum(parameter.numel() for parameter in spatial_trainable):,} spatial decoder parameters at lr={args.spatial_lr:g}")
    video_trainable = (list(model.temporal.parameters()) + list(model.motion_embed.parameters())
                       + list(model.speed_embed.parameters()))
    trainable = video_trainable + spatial_trainable
    optimizer_groups = [{"params": video_trainable, "lr": args.lr}]
    if spatial_trainable:
        optimizer_groups.append({"params": spatial_trainable, "lr": args.spatial_lr})
    optimizer = torch.optim.AdamW(optimizer_groups) if not args.trajectory_only else None
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
                    "video": vars(args),
                    "model_variant": ("motion_screen_v_prediction_v1" if args.no_canonicalize_motion
                                      else "motion_canonical_v_prediction_v1"),
                    "loss": float(np.mean(losses[-50:])) if losses else 0.0}, path)

    if args.fit_trajectory_table:
        # The motion head is exactly linear in normalized frame position, so a
        # per-condition least-squares fit is both cheaper and more accurate
        # than noisy minibatch updates of these 14 two-dimensional velocities.
        frame_positions = torch.linspace(-0.5, 0.5, args.frames)
        numerators = torch.zeros_like(model.motion_velocity.weight)
        denominators = torch.zeros(model.motion_velocity.num_embeddings)
        for index in range(len(data)):
            _, _, _, _, motion, speed, _, offsets = data[index]
            row = int(motion) * 2 + int(speed)
            numerators[row] += (offsets * frame_positions[:, None]).sum(dim=0)
            denominators[row] += frame_positions.square().sum()
        fitted = numerators / denominators[:, None].clamp_min(1e-8)
        with torch.no_grad():
            model.motion_velocity.weight.copy_(fitted.clamp(-0.4, 0.4))
        print(f"Closed-form motion-table fit: {len(data)} training clips, "
              f"max velocity={model.motion_velocity.weight.abs().max().item():.4f} normalized units")
        save_video_checkpoint(os.path.join(args.output_dir, "checkpoint_final.pt"), base_step)
        return

    for step in progress:
        try:
            clean, shapes, colors, sizes, motions, speeds, velocities, offsets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            clean, shapes, colors, sizes, motions, speeds, velocities, offsets = next(iterator)
        predicted_offsets = model.predict_offsets(motions, speeds)
        conditioned_offsets = model.predict_offsets(
            motions, speeds, velocity_px_per_frame=velocities,
            image_size=args.image_size, reference_frames=args.frames,
        )
        # Train path parameters separately from denoising so pixel-scale motion
        # gradients cannot dominate or be clipped by the diffusion objective.
        trajectory_loss = ((predicted_offsets - offsets) * args.image_size).square().mean()
        if not args.trajectory_only:
            timesteps = torch.randint(0, 1000, (clean.shape[0],))
            if args.canonical_noise:
                # The denoiser aligns x_t to object-centered coordinates.
                # Transporting the sampled noise with the same path makes the
                # canonicalized noise distribution match the sampler.
                offsets_px = torch.round(conditioned_offsets.detach() * args.image_size)
                independent_noise = torch.randn_like(clean.permute(0, 2, 1, 3, 4))
                shared_noise = torch.randn_like(independent_noise[:, :1])
                canonical_noise = (args.noise_correlation ** 0.5) * shared_noise.expand_as(independent_noise)
                canonical_noise = canonical_noise + ((1.0 - args.noise_correlation) ** 0.5) * independent_noise
                noise = model._translate_video(
                    canonical_noise, offsets_px, to_canonical=False, mode="nearest"
                ).permute(0, 2, 1, 3, 4).contiguous()
            else:
                noise = None
            noisy, noise = process.q_sample(clean, timesteps, noise)
            _, alpha, one_minus_alpha = process.schedule.get(timesteps, clean.device)
            alpha = alpha.view(-1, 1, 1, 1, 1)
            one_minus_alpha = one_minus_alpha.view(-1, 1, 1, 1, 1)
            target_v = alpha.sqrt() * noise - one_minus_alpha.sqrt() * clean
            predicted_v = model(
                noisy, timesteps, shapes, colors, motions, speeds,
                velocity_px_per_frame=velocities, reference_frames=args.frames,
                size_labels=sizes,
            )
            foreground = (clean > -0.95).any(dim=1, keepdim=True).to(clean.dtype)
            spatial_weights = 1.0 + args.foreground_weight * foreground
            v_pixel_error = (predicted_v - target_v).square().mean(dim=1, keepdim=True)
            v_loss_per_clip = (v_pixel_error * spatial_weights).flatten(1).sum(1) / spatial_weights.flatten(1).sum(1)
            v_loss = v_loss_per_clip.mean()

            predicted_x0 = alpha.sqrt() * noisy - one_minus_alpha.sqrt() * predicted_v
            foreground_x0_loss = torch.zeros_like(v_loss)
            if args.foreground_x0_weight > 0:
                foreground_x0_error = torch.nn.functional.smooth_l1_loss(
                    predicted_x0, clean, reduction="none", beta=0.1
                ).mean(dim=1, keepdim=True)
                foreground_x0_per_clip = (
                    (foreground_x0_error * foreground).flatten(1).sum(1)
                    / foreground.flatten(1).sum(1).clamp_min(1.0)
                )
                x0_confidence = alpha.flatten(1).mean(1).clamp(0.0, 1.0).sqrt()
                foreground_x0_loss = (
                    (foreground_x0_per_clip * x0_confidence).sum()
                    / x0_confidence.sum().clamp_min(1e-6)
                )
            video_structure_loss = torch.zeros_like(v_loss)
            if args.video_structure_weight > 0:
                predicted_foreground = torch.sigmoid(predicted_x0.max(dim=1).values * 4.0)
                target_foreground = foreground.squeeze(1)
                predicted_area = predicted_foreground.flatten(2).sum(dim=2)
                target_area = target_foreground.flatten(2).sum(dim=2)
                intersection = (predicted_foreground * target_foreground).flatten(2).sum(dim=2)
                dice_loss = 1.0 - (2.0 * intersection + 1.0) / (predicted_area + target_area + 1.0)
                area_loss = torch.nn.functional.smooth_l1_loss(
                    torch.log1p(predicted_area), torch.log1p(target_area), reduction="none"
                )
                per_frame_structure = dice_loss + 0.25 * area_loss
                structure_confidence = (
                    alpha.flatten(1).mean(1).clamp(0.0, 1.0).sqrt()
                    .view(-1, 1).expand(-1, clean.shape[2]).reshape(-1)
                )
                video_structure_loss = (
                    (per_frame_structure.reshape(-1) * structure_confidence).sum()
                    / structure_confidence.sum().clamp_min(1e-6)
                )
            pred_delta = predicted_x0[:, :, 1:] - predicted_x0[:, :, :-1]
            clean_delta = clean[:, :, 1:] - clean[:, :, :-1]
            temporal_error = (pred_delta - clean_delta).square().mean(dim=1, keepdim=True)
            temporal_mask = torch.maximum(foreground[:, :, 1:], foreground[:, :, :-1])
            temporal_weights = 1.0 + args.temporal_foreground_weight * temporal_mask
            temporal_loss_per_clip = (
                (temporal_error * temporal_weights).flatten(1).sum(1)
                / temporal_weights.flatten(1).sum(1)
            )
            normalized_timestep = timesteps.to(clean.dtype) / 999.0
            high_t_weight = 1.0 + args.temporal_high_t_boost * normalized_timestep.square()
            signal_weight = (
                alpha.flatten(1).mean(1).clamp_min(0.0).pow(args.temporal_snr_power)
                * high_t_weight
            )
            temporal_loss = (temporal_loss_per_clip * signal_weight).mean()
            screen_trajectory_loss = torch.zeros_like(temporal_loss)
            if args.screen_trajectory_weight > 0:
                predicted_foreground = torch.sigmoid(predicted_x0.max(dim=1).values * 4.0)
                target_foreground = foreground.squeeze(1)
                height, width = clean.shape[-2:]
                coordinate_x = torch.arange(width, device=clean.device, dtype=clean.dtype).view(1, 1, 1, width)
                coordinate_y = torch.arange(height, device=clean.device, dtype=clean.dtype).view(1, 1, height, 1)
                predicted_mass = predicted_foreground.sum(dim=(-2, -1)).clamp_min(1e-4)
                target_mass = target_foreground.sum(dim=(-2, -1)).clamp_min(1.0)
                predicted_center_x = (predicted_foreground * coordinate_x).sum(dim=(-2, -1)) / predicted_mass
                predicted_center_y = (predicted_foreground * coordinate_y).sum(dim=(-2, -1)) / predicted_mass
                target_center_x = (target_foreground * coordinate_x).sum(dim=(-2, -1)) / target_mass
                target_center_y = (target_foreground * coordinate_y).sum(dim=(-2, -1)) / target_mass
                predicted_centers = torch.stack((predicted_center_x, predicted_center_y), dim=-1)
                target_centers = torch.stack((target_center_x, target_center_y), dim=-1)
                centroid_error = ((predicted_centers - target_centers) / max(height, width)).square().mean(dim=(1, 2))
                centroid_confidence = alpha.flatten(1).mean(1).clamp(0.0, 1.0).sqrt()
                screen_trajectory_loss = (
                    (centroid_error * centroid_confidence).sum()
                    / centroid_confidence.sum().clamp_min(1e-6)
                )
            # A rigid object should have a constant appearance in its
            # motion-canonical coordinates. Aligning x0 directly targets
            # flicker while preserving the pixel-domain temporal objective.
            if args.canonical_temporal_weight > 0 and model.use_motion and model.canonicalize_motion:
                offsets_px = torch.round(conditioned_offsets.detach() * args.image_size)
                predicted_canonical_x0 = model._translate_video(
                    predicted_x0.permute(0, 2, 1, 3, 4), offsets_px,
                    to_canonical=True, mode="nearest",
                ).permute(0, 2, 1, 3, 4)
                clean_canonical = model._translate_video(
                    clean.permute(0, 2, 1, 3, 4), offsets_px,
                    to_canonical=True, mode="nearest",
                ).permute(0, 2, 1, 3, 4)
                canonical_delta_error = (
                    (predicted_canonical_x0[:, :, 1:] - predicted_canonical_x0[:, :, :-1])
                    - (clean_canonical[:, :, 1:] - clean_canonical[:, :, :-1])
                ).square().mean(dim=1, keepdim=True)
                canonical_foreground = (clean_canonical > -0.95).any(dim=1, keepdim=True).to(clean.dtype)
                canonical_mask = torch.maximum(canonical_foreground[:, :, 1:], canonical_foreground[:, :, :-1])
                canonical_weights = 1.0 + args.temporal_foreground_weight * canonical_mask
                canonical_loss_per_clip = (
                    (canonical_delta_error * canonical_weights).flatten(1).sum(1)
                    / canonical_weights.flatten(1).sum(1)
                )
                canonical_temporal_loss = (canonical_loss_per_clip * signal_weight).mean()
            else:
                canonical_temporal_loss = torch.zeros_like(temporal_loss)
            diffusion_loss = (v_loss + args.temporal_weight * temporal_loss
                              + args.canonical_temporal_weight * canonical_temporal_loss
                              + args.screen_trajectory_weight * screen_trajectory_loss
                              + args.video_structure_weight * video_structure_loss
                              + args.foreground_x0_weight * foreground_x0_loss)
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
            cumulative_step = base_step + step + 1
            save_video_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_step_{cumulative_step:06d}.pt"),
                cumulative_step,
            )
    save_video_checkpoint(os.path.join(args.output_dir, "checkpoint_final.pt"), base_step + args.steps)
    print(f"Saved checkpoint; final 50-step mean loss={np.mean(losses[-50:]):.6f}")


if __name__ == "__main__":
    main()
