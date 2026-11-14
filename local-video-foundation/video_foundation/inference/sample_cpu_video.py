"""Generate a short CPU video from a compact color-conditioned checkpoint."""

import argparse
import os
import sys

import torch
import json
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from video_foundation.models.unet import UNet
from video_foundation.models.cpu_video_unet import CPUVideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shape", type=int, default=0, choices=(0, 1, 2), help="0 circle, 1 square, 2 triangle")
    parser.add_argument("--color", type=int, default=0, choices=(0, 1, 2, 3), help="0 red, 1 green, 2 blue, 3 yellow")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=None, help="Defaults to the checkpoint's training resolution")
    parser.add_argument("--steps", type=int, default=None,
                        help="DDIM steps (defaults to 25 for either motion mode)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise_correlation", type=float, default=1.0,
                        help="Shared component in initial frame noise: 1 identical, 0 independent")
    parser.add_argument("--metrics_json", default=None, help="Optional path for temporal consistency metrics")
    parser.add_argument("--motion_mode", choices=("rigid", "learned"), default="learned",
                        help="Rigid: denoise one frame then render the path; learned: predict motion in the video denoiser")
    parser.add_argument("--motion", choices=("stationary", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "diagonal_down_right", "diagonal_down_left"), default="left_to_right")
    parser.add_argument("--speed", choices=("slowly", "quickly"), default="quickly")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 25
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if not 0.0 <= args.noise_correlation <= 1.0:
        parser.error("--noise_correlation must be between 0 and 1")
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    mc = cfg["model"]
    num_shape_classes = int(mc.get("num_classes", 2))
    if args.shape >= num_shape_classes:
        raise ValueError(
            f"Shape label {args.shape} is unsupported by this checkpoint, which has "
            f"{num_shape_classes} shape classes. Train with a three-class shape curriculum to generate triangles."
        )
    image_size = args.image_size or mc.get("image_size", 64)
    spatial = UNet(in_channels=3, model_channels=mc["model_channels"], channel_mult=mc["channel_mult"],
                   num_res_blocks=mc["num_res_blocks"], num_classes=mc["num_classes"],
                   num_color_classes=mc["num_color_classes"])
    model = CPUVideoUNet(spatial, args.frames, freeze_spatial=True, use_motion=True)
    model_state = checkpoint["model_state_dict"]
    expected_state = model.state_dict()
    incompatible_temporal = [
        key for key, value in model_state.items()
        if key.startswith("temporal.") and key in expected_state
        and value.shape != expected_state[key].shape
    ]
    if incompatible_temporal:
        if args.motion_mode == "learned":
            raise ValueError(
                "This checkpoint predates the motion-canonical denoiser. "
                "Use a motion_canonical_v_prediction_v1 checkpoint for --motion_mode learned."
            )
        # Rigid mode uses the spatial prior and trajectory table only; old
        # temporal filters can be ignored safely for this inference path.
        model_state = {key: value for key, value in model_state.items() if not key.startswith("temporal.")}
        incompatible = model.load_state_dict(model_state, strict=False)
        unexpected = incompatible.unexpected_keys
        missing = [key for key in incompatible.missing_keys if not key.startswith("temporal.")]
        if unexpected or missing:
            raise RuntimeError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    else:
        model.load_state_dict(model_state)
    model.eval()
    schedule = GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear")
    generator = torch.Generator().manual_seed(args.seed)
    ts = torch.linspace(999, 0, args.steps, dtype=torch.long)
    shape = torch.tensor([args.shape])
    color = torch.tensor([args.color])
    motion_ids = {
        "stationary": 0, "left_to_right": 1, "right_to_left": 2,
        "top_to_bottom": 3, "bottom_to_top": 4,
        "diagonal_down_right": 5, "diagonal_down_left": 6,
    }
    motion = torch.tensor([motion_ids[args.motion]])
    speed = torch.tensor([{"slowly": 0, "quickly": 1}[args.speed]])
    with torch.no_grad():
        offsets = model.predict_offsets(motion, speed)[0].numpy() * image_size
    with torch.no_grad():
        if args.motion_mode == "rigid":
            # This curriculum contains rigid translations. Sampling one clean
            # canonical frame avoids T duplicate U-Net trajectories and gives
            # every output frame exactly the same object appearance.
            x = torch.randn((1, 3, image_size, image_size), generator=generator)
            for i, timestep in enumerate(ts):
                t = timestep.expand(1)
                v = model.spatial_unet(x, t, shape, color)
                _, alpha, one_minus = schedule.get(t, x.device)
                alpha = alpha.view(1, 1, 1, 1)
                one_minus = one_minus.view(1, 1, 1, 1)
                x0 = (alpha.sqrt() * x - one_minus.sqrt() * v).clamp(-1, 1)
                eps = (x - alpha.sqrt() * x0) / one_minus.sqrt().clamp_min(1e-8)
                if i + 1 < len(ts):
                    _, alpha_prev, one_minus_prev = schedule.get(ts[i + 1].expand(1), x.device)
                    alpha_prev = alpha_prev.view(1, 1, 1, 1)
                    one_minus_prev = one_minus_prev.view(1, 1, 1, 1)
                    x = alpha_prev.sqrt() * x0 + one_minus_prev.sqrt() * eps
                else:
                    x = x0
            x = x.unsqueeze(2).expand(-1, -1, args.frames, -1, -1).clone()
        else:
            # Correlate initial frame noise to trade diversity for identity.
            shared = torch.randn((1, 3, 1, image_size, image_size), generator=generator)
            independent = torch.randn((1, 3, args.frames, image_size, image_size), generator=generator)
            x = (args.noise_correlation ** 0.5) * shared.expand_as(independent)
            x = x + ((1.0 - args.noise_correlation) ** 0.5) * independent
            for i, timestep in enumerate(ts):
                t = timestep.expand(1)
                v = model(x, t, shape, color, motion, speed)
                _, alpha, one_minus = schedule.get(t, x.device)
                alpha = alpha.view(1, 1, 1, 1, 1)
                one_minus = one_minus.view(1, 1, 1, 1, 1)
                x0 = (alpha.sqrt() * x - one_minus.sqrt() * v).clamp(-1, 1)
                eps = (x - alpha.sqrt() * x0) / one_minus.sqrt().clamp_min(1e-8)
                if i + 1 < len(ts):
                    _, alpha_prev, one_minus_prev = schedule.get(ts[i + 1].expand(1), x.device)
                    alpha_prev = alpha_prev.view(1, 1, 1, 1, 1)
                    one_minus_prev = one_minus_prev.view(1, 1, 1, 1, 1)
                    x = alpha_prev.sqrt() * x0 + one_minus_prev.sqrt() * eps
                else:
                    x = x0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    frames = []
    rendered_pixels = []
    observed_centroids = []
    base_centroids = []
    foreground_areas = []
    applied_offsets = []
    crop_fractions = []

    def foreground_mask(rgb):
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        if args.color == 0:
            return (red > green * 1.35) & (red > blue * 1.35) & (red > 55)
        if args.color == 1:
            return (green > red * 1.2) & (green > blue * 1.2) & (green > 55)
        if args.color == 2:
            return (blue > red * 1.25) & (blue > green * 1.15) & (blue > 55)
        return (red > blue * 1.3) & (green > blue * 1.15) & (red > 55) & (green > 55)

    def largest_component(mask):
        """Discard isolated color-noise pixels before estimating shape bounds."""
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        largest = []
        for seed_y, seed_x in zip(*np.where(mask)):
            if visited[seed_y, seed_x]:
                continue
            visited[seed_y, seed_x] = True
            stack = [(int(seed_y), int(seed_x))]
            component = []
            while stack:
                y, x_coord = stack.pop()
                component.append((y, x_coord))
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x_coord - 1), min(width, x_coord + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(component) > len(largest):
                largest = component
        cleaned = np.zeros_like(mask, dtype=bool)
        if largest:
            ys, xs = zip(*largest)
            cleaned[np.asarray(ys), np.asarray(xs)] = True
        return cleaned

    def centroid(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (image_size - 1) / 2, (image_size - 1) / 2
        return float(xs.mean()), float(ys.mean())

    for frame_index, frame in enumerate(x[0].permute(1, 2, 3, 0)):
        pixels = ((frame.clamp(-1, 1) + 1) * 127.5).byte().numpy()
        image = Image.fromarray(pixels)
        mask = largest_component(foreground_mask(pixels))
        base_area = int(mask.sum())
        base_center = centroid(mask)
        base_centroids.append(base_center)
        foreground_areas.append(base_area)
        if args.motion_mode == "rigid":
            # The canonical-frame path needs an explicit final render. The
            # learned path already predicts translated x0 frames in-model.
            row_counts, col_counts = mask.sum(axis=1), mask.sum(axis=0)
            rows = np.flatnonzero(row_counts >= max(2, row_counts.max(initial=0) * 0.08))
            cols = np.flatnonzero(col_counts >= max(2, col_counts.max(initial=0) * 0.08))
            if len(rows) and len(cols):
                min_x, max_x = int(cols[0]), int(cols[-1])
                min_y, max_y = int(rows[0]), int(rows[-1])
                dx = float(np.clip(offsets[frame_index, 0], -min_x, image_size - 1 - max_x))
                dy = float(np.clip(offsets[frame_index, 1], -min_y, image_size - 1 - max_y))
            else:
                dx = dy = 0.0
            applied_offsets.append((dx, dy))
            moved = image.transform(image.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy),
                                    resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        else:
            dx = dy = 0.0
            applied_offsets.append((float(offsets[frame_index, 0]), float(offsets[frame_index, 1])))
            moved = image
        moved_pixels = np.asarray(moved)
        rendered_pixels.append(moved_pixels.astype(np.float32) / 127.5 - 1.0)
        moved_mask = largest_component(foreground_mask(moved_pixels))
        observed_centroids.append(centroid(moved_mask))
        moved_area = int(moved_mask.sum())
        crop_fractions.append(max(0.0, 1.0 - moved_area / max(base_area, 1)))
        frames.append(moved.resize((image_size * 8, image_size * 8), Image.Resampling.NEAREST))
    trajectory_center = np.mean(np.asarray(base_centroids), axis=0)
    expected_centroids = [
        (trajectory_center[0] + applied_offsets[i][0], trajectory_center[1] + applied_offsets[i][1])
        for i in range(args.frames)
    ]
    output_trajectory_rmse = float(np.mean([
        (observed_centroids[i][0] - expected_centroids[i][0]) ** 2
        + (observed_centroids[i][1] - expected_centroids[i][1]) ** 2
        for i in range(args.frames)
    ]) ** 0.5)
    raw_trajectory_rmse = float(np.mean([
        (base_centroids[i][0] - expected_centroids[i][0]) ** 2
        + (base_centroids[i][1] - expected_centroids[i][1]) ** 2
        for i in range(args.frames)
    ]) ** 0.5)
    output_centroid_span = [
        float(max(point[axis] for point in observed_centroids) - min(point[axis] for point in observed_centroids))
        for axis in range(2)
    ]
    rendered_video = np.stack(rendered_pixels)
    temporal_l1 = float(np.abs(rendered_video[1:] - rendered_video[:-1]).mean())
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=120, loop=0)
    metrics = {
        "frames": args.frames,
        "noise_correlation": args.noise_correlation,
        "motion_mode": args.motion_mode,
        "adjacent_frame_mean_abs_difference": temporal_l1,
        "motion": args.motion,
        "speed": args.speed,
        "image_size": image_size,
        "mean_foreground_crop_fraction": float(np.mean(crop_fractions)),
        "raw_trajectory_centroid_rmse_pixels": raw_trajectory_rmse,
        "output_trajectory_centroid_rmse_pixels": output_trajectory_rmse,
        "output_centroid_span_pixels": output_centroid_span,
        "mean_raw_foreground_pixels": float(np.mean(foreground_areas)),
        "predicted_offsets_pixels": [[float(x), float(y)] for x, y in applied_offsets],
        "mean_observed_centroid_pixels": [
            float(np.mean([point[0] for point in observed_centroids])),
            float(np.mean([point[1] for point in observed_centroids])),
        ],
    }
    if args.metrics_json:
        os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    print(f"Saved {len(frames)} {args.speed} {args.motion} frames at {image_size}px to {args.output}; "
          f"adjacent L1={temporal_l1:.5f}, output centroid RMSE={output_trajectory_rmse:.2f}px, "
          f"foreground crop={np.mean(crop_fractions):.1%}")


if __name__ == "__main__":
    main()
