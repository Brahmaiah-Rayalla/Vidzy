"""Generate two model-conditioned circle sprites and simulate their CPU collision."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.inference.shape_rendering import render_canonical_shape
from video_foundation.models.unet import UNet


ROOT = Path(__file__).resolve().parents[2]
COLOR_IDS = {"red": 0, "green": 1, "blue": 2, "yellow": 3}
PALETTE = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}


def load_spatial_model(checkpoint_path: Path, threads: int):
    """Load only the spatial denoiser from a CPU video checkpoint."""
    torch.set_num_threads(threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]["model"]
    model = UNet(
        in_channels=model_config.get("in_channels", 3),
        model_channels=model_config["model_channels"],
        channel_mult=model_config["channel_mult"],
        num_res_blocks=model_config["num_res_blocks"],
        num_classes=model_config["num_classes"],
        num_color_classes=model_config["num_color_classes"],
        num_size_classes=model_config.get("num_size_classes", 0),
        spatial_size_conditioning=model_config.get("spatial_size_conditioning", False),
        high_resolution_refinement=model_config.get("high_resolution_refinement", False),
        size_embedding_scale=model_config.get("size_embedding_scale", 1.0),
        middle_dilation=int(model_config.get("middle_dilation", 1)),
    )
    state = checkpoint["model_state_dict"]
    spatial_state = {
        key.removeprefix("spatial_unet."): value
        for key, value in state.items() if key.startswith("spatial_unet.")
    }
    if not spatial_state:
        spatial_state = state
    model.load_state_dict(spatial_state)
    model.eval()
    schedule = GaussianDiffusionSchedule(
        num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear",
    )
    return model, model_config, schedule


def palette_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    red, green, blue = (rgb[..., channel] for channel in range(3))
    if color == "red":
        return (red > green * 1.35) & (red > blue * 1.35) & (red > 55)
    if color == "green":
        return (green > red * 1.2) & (green > blue * 1.2) & (green > 55)
    if color == "blue":
        return (blue > red * 1.25) & (blue > green * 1.15) & (blue > 55)
    return (red > blue * 1.3) & (green > blue * 1.15) & (red > 55) & (green > 55)


def largest_component(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    largest = []
    for seed_y, seed_x in zip(*np.where(mask)):
        if visited[seed_y, seed_x]:
            continue
        visited[seed_y, seed_x] = True
        stack, component = [(int(seed_y), int(seed_x))], []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(component) > len(largest):
            largest = component
    clean = np.zeros_like(mask, dtype=bool)
    if largest:
        ys, xs = zip(*largest)
        clean[np.asarray(ys), np.asarray(xs)] = True
    return clean


def generate_sprite(model, model_config, schedule, color: str, size: str,
                    seed: int, steps: int) -> tuple[Image.Image, tuple[int, int], float]:
    """Use the trained denoiser for object localization, then refine its circle mask."""
    started = time.perf_counter()
    image_size = int(model_config["image_size"])
    size_ids = {"small": 0, "medium": 1, "large": 2}
    if int(model_config.get("num_size_classes", 0)) < 3:
        raise ValueError("The selected checkpoint must support small, medium, and large size labels")
    x = torch.randn(
        (1, 3, image_size, image_size),
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    shape_label = torch.tensor([0], dtype=torch.long)
    color_label = torch.tensor([COLOR_IDS[color]], dtype=torch.long)
    size_label = torch.tensor([size_ids[size]], dtype=torch.long)
    timesteps = torch.linspace(999, 0, steps, dtype=torch.long)
    with torch.no_grad():
        for index, timestep in enumerate(timesteps):
            t = timestep.expand(1)
            velocity = model(x, t, shape_label, color_label, size_label)
            _, alpha, one_minus = schedule.get(t, x.device)
            alpha = alpha.view(1, 1, 1, 1)
            one_minus = one_minus.view(1, 1, 1, 1)
            x0 = (alpha.sqrt() * x - one_minus.sqrt() * velocity).clamp(-1, 1)
            eps = (x - alpha.sqrt() * x0) / one_minus.sqrt().clamp_min(1e-8)
            if index + 1 < len(timesteps):
                _, alpha_prev, one_minus_prev = schedule.get(timesteps[index + 1].expand(1), x.device)
                alpha_prev = alpha_prev.view(1, 1, 1, 1)
                one_minus_prev = one_minus_prev.view(1, 1, 1, 1)
                x = alpha_prev.sqrt() * x0 + one_minus_prev.sqrt() * eps
            else:
                x = x0

    rgb = (((x[0].clamp(-1, 1).permute(1, 2, 0).numpy() + 1.0) * 127.5)
           .astype(np.uint8))
    mask = largest_component(palette_mask(rgb, color))
    ys, xs = np.where(mask)
    center = ((float(xs.mean()), float(ys.mean())) if len(xs)
              else ((image_size - 1) / 2.0, (image_size - 1) / 2.0))
    extent = max(1, round((15, 29, 42)[size_ids[size]] * image_size / 64))
    display_scale, render_scale = 8, 16
    high_center = tuple((coordinate + 0.5) * render_scale - 0.5 for coordinate in center)
    high_mask = render_canonical_shape(
        0, high_center, extent * render_scale, image_size * render_scale,
    )
    alpha = Image.fromarray(high_mask.astype(np.uint8) * 255).resize(
        (image_size * display_scale, image_size * display_scale), Image.Resampling.LANCZOS,
    )
    bounds = alpha.getbbox()
    if bounds is None:
        raise ValueError("Canonical circle rasterization produced an empty sprite")
    alpha = alpha.crop(bounds)
    rgba = Image.new("RGBA", alpha.size, PALETTE[color] + (0,))
    rgba.putalpha(alpha)
    return rgba, (image_size * display_scale, image_size * display_scale), time.perf_counter() - started


def advance_two_equal_circles(x_left: float, x_right: float, velocity_left: float,
                              velocity_right: float, dt: float, left_wall: float,
                              right_wall: float, contact_distance: float):
    """Advance two equal-mass circles with elastic circle and wall collisions."""
    remaining = dt
    for _ in range(8):
        if remaining <= 1e-10:
            break
        events = []
        if velocity_left < -1e-10:
            events.append(((left_wall - x_left) / velocity_left, "left_wall"))
        if velocity_right > 1e-10:
            events.append(((right_wall - x_right) / velocity_right, "right_wall"))
        gap = x_right - x_left - contact_distance
        relative_speed = velocity_left - velocity_right
        if relative_speed > 1e-10:
            events.append((max(0.0, gap) / relative_speed, "circles"))
        valid = [(max(0.0, event_time), kind) for event_time, kind in events
                 if event_time >= -1e-9 and event_time <= remaining + 1e-9]
        if not valid:
            x_left += velocity_left * remaining
            x_right += velocity_right * remaining
            break
        event_time, event_kind = min(valid, key=lambda item: item[0])
        x_left += velocity_left * event_time
        x_right += velocity_right * event_time
        remaining -= event_time
        if event_kind == "left_wall":
            x_left = left_wall
            velocity_left = -velocity_left
        elif event_kind == "right_wall":
            x_right = right_wall
            velocity_right = -velocity_right
        else:
            midpoint = (x_left + x_right) / 2.0
            x_left = midpoint - contact_distance / 2.0
            x_right = midpoint + contact_distance / 2.0
            velocity_left, velocity_right = velocity_right, velocity_left
        # Guard against repeatedly resolving the same zero-time contact.
        if event_time <= 1e-10 and event_kind == "circles":
            remaining = max(0.0, remaining - 1e-9)
    return x_left, x_right, velocity_left, velocity_right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(
        ROOT / "checkpoints" / "cpu_video_canonical_position_aug24_v37" / "checkpoint_step_008500.pt"
    ))
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--left-color", choices=tuple(COLOR_IDS), default="red")
    parser.add_argument("--right-color", choices=tuple(COLOR_IDS), default="red")
    parser.add_argument("--duration-seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--collision-time", type=float, default=None,
                        help="First head-on collision time; defaults to half the clip duration")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not 0 < args.duration_seconds <= 6:
        parser.error("--duration-seconds must be greater than 0 and at most 6")
    if not 1 <= args.fps <= 30:
        parser.error("--fps must be between 1 and 30")
    if args.steps < 2 or args.threads < 1:
        parser.error("--steps must be at least 2 and --threads must be positive")
    collision_time = (args.duration_seconds / 2.0 if args.collision_time is None
                      else args.collision_time)
    if not 0 < collision_time < args.duration_seconds:
        parser.error("--collision-time must be between zero and the clip duration")

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_count = max(2, int(round(args.duration_seconds * args.fps)))
    started = time.perf_counter()
    model_started = time.perf_counter()
    model, model_config, schedule = load_spatial_model(checkpoint, args.threads)
    model_load_seconds = time.perf_counter() - model_started
    sprite_times = 0.0

    sprites = {}
    canvas_size = (int(model_config["image_size"]) * 8, int(model_config["image_size"]) * 8)
    for color in dict.fromkeys((args.left_color, args.right_color)):
        sprite, canvas_size, elapsed = generate_sprite(
            model, model_config, schedule, color, args.size,
            args.seed + COLOR_IDS[color], args.steps,
        )
        sprites[color] = sprite
        sprite_times += elapsed

    left_sprite, right_sprite = sprites[args.left_color], sprites[args.right_color]
    left_radius = max(left_sprite.size) / 2.0
    right_radius = max(right_sprite.size) / 2.0
    margin = 12.0
    base_width, base_height = canvas_size
    minimum_width = math.ceil(2.0 * (left_radius + right_radius) + 2.0 * margin + 120.0)
    width = max(base_width, minimum_width)
    height = max(base_height, math.ceil(2.0 * max(left_radius, right_radius) + 2.0 * margin))
    left_wall = left_radius + margin
    right_wall = width - right_radius - margin
    contact_distance = left_radius + right_radius
    gap = right_wall - left_wall - contact_distance
    if gap <= 0:
        parser.error("The selected sprites do not fit in the collision canvas")
    speed = gap / (2.0 * collision_time)
    x_left, x_right = left_wall, right_wall
    velocity_left, velocity_right = speed, -speed
    y = (height - 1) / 2.0
    output_frames = []
    dt = 1.0 / args.fps

    for index in range(frames_count):
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        for sprite, center_x in ((left_sprite, x_left), (right_sprite, x_right)):
            sprite_center = ((sprite.width - 1) / 2.0, (sprite.height - 1) / 2.0)
            canvas.alpha_composite(sprite, dest=(
                round(center_x - sprite_center[0]), round(y - sprite_center[1])
            ))
        output_frames.append(canvas.convert("RGB"))
        x_left, x_right, velocity_left, velocity_right = advance_two_equal_circles(
            x_left, x_right, velocity_left, velocity_right, dt,
            left_wall, right_wall, contact_distance,
        )

    frame_duration_ms = args.duration_seconds * 1000.0 / frames_count
    boundaries = [int(round(i * frame_duration_ms / 10.0)) * 10 for i in range(frames_count + 1)]
    durations = [max(10, boundaries[i + 1] - boundaries[i]) for i in range(frames_count)]
    output_frames[0].save(output, save_all=True, append_images=output_frames[1:],
                           duration=durations, loop=0, optimize=False)
    preview_index = min(frames_count - 1, int(round(collision_time * args.fps)))
    preview_path = output.with_name(f"{output.stem}_contact.png")
    output_frames[preview_index].save(preview_path)

    metrics = {
        "output": str(output),
        "checkpoint": str(checkpoint),
        "frames": frames_count,
        "fps": args.fps,
        "duration_seconds": sum(durations) / 1000.0,
        "resolution": [width, height],
        "size": args.size,
        "colors": [args.left_color, args.right_color],
        "collision_time_seconds": collision_time,
        "initial_speed_px_per_second": speed,
        "physics": "equal-mass 1D elastic collision with wall reflections",
        "model_load_seconds": model_load_seconds,
        "sprite_generation_seconds": sprite_times,
        "total_seconds": time.perf_counter() - started,
        "contact_preview": str(preview_path),
        "motion_is_analytic": True,
    }
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {frames_count} frames ({metrics['duration_seconds']:.2f}s) at {width}x{height}; "
          f"first collision at {collision_time:.2f}s; model sprite sampling "
          f"{sprite_times:.2f}s; total {metrics['total_seconds']:.2f}s on CPU/{args.threads} threads")


if __name__ == "__main__":
    main()
