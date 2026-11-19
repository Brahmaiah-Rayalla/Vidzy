"""Sample L34 geometry and render a CPU-only, event-driven circle collision."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from video_foundation.inference.trajectory_profile import (
    TrajectoryProfileNet, predict_profile, sample_profile_parameters, target_profile_scale,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEOMETRY_CHECKPOINT = (
    ROOT / "experiments_stage3_2_9B" / "phase2L_training_repair" /
    "extended_250_steps" / "l34_structured_geometry_diffusion" / "checkpoint_final.pt"
)
DEFAULT_MOTION_CHECKPOINT = (
    ROOT / "checkpoints" / "cpu_video_canonical_position_aug24_v37" / "checkpoint_step_008500.pt"
)
DEFAULT_PROFILE_CHECKPOINT = (
    ROOT / "experiments_stage3_2_9B" / "phase2L_training_repair" /
    "extended_250_steps" / "l39_learned_speed_profiles" / "trajectory_profile.pt"
)
DATA_MANIFEST = (
    ROOT / "experiments_stage3_2_9B" / "phase2L_training_repair" /
    "extended_250_steps" / "l32_cpu_factorized_salience_pilot" / "data" / "manifest.jsonl"
)
CALIBRATION_MANIFEST = DATA_MANIFEST
COLOR_NAMES = ("red", "blue", "yellow")
COLORS = ((220, 30, 30), (30, 80, 220), (230, 200, 30))
T = 1000


class GeometryDenoiser(nn.Module):
    """The compact 5D conditional denoiser used by L34."""

    def __init__(self):
        super().__init__()
        self.time_embed = nn.Embedding(T, 16)
        self.color_embed = nn.Embedding(3, 8)
        self.shape_embed = nn.Embedding(2, 8)
        self.input = nn.Linear(37, 128)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(128, 128), nn.SiLU()) for _ in range(3)]
        )
        self.output = nn.Linear(128, 5)

    def forward(self, noisy, timesteps, class_ids):
        condition = torch.cat((
            self.time_embed(timesteps), self.color_embed(class_ids // 2),
            self.shape_embed(class_ids % 2),
        ), dim=1)
        hidden = F.silu(self.input(torch.cat((noisy, condition), dim=1)))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


def load_geometry_model(checkpoint_path: Path, threads: int):
    torch.set_num_threads(threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GeometryDenoiser()
    model.load_state_dict(checkpoint["ema_model"])
    model.eval()
    betas = torch.linspace(0.0001, 0.02, T)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    return model, alpha_bar, checkpoint


def _quantile_map(values, reference, target, rng):
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    target = np.sort(np.asarray(target, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    ranks = left + rng.random(values.shape) * (right - left)
    quantiles = np.clip((ranks + 0.5) / (len(reference) + 1.0), 0.0, 1.0)
    indices = np.minimum((quantiles * len(target)).astype(np.int64), len(target) - 1)
    return target[indices]


def apply_geometry_calibration(vector, calibration_row, seed):
    """Quantile-map center and radius marginals using only the L34 train split."""
    reference = np.asarray(calibration_row["reference_vectors"], dtype=np.float64)
    target = np.asarray(calibration_row["target_vectors"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    result = np.asarray(vector, dtype=np.float64).copy()
    for dimension in range(3):
        result[dimension] = _quantile_map(
            np.asarray([result[dimension]]), reference[:, dimension], target[:, dimension], rng,
        )[0]
    return result.astype(np.float32)


def sample_circle(model, alpha_bar, color: str, seed: int, sampling_steps: int,
                  calibration_row=None):
    color_id = COLOR_NAMES.index(color)
    class_id = torch.tensor([color_id * 2], dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn((1, 5), generator=generator)
    times = torch.linspace(T - 1, 0, sampling_steps + 1).long()
    with torch.inference_mode():
        for index in range(len(times) - 1):
            timestep, previous = int(times[index]), int(times[index + 1])
            time_batch = torch.full((1,), timestep, dtype=torch.long)
            epsilon = model(state, time_batch, class_id)
            alpha, alpha_previous = alpha_bar[timestep], alpha_bar[previous]
            clean = ((state - (1.0 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1.0, 1.0)
            state = alpha_previous.sqrt() * clean + (1.0 - alpha_previous).sqrt() * epsilon
        time_zero = torch.zeros((1,), dtype=torch.long)
        epsilon_zero = model(state, time_zero, class_id)
        alpha_zero = alpha_bar[0]
        state = ((state - (1.0 - alpha_zero).sqrt() * epsilon_zero) / alpha_zero.sqrt()).clamp(-1.0, 1.0)
    values = state[0].numpy()
    if calibration_row is not None:
        values = apply_geometry_calibration(values, calibration_row, seed)
    return {
        "color_id": color_id,
        "color": color,
        "cx_sampled": float(np.clip(32.0 + 10.0 * values[0], 22.0, 42.0)),
        "cy_sampled": float(np.clip(32.0 + 10.0 * values[1], 22.0, 42.0)),
        "radius": int(np.clip(round(9.5 + 2.5 * values[2]), 7, 12)),
        "shape": "circle",
        "geometry_source": "train_only_quantile_calibrated_l34" if calibration_row is not None else "l34_ema_diffusion",
    }


def sample_factorized_circle(color: str, seed: int, train_rows):
    """Sample independent circle geometry marginals fitted on training rows."""
    color_id = COLOR_NAMES.index(color)
    class_id = color_id * 2
    rows = [row for row in train_rows if int(row["class_id"]) == class_id]
    if not rows:
        raise ValueError(f"No training geometry rows found for class {class_id}")
    rng = np.random.default_rng(seed)
    cx = int(rng.choice([int(row["cx"]) for row in rows]))
    cy = int(rng.choice([int(row["cy"]) for row in rows]))
    radius = int(rng.choice([int(row["size"]) for row in rows]))
    return {
        "color_id": color_id, "color": color, "cx_sampled": float(cx),
        "cy_sampled": float(cy), "radius": radius, "shape": "circle",
        "geometry_source": "factorized_empirical_prior_from_train_split",
    }


def generate_circle(model, alpha_bar, color, seed, steps, sampler, calibration_table, train_rows):
    if sampler == "factorized_train_prior":
        return sample_factorized_circle(color, seed, train_rows)
    color_class = str(COLOR_NAMES.index(color) * 2)
    calibration_row = calibration_table[color_class] if calibration_table is not None else None
    return sample_circle(model, alpha_bar, color, seed, steps, calibration_row)


def _next_event(x_left, x_right, v_left, v_right, left_bound, right_bound, contact):
    candidates = []
    if v_left < -1e-10:
        candidates.append(((left_bound - x_left) / v_left, "left_wall"))
    if v_right > 1e-10:
        candidates.append(((right_bound - x_right) / v_right, "right_wall"))
    closing_speed = v_left - v_right
    gap = x_right - x_left - contact
    if closing_speed > 1e-10:
        candidates.append((max(0.0, gap) / closing_speed, "circle_collision"))
    return min(candidates, key=lambda pair: pair[0]) if candidates else (math.inf, None)


def advance_event_segment(state, dt, elapsed, left_bound, right_bound, contact):
    """Advance a pair exactly between collision events, preserving object identity."""
    x_left, x_right, v_left, v_right = state
    remaining = dt
    events = []
    while remaining > 1e-10:
        event_time, kind = _next_event(
            x_left, x_right, v_left, v_right, left_bound, right_bound, contact,
        )
        if kind is None or event_time > remaining + 1e-9:
            x_left += v_left * remaining
            x_right += v_right * remaining
            elapsed += remaining
            remaining = 0.0
            break
        event_time = max(0.0, event_time)
        x_left += v_left * event_time
        x_right += v_right * event_time
        elapsed += event_time
        remaining -= event_time
        before = [v_left, v_right]
        if kind == "left_wall":
            x_left = left_bound
            v_left = -v_left
            object_ids = ["circle_left"]
        elif kind == "right_wall":
            x_right = right_bound
            v_right = -v_right
            object_ids = ["circle_right"]
        else:
            midpoint = (x_left + x_right) / 2.0
            x_left, x_right = midpoint - contact / 2.0, midpoint + contact / 2.0
            v_left, v_right = v_right, v_left
            object_ids = ["circle_left", "circle_right"]
        events.append({
            "time_seconds": elapsed, "type": kind, "object_ids": object_ids,
            "velocity_before_px_s": before, "velocity_after_px_s": [v_left, v_right],
        })
        if event_time <= 1e-10 and remaining > 1e-10:
            # After resolving an event, its normal velocity points away from the surface.
            # This guard only prevents pathological floating-point zero-time loops.
            elapsed += min(1e-9, remaining)
            remaining -= min(1e-9, remaining)
    return (x_left, x_right, v_left, v_right), elapsed, events


def simulate_pair(left_radius, right_radius, duration, fps, seed, width_floor=160,
                  target_collision_seconds=None, speed_profile=None, profile_substeps=4):
    rng = random.Random(seed)
    margin = 12.0
    contact = float(left_radius + right_radius)
    width = max(width_floor, math.ceil(2 * contact + 2 * margin + 100))
    height = max(64, math.ceil(2 * max(left_radius, right_radius) + 2 * margin))
    left_bound = float(left_radius) + margin
    right_bound = float(width - right_radius) - margin
    clearance = right_bound - left_bound - contact
    if clearance <= 0:
        raise ValueError("Generated circles do not fit in the collision canvas")
    # Leave enough time after impact to show the velocity exchange without a second
    # collision crowding the end of a six-second clip.
    if target_collision_seconds is not None:
        target_collision = float(target_collision_seconds)
        if not 0.0 < target_collision < duration:
            raise ValueError("target collision time must be inside the clip")
    elif duration >= 5.0:
        first_frame = math.ceil(2.05 * fps)
        last_frame = math.floor(min(2.4, duration * 0.42) * fps)
        target_collision = (rng.randint(first_frame, last_frame) / fps
                            if last_frame >= first_frame else duration * 0.4)
    else:
        target_collision = min(rng.uniform(duration * 0.34, duration * 0.40), duration * 0.42)
    frame_count = max(2, round(duration * fps))
    substeps = profile_substeps if speed_profile is not None else 1
    total_substeps = frame_count * substeps
    dt = duration / total_substeps
    profile = (np.asarray(speed_profile, dtype=np.float64) if speed_profile is not None
               else np.ones(total_substeps, dtype=np.float64))
    if len(profile) != total_substeps or np.any(profile <= 0):
        raise ValueError("speed profile must have one positive scale per simulation substep")
    collision_step = int(round(target_collision / dt))
    collision_step = min(total_substeps, max(1, collision_step))
    integrated_profile_time = (float(profile[:collision_step].sum() * dt)
                               if speed_profile is not None else target_collision)
    speed = clearance / (2.0 * integrated_profile_time)
    state = (left_bound, right_bound, speed * profile[0], -speed * profile[0])
    substep_dt = dt
    centers, events = [], []
    elapsed = 0.0
    minimum_gap = math.inf
    for frame_index in range(frame_count):
        centers.append([state[0], state[1]])
        minimum_gap = min(minimum_gap, state[1] - state[0] - contact)
        for inner_step in range(substeps):
            profile_index = frame_index * substeps + inner_step
            scale = profile[profile_index]
            direction_left = 1.0 if state[2] >= 0 else -1.0
            direction_right = 1.0 if state[3] >= 0 else -1.0
            state = (state[0], state[1], direction_left * speed * scale,
                     direction_right * speed * scale)
            state, elapsed, inner_events = advance_event_segment(
                state, substep_dt, elapsed, left_bound, right_bound, contact,
            )
            events.extend(inner_events)
    first_collision = next((event for event in events if event["type"] == "circle_collision"), None)
    endpoint_error = (abs(first_collision["time_seconds"] - target_collision)
                      if first_collision else None)
    center_bounds_ok = all(
        left_bound - 1e-6 <= pair[0] <= width - left_radius - margin + 1e-6 and
        right_radius + margin - 1e-6 <= pair[1] <= right_bound + 1e-6
        for pair in centers
    )
    return {
        "width": width, "height": height, "margin": margin,
        "left_bound": left_bound, "right_bound": right_bound,
        "target_first_collision_seconds": target_collision,
        "base_speed_px_per_second": speed,
        "initial_speed_px_per_second": float(speed * profile[0]),
        "maximum_speed_px_per_second": float(speed * np.max(profile)),
        "speed_profile_summary": {
            "substeps": int(total_substeps),
            "min_scale": float(profile.min()), "max_scale": float(profile.max()),
            "mean_scale": float(profile.mean()), "std_scale": float(profile.std()),
            "coefficient_of_variation": float(profile.std() / max(profile.mean(), 1e-8)),
            "max_adjacent_scale_change": float(np.max(np.abs(np.diff(profile)))) if len(profile) > 1 else 0.0,
        },
        "centers_per_frame": centers,
        "events": events,
        "first_collision_time_error_seconds": endpoint_error,
        "minimum_sampled_separation_margin_px": minimum_gap,
        "center_bounds_ok": center_bounds_ok,
        "no_sampled_overlap": minimum_gap >= -1e-6,
        "circle_collision_count": sum(e["type"] == "circle_collision" for e in events),
    }


def predict_learned_speed_style(checkpoint_path: Path, style: str, duration: float, fps: int):
    """Use v37's learned slow/quick motion table to set the collision tempo."""
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"motion checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    table = checkpoint["model_state_dict"].get("motion_velocity.weight")
    if table is None or table.shape[0] < 6:
        raise ValueError("motion checkpoint lacks the learned direction/speed velocity table")
    table = table.detach().cpu().numpy()
    # Existing motion IDs: left_to_right=1 and right_to_left=2; speed IDs 0/1.
    slow_score = float((abs(table[2, 0]) + abs(table[4, 0])) / 2.0)
    quick_score = float((abs(table[3, 0]) + abs(table[5, 0])) / 2.0)
    if quick_score <= slow_score:
        raise ValueError("learned quick-speed row is not faster than the slow-speed row")
    score = slow_score if style == "slowly" else quick_score
    normalized_speed = (score - slow_score) / max(quick_score - slow_score, 1e-8)
    # Keep the initial contact in a window that leaves time to show the rebound
    # and avoids a second circle contact during the six-second reference clip.
    # The variable-speed profile can create another rebound sooner than the
    # constant-velocity model. Keep the first impact in a late enough window
    # that a second collision is unlikely within this short clip.
    low_time = duration * 0.40
    high_time = min(duration * 0.44, duration - 1.0 / fps)
    if high_time < low_time:
        low_time, high_time = duration * 0.35, duration * 0.40
    wanted = high_time - normalized_speed * (high_time - low_time)
    first_frame = math.ceil(low_time * fps)
    last_frame = math.floor(high_time * fps)
    if last_frame >= first_frame:
        collision_frame = min(last_frame, max(first_frame, round(wanted * fps)))
        collision_time = collision_frame / fps
    else:
        collision_time = wanted
    return {
        "style": style,
        "learned_slow_velocity_score": slow_score,
        "learned_quick_velocity_score": quick_score,
        "learned_speed_rank_0_to_1": normalized_speed,
        "target_collision_seconds": collision_time,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_seed": checkpoint.get("video", {}).get("seed"),
    }


def load_trajectory_profile_model(checkpoint_path: Path, threads: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    width = int(checkpoint["config"]["architecture"]["width"])
    model = TrajectoryProfileNet(width=width)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.set_num_threads(threads)
    return model, checkpoint


def make_learned_profile(model, style, seed, steps, threads):
    style_id = 0 if style == "slowly" else 1
    parameters = sample_profile_parameters(seed, style_id)
    predicted, times = predict_profile(model, parameters, steps, threads)
    teacher = target_profile_scale(times, **parameters).astype(np.float64)
    errors = predicted - teacher
    return predicted, {
        "parameters": parameters,
        "teacher_profile_mae": float(np.mean(np.abs(errors))),
        "teacher_profile_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "teacher_profile_max_error": float(np.max(np.abs(errors))),
        "scale_min": float(predicted.min()), "scale_max": float(predicted.max()),
        "scale_mean": float(predicted.mean()),
        "scale_cv": float(predicted.std() / max(predicted.mean(), 1e-8)),
        "profile_scales": predicted.astype(float).tolist(),
    }


def render_clip(path, simulation, left_geometry, right_geometry, duration, fps):
    width, height = simulation["width"], simulation["height"]
    frame_count = len(simulation["centers_per_frame"])
    frames = []
    scale = 4
    for centers in simulation["centers_per_frame"]:
        frame = Image.new("RGB", (width * scale, height * scale), (245, 245, 245))
        draw = ImageDraw.Draw(frame)
        y = simulation.get("center_y_px", height / 2.0) * scale
        for center_x, geometry in zip(centers, (left_geometry, right_geometry)):
            radius = geometry["radius"] * scale
            cx = center_x * scale
            draw.ellipse((round(cx - radius), round(y - radius), round(cx + radius),
                          round(y + radius)), fill=COLORS[geometry["color_id"]])
        frames.append(frame.resize((width, height), Image.Resampling.LANCZOS))
    frame_ms = duration * 1000.0 / frame_count
    boundaries = [round(i * frame_ms / 10.0) * 10 for i in range(frame_count + 1)]
    durations = [max(10, boundaries[i + 1] - boundaries[i]) for i in range(frame_count)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False)
    contact_index = min(frame_count - 1, round(simulation["target_first_collision_seconds"] * fps))
    preview = path.with_name(f"{path.stem}_contact.png")
    frames[contact_index].resize((width * 3, height * 3), Image.Resampling.NEAREST).save(preview)
    return preview, sum(durations) / 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_GEOMETRY_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--left-color", choices=COLOR_NAMES, default="red")
    parser.add_argument("--right-color", choices=COLOR_NAMES, default="red")
    parser.add_argument("--validation-seeds", type=int, default=12)
    parser.add_argument("--speed-style", choices=("random", "slowly", "quickly"), default="random",
                        help="Use the v37 learned speed table to choose slow/quick collision tempo")
    parser.add_argument("--motion-checkpoint", type=Path, default=DEFAULT_MOTION_CHECKPOINT)
    parser.add_argument("--trajectory-profile", choices=("constant", "learned_accel_decel"),
                        default="constant")
    parser.add_argument("--profile-checkpoint", type=Path, default=DEFAULT_PROFILE_CHECKPOINT)
    parser.add_argument("--profile-substeps", type=int, default=4)
    parser.add_argument("--geometry-sampler", choices=("raw_diffusion", "quantile_calibrated", "factorized_train_prior"),
                        default="raw_diffusion")
    parser.add_argument("--geometry-calibration", type=Path, default=None,
                        help="Optional L36 train-only quantile calibration table")
    args = parser.parse_args()
    if not 0 < args.duration <= 6 or args.fps < 1 or args.steps < 2:
        parser.error("duration must be in (0, 6], fps >= 1, and steps >= 2")
    if args.validation_seeds < 1 or args.threads < 1 or args.profile_substeps < 1:
        parser.error("validation seeds, CPU threads, and profile substeps must be positive")
    args.checkpoint = args.checkpoint.resolve()
    args.motion_checkpoint = args.motion_checkpoint.resolve()
    args.profile_checkpoint = args.profile_checkpoint.resolve()
    if args.geometry_calibration is not None and args.geometry_sampler == "raw_diffusion":
        args.geometry_sampler = "quantile_calibrated"
    if args.geometry_sampler == "quantile_calibrated" and args.geometry_calibration is None:
        parser.error("quantile_calibrated requires --geometry-calibration")
    if args.geometry_sampler == "factorized_train_prior" and args.geometry_calibration is not None:
        parser.error("factorized_train_prior uses training marginals directly; omit --geometry-calibration")
    if args.geometry_sampler != "factorized_train_prior" and not args.checkpoint.is_file():
        parser.error(f"geometry checkpoint does not exist: {args.checkpoint}")
    if args.trajectory_profile == "learned_accel_decel" and not args.profile_checkpoint.is_file():
        parser.error(f"trajectory profile checkpoint does not exist: {args.profile_checkpoint}")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    manifest_rows = [json.loads(line) for line in DATA_MANIFEST.read_text(encoding="utf-8").splitlines()]
    train_rows = [row for row in manifest_rows if row["file"].startswith("train_")]
    checkpoint = None
    model, alpha_bar = None, None
    model_load_seconds = 0.0
    checkpoint_hash = None
    if args.geometry_sampler != "factorized_train_prior":
        model_load_start = time.perf_counter()
        model, alpha_bar, checkpoint = load_geometry_model(args.checkpoint, args.threads)
        model_load_seconds = time.perf_counter() - model_load_start
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    calibration_table = None
    calibration_sha256 = None
    if args.geometry_calibration is not None:
        calibration_path = args.geometry_calibration.resolve()
        calibration_blob = calibration_path.read_bytes()
        calibration_sha256 = hashlib.sha256(calibration_blob).hexdigest()
        calibration_payload = json.loads(calibration_blob.decode("utf-8"))
        current_manifest_hash = hashlib.sha256(CALIBRATION_MANIFEST.read_bytes()).hexdigest()
        if calibration_payload.get("training_data_manifest_sha256") != current_manifest_hash:
            parser.error("calibration table was fitted against a different training data manifest")
        calibration_by_hash = calibration_payload.get("checkpoints", {}).get(checkpoint_hash)
        if calibration_by_hash is None:
            parser.error("calibration table has no entry for the selected geometry checkpoint")
        calibration_table = calibration_by_hash
    geometry_start = time.perf_counter()
    left_geo = generate_circle(model, alpha_bar, args.left_color, args.seed + 101, args.steps,
                               args.geometry_sampler, calibration_table, train_rows)
    right_geo = generate_circle(model, alpha_bar, args.right_color, args.seed + 202, args.steps,
                                args.geometry_sampler, calibration_table, train_rows)
    geometry_seconds = time.perf_counter() - geometry_start
    effective_speed_style = args.speed_style
    if args.trajectory_profile == "learned_accel_decel" and effective_speed_style == "random":
        effective_speed_style = ("slowly", "quickly")[args.seed % 2]
    learned_speed = None
    motion_controller_seconds = 0.0
    if effective_speed_style != "random":
        motion_controller_start = time.perf_counter()
        try:
            learned_speed = predict_learned_speed_style(
                args.motion_checkpoint, effective_speed_style, args.duration, args.fps,
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        motion_controller_seconds = time.perf_counter() - motion_controller_start
    target_collision = learned_speed["target_collision_seconds"] if learned_speed else None
    profile_model = None
    profile_checkpoint = None
    profile_setup_seconds = 0.0
    speed_profile = None
    profile_metrics = None
    frame_count = max(2, round(args.duration * args.fps))
    if args.trajectory_profile == "learned_accel_decel":
        profile_setup_start = time.perf_counter()
        profile_model, profile_checkpoint = load_trajectory_profile_model(args.profile_checkpoint, args.threads)
        speed_profile, profile_metrics = make_learned_profile(
            profile_model, effective_speed_style, args.seed + 3907,
            frame_count * args.profile_substeps, args.threads,
        )
        profile_setup_seconds = time.perf_counter() - profile_setup_start
    simulation = simulate_pair(left_geo["radius"], right_geo["radius"],
                               args.duration, args.fps, args.seed,
                               target_collision_seconds=target_collision,
                               speed_profile=speed_profile,
                               profile_substeps=args.profile_substeps)
    center_y = min(simulation["height"] / 2.0,
                   max(max(left_geo["radius"], right_geo["radius"]) + simulation["margin"],
                       (left_geo["cy_sampled"] + right_geo["cy_sampled"]) / 2.0))
    # The pair shares a common y lane so the 1D elastic collision is physically valid.
    simulation["center_y_px"] = center_y
    video_start = time.perf_counter()
    preview, actual_duration = render_clip(args.output, simulation, left_geo, right_geo,
                                           args.duration, args.fps)
    video_seconds = time.perf_counter() - video_start

    validation = []
    for offset in range(args.validation_seeds):
        seed = args.seed + 1000 + offset
        left = generate_circle(model, alpha_bar, args.left_color, seed * 2 + 1, args.steps,
                               args.geometry_sampler, calibration_table, train_rows)
        right = generate_circle(model, alpha_bar, args.right_color, seed * 2 + 2, args.steps,
                                args.geometry_sampler, calibration_table, train_rows)
        validation_profile = None
        validation_profile_metrics = None
        if profile_model is not None:
            validation_profile, validation_profile_metrics = make_learned_profile(
                profile_model, effective_speed_style, seed + 3907,
                frame_count * args.profile_substeps, args.threads,
            )
        case = simulate_pair(left["radius"], right["radius"], args.duration, args.fps, seed,
                             target_collision_seconds=target_collision,
                             speed_profile=validation_profile,
                             profile_substeps=args.profile_substeps)
        validation.append({
            "seed": seed,
            "radii_px": [left["radius"], right["radius"]],
            "first_collision_time_error_seconds": case["first_collision_time_error_seconds"],
            "circle_collision_count": case["circle_collision_count"],
            "event_types": [event["type"] for event in case["events"]],
            "no_overlap": case["no_sampled_overlap"],
            "within_wall_bounds": case["center_bounds_ok"],
            "speed_profile_cv": case["speed_profile_summary"]["coefficient_of_variation"],
            "teacher_profile_rmse": (validation_profile_metrics["teacher_profile_rmse"]
                                     if validation_profile_metrics else 0.0),
        })
    checkpoint_config = checkpoint.get("config", {}) if checkpoint is not None else {}
    config_bytes = json.dumps(checkpoint_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    profile_config = profile_checkpoint.get("config", {}) if profile_checkpoint is not None else {}
    data_hash = hashlib.sha256(DATA_MANIFEST.read_bytes()).hexdigest() if DATA_MANIFEST.is_file() else None
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metrics = {
        "experiment": "L35 L34 geometry-conditioned event-aware collision",
        "generation_mode": (f"{args.geometry_sampler} geometry + canonical CPU renderer + "
                            f"{args.trajectory_profile} trajectory + exact event-driven physics"),
        "raw_rgb_diffusion_used": False,
        "learned_weights": (
            ("L34 EMA geometry checkpoint" if checkpoint is not None else "factorized train prior") +
            ("; v37 learned speed-style table" if learned_speed is not None else "") +
            ("; L39 learned speed-profile network" if profile_checkpoint is not None else "") +
            "; contact resolution is explicit physics"
        ),
        "requested_speed_style": args.speed_style,
        "speed_style": effective_speed_style,
        "learned_speed_style": learned_speed,
        "trajectory_profile": {
            "kind": args.trajectory_profile,
            "substeps_per_frame": args.profile_substeps if profile_checkpoint is not None else 1,
            "checkpoint": str(args.profile_checkpoint) if profile_checkpoint is not None else None,
            "checkpoint_sha256": (hashlib.sha256(args.profile_checkpoint.read_bytes()).hexdigest()
                                  if profile_checkpoint is not None else None),
            "checkpoint_step": profile_checkpoint.get("best_step") if profile_checkpoint is not None else None,
            "training_config": profile_config.get("training") if profile_checkpoint is not None else None,
            "inference_metrics": profile_metrics,
        },
        "checkpoint": str(args.checkpoint) if checkpoint is not None else None, "checkpoint_sha256": checkpoint_hash,
        "geometry_sampling_strategy": args.geometry_sampler,
        "geometry_calibration": str(args.geometry_calibration.resolve()) if args.geometry_calibration else None,
        "geometry_calibration_sha256": calibration_sha256,
        "checkpoint_step": checkpoint.get("step") if checkpoint is not None else None,
        "checkpoint_seed": checkpoint_config.get("training", {}).get("seed"),
        "checkpoint_config": checkpoint_config,
        "checkpoint_config_sha256": hashlib.sha256(config_bytes).hexdigest() if checkpoint is not None else None,
        "training_data_manifest": str(DATA_MANIFEST) if DATA_MANIFEST.is_file() else None,
        "training_data_manifest_sha256": data_hash,
        "inference_source_sha256": source_hash,
        "seed": args.seed, "cpu_threads": torch.get_num_threads(),
        "duration_seconds": actual_duration, "fps": args.fps,
        "frames": len(simulation["centers_per_frame"]),
        "resolution": [simulation["width"], simulation["height"]],
        "objects": [left_geo, right_geo], "object_ids": ["circle_left", "circle_right"],
        "center_y_px": center_y, "physics": "equal-mass 1D elastic circle collision with event-resolved wall reflections",
        "motion_events": simulation["events"],
        "target_first_collision_seconds": simulation["target_first_collision_seconds"],
        "first_collision_time_error_seconds": simulation["first_collision_time_error_seconds"],
        "circle_collision_count": simulation["circle_collision_count"],
        "minimum_sampled_separation_margin_px": simulation["minimum_sampled_separation_margin_px"],
        "no_sampled_overlap": simulation["no_sampled_overlap"],
        "within_wall_bounds": simulation["center_bounds_ok"],
        "validation_cases": validation,
        "validation_summary": {
            "case_count": len(validation),
            "all_no_overlap": all(row["no_overlap"] for row in validation),
            "all_within_wall_bounds": all(row["within_wall_bounds"] for row in validation),
            "all_one_circle_collision": all(row["circle_collision_count"] == 1 for row in validation),
            "max_first_collision_time_error_seconds": max(
                (row["first_collision_time_error_seconds"] or 0.0 for row in validation), default=0.0,
            ),
        },
        "timing_seconds": {
            "checkpoint_load": model_load_seconds, "geometry_sampling": geometry_seconds,
            "learned_motion_controller_setup": motion_controller_seconds,
            "trajectory_profile_setup_and_inference": profile_setup_seconds,
            "video_render_and_save": video_seconds, "total": time.perf_counter() - start,
        },
        "video": str(args.output), "contact_preview": str(preview),
    }
    def json_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize value of type {type(value).__name__}")

    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2, default=json_scalar), encoding="utf-8",
    )
    print(json.dumps({
        "video": str(args.output), "preview": str(preview), "frames": metrics["frames"],
        "resolution": metrics["resolution"], "events": simulation["events"],
        "validation_summary": metrics["validation_summary"],
        "timing_seconds": metrics["timing_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
