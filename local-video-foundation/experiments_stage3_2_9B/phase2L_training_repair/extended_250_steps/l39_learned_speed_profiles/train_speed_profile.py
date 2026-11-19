"""Train/evaluate a compact CPU speed-profile model on synthetic trajectories."""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from video_foundation.inference.trajectory_profile import (
    TrajectoryProfileNet, make_profile_features, target_profile_scale,
)


HERE = Path(__file__).resolve().parent


def make_dataset(profile_count, points_per_profile, seed):
    rng = np.random.default_rng(seed)
    feature_rows, targets, styles = [], [], []
    for index in range(profile_count):
        style_id = index % 2
        acceleration = float(rng.uniform(-1.0, 1.0))
        amplitude = float(rng.uniform(0.0, 1.0))
        phase = float(rng.uniform(-np.pi, np.pi))
        cycles = float(rng.uniform(0.5, 2.0))
        times = rng.uniform(0.0, 1.0, size=points_per_profile).astype(np.float32)
        feature_rows.append(make_profile_features(
            times, style_id, acceleration, amplitude, phase, cycles,
        ))
        targets.append(target_profile_scale(
            times, style_id, acceleration, amplitude, phase, cycles,
        ))
        styles.extend([style_id] * points_per_profile)
    return (np.concatenate(feature_rows), np.concatenate(targets),
            np.asarray(styles, dtype=np.int64))


def evaluate(model, features, targets, styles):
    model.eval()
    with torch.inference_mode():
        predicted = model(torch.from_numpy(features)).numpy()
    errors = predicted - targets
    result = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_absolute_error": float(np.max(np.abs(errors))),
        "constant_speed_baseline_rmse": float(np.sqrt(np.mean((1.0 - targets) ** 2))),
        "by_style": {},
    }
    for style_id, name in ((0, "slowly"), (1, "quickly")):
        selected = styles == style_id
        result["by_style"][name] = {
            "mae": float(np.mean(np.abs(errors[selected]))),
            "rmse": float(np.sqrt(np.mean(errors[selected] ** 2))),
            "constant_speed_baseline_rmse": float(np.sqrt(np.mean((1.0 - targets[selected]) ** 2))),
        }
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=HERE)
    parser.add_argument("--profiles", type=int, default=1800)
    parser.add_argument("--validation_profiles", type=int, default=400)
    parser.add_argument("--points_per_profile", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3901)
    args = parser.parse_args()
    if min(args.profiles, args.validation_profiles, args.points_per_profile,
           args.steps, args.batch_size, args.width, args.threads) < 1:
        parser.error("profile counts, steps, batch size, width, and threads must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train_x, train_y, train_styles = make_dataset(
        args.profiles, args.points_per_profile, args.seed + 1,
    )
    val_x, val_y, val_styles = make_dataset(
        args.validation_profiles, args.points_per_profile, args.seed + 2,
    )
    train_hash = hashlib.sha256(train_x.tobytes() + train_y.tobytes()).hexdigest()
    val_hash = hashlib.sha256(val_x.tobytes() + val_y.tobytes()).hexdigest()
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y)
    model = TrajectoryProfileNet(args.width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    index_generator = torch.Generator().manual_seed(args.seed + 3)
    best_state = copy.deepcopy(model.state_dict())
    best_val = evaluate(model, val_x, val_y, val_styles)
    best_rmse = best_val["rmse"]
    best_step = 0
    history = [{"step": 0, "train_mse": None, "validation": best_val, "elapsed_seconds": 0.0}]
    start = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(x), (args.batch_size,), generator=index_generator)
        prediction = model(x[indices])
        loss = F.mse_loss(prediction, y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == args.steps:
            validation = evaluate(model, val_x, val_y, val_styles)
            record = {"step": step, "train_mse": float(loss.detach()),
                      "validation": validation,
                      "elapsed_seconds": round(time.perf_counter() - start, 3)}
            history.append(record)
            if validation["rmse"] < best_rmse:
                best_rmse, best_val = validation["rmse"], validation
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
            print(f"step={step} train_mse={record['train_mse']:.6f} "
                  f"val_rmse={validation['rmse']:.6f} mae={validation['mae']:.6f}", flush=True)
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    final_val = evaluate(model, val_x, val_y, val_styles)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "input_features": ["time_0_1", "speed_style_minus1_plus1", "acceleration_control",
                           "wave_amplitude", "sin_phase", "cos_phase", "cycles"],
        "output": "positive_speed_scale_clipped_to_0.55_1.45_at_inference",
        "teacher": "1 + 0.20*a*(2t-1) + 0.14*b*sin(2*pi*f*t+phase) + 0.06*style*sin(pi*t)",
        "architecture": {"kind": "MLP", "width": args.width, "hidden_layers": 2,
                         "parameters": parameter_count},
        "training": {"profiles": args.profiles, "validation_profiles": args.validation_profiles,
                     "points_per_profile": args.points_per_profile, "steps": args.steps,
                     "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                     "seed": args.seed, "threads": torch.get_num_threads(),
                     "train_data_sha256": train_hash, "validation_data_sha256": val_hash},
    }
    checkpoint_path = args.output_dir / "trajectory_profile.pt"
    torch.save({"model_state_dict": best_state, "config": config,
                "best_validation_rmse": best_rmse, "best_step": best_step,
                "training_elapsed_seconds": round(training_seconds, 3)}, checkpoint_path)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {
        "experiment": "L39 CPU learned acceleration/deceleration profile",
        "training_mode": "supervised synthetic profile regression; not video-pixel training",
        "config": config,
        "validation_at_initialization": history[0]["validation"],
        "validation_best": final_val,
        "best_step": best_step,
        "training_elapsed_seconds": round(training_seconds, 3),
        "total_training_and_evaluation_seconds": round(time.perf_counter() - start, 3),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "training_source_sha256": source_hash,
        "history": history,
    }
    (args.output_dir / "L39_PROFILE_RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"parameters": parameter_count, "best_step": result["best_step"],
                      "validation": final_val,
                      "training_seconds": result["training_elapsed_seconds"],
                      "checkpoint": result["checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
