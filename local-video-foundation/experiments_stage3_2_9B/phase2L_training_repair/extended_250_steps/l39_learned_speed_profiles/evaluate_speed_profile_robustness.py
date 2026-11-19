"""Three-checkpoint, two-style event-physics evaluation for L39 profiles."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from video_foundation.inference.sample_geometry_collision import (
    DEFAULT_MOTION_CHECKPOINT, load_trajectory_profile_model, make_learned_profile,
    predict_learned_speed_style, simulate_pair,
)


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--scenarios-per-style", type=int, default=100)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--profile-substeps", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3910)
    parser.add_argument("--output", type=Path, default=HERE / "L39_MULTI_SEED_GATE.json")
    args = parser.parse_args()
    started = time.perf_counter()
    run_rows = []
    styles = ("slowly", "quickly")
    motion_controls = {
        style: predict_learned_speed_style(
            DEFAULT_MOTION_CHECKPOINT, style, args.duration, args.fps,
        ) for style in styles
    }
    for checkpoint_path in args.checkpoints:
        checkpoint_path = checkpoint_path.resolve()
        model, checkpoint = load_trajectory_profile_model(checkpoint_path, args.threads)
        result_path = checkpoint_path.with_name("L39_PROFILE_RESULTS.json")
        training_results = json.loads(result_path.read_text(encoding="utf-8"))
        checkpoint_seed = int(training_results["config"]["training"]["seed"])
        for style in styles:
            style_id = 0 if style == "slowly" else 1
            target_time = motion_controls[style]["target_collision_seconds"]
            cases = []
            for index in range(args.scenarios_per_style):
                seed = args.seed + checkpoint_seed * 3 + style_id * 10000 + index
                rng = random.Random(seed)
                left_radius, right_radius = rng.randint(7, 12), rng.randint(7, 12)
                profile, profile_metrics = make_learned_profile(
                    model, style, seed + 3907,
                    max(2, round(args.duration * args.fps)) * args.profile_substeps,
                    args.threads,
                )
                simulation = simulate_pair(
                    left_radius, right_radius, args.duration, args.fps, seed,
                    target_collision_seconds=target_time, speed_profile=profile,
                    profile_substeps=args.profile_substeps,
                )
                cases.append({
                    "seed": seed,
                    "radii_px": [left_radius, right_radius],
                    "circle_collision_count": simulation["circle_collision_count"],
                    "no_overlap": simulation["no_sampled_overlap"],
                    "within_wall_bounds": simulation["center_bounds_ok"],
                    "first_collision_time_error_seconds": simulation["first_collision_time_error_seconds"],
                    "profile_rmse_vs_teacher": profile_metrics["teacher_profile_rmse"],
                    "profile_cv": simulation["speed_profile_summary"]["coefficient_of_variation"],
                    "event_types": [event["type"] for event in simulation["events"]],
                })
            profile_validation = training_results["validation_best"]
            run_rows.append({
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "training_seed": checkpoint_seed,
                "best_training_step": training_results["best_step"],
                "validation_profile_rmse": profile_validation["rmse"],
                "constant_speed_baseline_rmse": profile_validation["constant_speed_baseline_rmse"],
                "validation_rmse_reduction_percent": (
                    (profile_validation["constant_speed_baseline_rmse"] - profile_validation["rmse"]) /
                    profile_validation["constant_speed_baseline_rmse"] * 100.0
                ),
                "speed_style": style,
                "target_collision_seconds": target_time,
                "motion_style_checkpoint_sha256": motion_controls[style]["checkpoint_sha256"],
                "scenario_count": len(cases),
                "scenario_summary": {
                    "all_exactly_one_circle_collision": all(row["circle_collision_count"] == 1 for row in cases),
                    "all_no_overlap": all(row["no_overlap"] for row in cases),
                    "all_within_wall_bounds": all(row["within_wall_bounds"] for row in cases),
                    "max_collision_time_error_seconds": max(
                        row["first_collision_time_error_seconds"] or 0.0 for row in cases),
                    "mean_profile_rmse_vs_teacher": float(np.mean([row["profile_rmse_vs_teacher"] for row in cases])),
                    "mean_speed_profile_cv": float(np.mean([row["profile_cv"] for row in cases])),
                    "constant_speed_cv": 0.0,
                },
                "scenarios": cases,
            })
    result = {
        "experiment": "L39 learned acceleration/deceleration profile robustness",
        "profile_supervision": "synthetic analytical speed-profile teacher; not learned from real video trajectories",
        "trajectory_semantics": "L39 predicts bounded positive within-clip speed variation; L38 checkpoint selects slow/quick tempo; exact event solver handles collision and walls",
        "settings": {"duration_seconds": args.duration, "fps": args.fps,
                     "profile_substeps": args.profile_substeps,
                     "scenarios_per_style_per_checkpoint": args.scenarios_per_style,
                     "threads": args.threads, "seed": args.seed},
        "runs": run_rows,
        "aggregate": {
            "trained_profile_seeds": sorted(set(row["training_seed"] for row in run_rows)),
            "all_validation_rmse_beats_constant_baseline": all(
                row["validation_profile_rmse"] < row["constant_speed_baseline_rmse"] for row in run_rows),
            "all_physics_scenarios_pass": all(
                row["scenario_summary"]["all_exactly_one_circle_collision"] and
                row["scenario_summary"]["all_no_overlap"] and
                row["scenario_summary"]["all_within_wall_bounds"] for row in run_rows),
            "total_physics_scenarios": sum(row["scenario_count"] for row in run_rows),
            "max_contact_time_error_seconds": max(
                row["scenario_summary"]["max_collision_time_error_seconds"] for row in run_rows),
            "mean_profile_validation_rmse": float(np.mean([row["validation_profile_rmse"] for row in run_rows])),
            "mean_constant_speed_baseline_rmse": float(np.mean([row["constant_speed_baseline_rmse"] for row in run_rows])),
            "mean_validation_rmse_reduction_percent": float(np.mean([row["validation_rmse_reduction_percent"] for row in run_rows])),
            "mean_profile_speed_cv": float(np.mean([row["scenario_summary"]["mean_speed_profile_cv"] for row in run_rows])),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def json_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize value of type {type(value).__name__}")

    args.output.write_text(json.dumps(result, indent=2, default=json_scalar), encoding="utf-8")
    print(json.dumps({"aggregate": result["aggregate"], "elapsed_seconds": result["elapsed_seconds"],
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
