"""Evaluate train-only per-class quantile calibration for L34 geometry diffusion."""

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


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "experiments_stage3_2_9B" / "phase2L_training_repair" / "extended_250_steps" / "l32_cpu_factorized_salience_pilot" / "data"
T = 1000
SIZE = 64
COLORS = [(220, 30, 30), (30, 80, 220), (230, 200, 30)]
COLOR_NAMES = ["red", "blue", "yellow"]
SHAPE_NAMES = ["circle", "triangle"]
BACKGROUND = (245, 245, 245)


class GeometryDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Embedding(T, 16)
        self.color_embed = nn.Embedding(3, 8)
        self.shape_embed = nn.Embedding(2, 8)
        self.input = nn.Linear(37, 128)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(128, 128), nn.SiLU()) for _ in range(3)])
        self.output = nn.Linear(128, 5)

    def forward(self, noisy, timesteps, class_ids):
        condition = torch.cat((self.time_embed(timesteps), self.color_embed(class_ids // 2),
                               self.shape_embed(class_ids % 2)), dim=1)
        hidden = F.silu(self.input(torch.cat((noisy, condition), dim=1)))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


def recover_rotation(row):
    rng = random.Random(int(row["seed"]))
    rng.randint(7, 12)
    rng.randint(22, 42)
    rng.randint(22, 42)
    return rng.uniform(0.0, 2.0 * math.pi)


def encoded_row(row):
    shape_id = int(row["class_id"]) % 2
    cx, cy = (float(row["cx"]) - 32.0) / 10.0, (float(row["cy"]) - 32.0) / 10.0
    if shape_id == 0:
        extent, angle = (float(row["size"]) - 9.5) / 2.5, 0.0
    else:
        extent = (float(row["size"]) - 25.5) / 6.5
        angle = recover_rotation(row) % (2.0 * math.pi / 3.0)
    return np.asarray([cx, cy, extent, math.cos(3.0 * angle), math.sin(3.0 * angle)], dtype=np.float32)


def sample_vectors(model, alpha_bar, class_count, per_class, seed, steps):
    classes = torch.arange(6, dtype=torch.long).repeat_interleave(per_class)
    state = torch.randn((len(classes), 5), generator=torch.Generator().manual_seed(seed))
    times = torch.linspace(T - 1, 0, steps + 1).long()
    model.eval()
    with torch.inference_mode():
        for index in range(len(times) - 1):
            timestep, previous = int(times[index]), int(times[index + 1])
            time_batch = torch.full((len(classes),), timestep, dtype=torch.long)
            epsilon = model(state, time_batch, classes)
            alpha, alpha_prev = alpha_bar[timestep], alpha_bar[previous]
            clean = ((state - (1.0 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1.0, 1.0)
            state = alpha_prev.sqrt() * clean + (1.0 - alpha_prev).sqrt() * epsilon
        epsilon0 = model(state, torch.zeros_like(classes), classes)
        alpha0 = alpha_bar[0]
        state = ((state - (1.0 - alpha0).sqrt() * epsilon0) / alpha0.sqrt()).clamp(-1.0, 1.0)
    return classes.numpy(), state.numpy()


def rank_map(values, reference_samples, target_samples, rng):
    """Map model sample ranks to the training-split empirical marginal."""
    reference = np.sort(np.asarray(reference_samples, dtype=np.float64))
    target = np.sort(np.asarray(target_samples, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    tie_fraction = rng.random(values.shape)
    ranks = left + tie_fraction * (right - left)
    quantiles = np.clip((ranks + 0.5) / (len(reference) + 1.0), 0.0, 1.0)
    indices = np.minimum((quantiles * len(target)).astype(np.int64), len(target) - 1)
    return target[indices]


def decode(class_id, vector):
    shape_id = class_id % 2
    cx = float(np.clip(32 + 10 * vector[0], 22, 42))
    cy = float(np.clip(32 + 10 * vector[1], 22, 42))
    if shape_id == 0:
        extent = int(np.clip(round(9.5 + 2.5 * vector[2]), 7, 12))
        angle = 0.0
    else:
        extent = int(np.clip(round(25.5 + 6.5 * vector[2]), 19, 32))
        angle = (math.atan2(float(vector[4]), float(vector[3])) / 3.0) % (2.0 * math.pi / 3.0)
    return {"class_id": class_id, "color_id": class_id // 2, "shape_id": shape_id,
            "cx": cx, "cy": cy, "extent": extent, "angle_rad": angle}


def calibrate_class(class_id, vectors, reference, targets, seed):
    output = vectors.copy()
    rng = np.random.default_rng(seed)
    source = np.asarray(reference[class_id])
    target = np.asarray(targets[class_id])
    for dimension, name in enumerate(("cx", "cy", "extent")):
        output[:, dimension] = rank_map(output[:, dimension], source[:, dimension], target[:, dimension], rng)
    if class_id % 2 == 1:
        ref_angle = np.mod(np.arctan2(source[:, 4], source[:, 3]) / 3.0, 2.0 * math.pi / 3.0)
        target_angle = np.mod(np.arctan2(target[:, 4], target[:, 3]) / 3.0, 2.0 * math.pi / 3.0)
        sample_angle = np.mod(np.arctan2(vectors[:, 4], vectors[:, 3]) / 3.0,
                              2.0 * math.pi / 3.0)
        angle = rank_map(sample_angle, ref_angle, target_angle, rng)
        output[:, 3], output[:, 4] = np.cos(3.0 * angle), np.sin(3.0 * angle)
    return output


def ks_distance(reference, generated):
    reference = np.sort(np.asarray(reference))
    generated = np.sort(np.asarray(generated))
    points = np.sort(np.concatenate((reference, generated)))
    return float(np.max(np.abs(
        np.searchsorted(reference, points, side="right") / len(reference) -
        np.searchsorted(generated, points, side="right") / len(generated)
    )))


def render(geometry, supersample=4):
    scale = supersample
    high = Image.new("RGB", (SIZE * scale, SIZE * scale), BACKGROUND)
    draw = ImageDraw.Draw(high)
    cx, cy, extent = geometry["cx"] * scale, geometry["cy"] * scale, geometry["extent"] * scale
    color = COLORS[geometry["color_id"]]
    if geometry["shape_id"] == 0:
        draw.ellipse((cx - extent, cy - extent, cx + extent, cy + extent), fill=color)
    else:
        radius = extent / math.sqrt(3.0)
        angle = geometry["angle_rad"]
        points = [(cx + radius * math.cos(angle + 2.0 * math.pi * k / 3.0),
                   cy + radius * math.sin(angle + 2.0 * math.pi * k / 3.0)) for k in range(3)]
        draw.polygon(points, fill=color)
    return high.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def geometry_metrics(rows, class_id, generated):
    reference = [row for row in rows if int(row["class_id"]) == class_id]
    features = {
        "cx": ([float(row["cx"]) for row in reference], [item["cx"] for item in generated]),
        "cy": ([float(row["cy"]) for row in reference], [item["cy"] for item in generated]),
        "extent": ([float(row["size"]) for row in reference], [item["extent"] for item in generated]),
    }
    if class_id % 2 == 1:
        features["angle_rad_mod_120deg"] = (
            [recover_rotation(row) % (2.0 * math.pi / 3.0) for row in reference],
            [item["angle_rad"] for item in generated],
        )
    return {key: {"ks_distance": ks_distance(ref, gen),
                  "heldout_mean": float(np.mean(ref)), "generated_mean": float(np.mean(gen)),
                  "heldout_std": float(np.std(ref)), "generated_std": float(np.std(gen))}
            for key, (ref, gen) in features.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=HERE / "l36_quantile_calibration")
    parser.add_argument("--calibration_samples_per_class", type=int, default=512)
    parser.add_argument("--evaluation_samples_per_class", type=int, default=100)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3600)
    args = parser.parse_args()
    if min(args.calibration_samples_per_class, args.evaluation_samples_per_class,
           args.sampling_steps, args.threads) < 1:
        parser.error("sample counts, sampling steps, and threads must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    rows = [json.loads(line) for line in (DATA_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    train_rows = [row for row in rows if row["file"].startswith("train_")]
    val_rows = [row for row in rows if row["file"].startswith("val_")]
    targets = {class_id: np.stack([encoded_row(row) for row in train_rows
                                   if int(row["class_id"]) == class_id]) for class_id in range(6)}
    manifest_hash = hashlib.sha256((DATA_DIR / "manifest.jsonl").read_bytes()).hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    eval_started = time.perf_counter()
    runs, calibration_tables = [], {}
    first_grid = None
    for checkpoint_path in args.checkpoints:
        checkpoint_path = checkpoint_path.resolve()
        checkpoint_bytes = checkpoint_path.read_bytes()
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config_bytes = json.dumps(checkpoint.get("config", {}), sort_keys=True,
                                  separators=(",", ":")).encode("utf-8")
        model = GeometryDenoiser()
        model.load_state_dict(checkpoint["ema_model"])
        model.eval()
        alpha_bar = torch.cumprod(1.0 - torch.linspace(0.0001, 0.02, T), dim=0)
        model_seed = int(checkpoint["config"]["training"]["seed"])

        cal_classes, cal_vectors = sample_vectors(
            model, alpha_bar, 6, args.calibration_samples_per_class,
            args.seed + model_seed * 13, args.sampling_steps,
        )
        reference = {class_id: cal_vectors[cal_classes == class_id].copy() for class_id in range(6)}
        raw_classes, raw_vectors = sample_vectors(
            model, alpha_bar, 6, args.evaluation_samples_per_class,
            args.seed + model_seed * 17 + 1, args.sampling_steps,
        )
        calibrated_vectors = raw_vectors.copy()
        tables_for_checkpoint = {}
        for class_id in range(6):
            selector = raw_classes == class_id
            calibrated_vectors[selector] = calibrate_class(
                class_id, raw_vectors[selector], reference, targets,
                args.seed + model_seed * 31 + class_id,
            )
            tables_for_checkpoint[str(class_id)] = {
                "reference_vectors": reference[class_id].tolist(),
                "target_vectors": targets[class_id].tolist(),
            }
        calibration_tables[checkpoint_hash] = tables_for_checkpoint

        raw_samples, calibrated_samples = [], []
        per_class = {}
        grid_items = []
        for class_id in range(6):
            raw_items = [decode(class_id, vector) for vector in raw_vectors[raw_classes == class_id]]
            new_items = [decode(class_id, vector) for vector in calibrated_vectors[raw_classes == class_id]]
            raw_samples.extend(raw_items)
            calibrated_samples.extend(new_items)
            raw_score = geometry_metrics(val_rows, class_id, raw_items)
            calibrated_score = geometry_metrics(val_rows, class_id, new_items)
            per_class[str(class_id)] = {"raw": raw_score, "calibrated": calibrated_score}
            if first_grid is None:
                grid_items.append((class_id, 0, raw_items[0], new_items[0]))
        raw_ks = [value["ks_distance"] for row in per_class.values() for value in row["raw"].values()]
        calibrated_ks = [value["ks_distance"] for row in per_class.values() for value in row["calibrated"].values()]
        result = {
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
            "checkpoint_seed": model_seed,
            "checkpoint_config": checkpoint.get("config", {}),
            "checkpoint_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "raw_max_feature_ks": max(raw_ks), "calibrated_max_feature_ks": max(calibrated_ks),
            "raw_mean_feature_ks": float(np.mean(raw_ks)),
            "calibrated_mean_feature_ks": float(np.mean(calibrated_ks)),
            "per_class_feature_metrics": per_class,
            "sample_support": {
                "all_calibrated_centers_in_22_42": all(22 <= item["cx"] <= 42 and 22 <= item["cy"] <= 42
                                                        for item in calibrated_samples),
                "all_calibrated_extents_observed_in_training": all(
                    item["extent"] in {int(row["size"]) for row in train_rows
                                       if int(row["class_id"]) == item["class_id"]}
                    for item in calibrated_samples),
            },
        }
        runs.append(result)
        if first_grid is None:
            first_grid = (checkpoint_path, grid_items)

    tile, row_h, label_w = 128, 144, 112
    grid = Image.new("RGB", (label_w + 2 * tile, 52 + 6 * row_h), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((label_w, 8), "Raw L34 vs train-quantile calibrated (same noise)", fill="black")
    for class_id, sample_id, raw_item, calibrated_item in first_grid[1]:
        y = 52 + class_id * row_h
        draw.text((4, y + 52), f"{COLOR_NAMES[class_id // 2]}\n{SHAPE_NAMES[class_id % 2]}\n#{sample_id}", fill="black")
        for column, item in enumerate((raw_item, calibrated_item)):
            x = label_w + column * tile
            preview = render(item).resize((tile, tile), Image.Resampling.NEAREST)
            grid.paste(preview, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(50, 120, 50), width=1)
    grid_path = args.output_dir / "L36_raw_vs_calibrated_grid.png"
    grid.save(grid_path)
    calibration_path = args.output_dir / "L36_CALIBRATION_TABLES.json"
    calibration_path.write_text(json.dumps({
        "method": "class-conditioned empirical quantile mapping; fit only on train geometry and independent L34 samples",
        "training_data_manifest_sha256": manifest_hash,
        "inference_source_sha256": source_hash,
        "fit_config": {"calibration_samples_per_class": args.calibration_samples_per_class,
                       "sampling_steps": args.sampling_steps, "seed": args.seed,
                       "threads": torch.get_num_threads()},
        "checkpoints": calibration_tables,
    }), encoding="utf-8")

    flat_raw = [run["raw_max_feature_ks"] for run in runs]
    flat_cal = [run["calibrated_max_feature_ks"] for run in runs]
    summary = {
        "experiment": "L36 train-only quantile calibration of L34 geometry diffusion",
        "dataset": {"train_rows": len(train_rows), "heldout_rows": len(val_rows),
                    "manifest_sha256": manifest_hash},
        "sampling": {"calibration_samples_per_class": args.calibration_samples_per_class,
                     "evaluation_samples_per_class": args.evaluation_samples_per_class,
                     "sampling_steps": args.sampling_steps, "threads": torch.get_num_threads(),
                     "seed": args.seed},
        "inference_source_sha256": source_hash,
        "runs": runs,
        "multi_seed": {
            "model_seeds": [run["checkpoint_seed"] for run in runs],
            "mean_raw_max_feature_ks": float(np.mean(flat_raw)),
            "mean_calibrated_max_feature_ks": float(np.mean(flat_cal)),
            "mean_raw_feature_ks": float(np.mean([run["raw_mean_feature_ks"] for run in runs])),
            "mean_calibrated_feature_ks": float(np.mean([run["calibrated_mean_feature_ks"] for run in runs])),
            "mean_feature_ks_relative_reduction_percent": float(
                (np.mean([run["raw_mean_feature_ks"] for run in runs]) -
                 np.mean([run["calibrated_mean_feature_ks"] for run in runs])) /
                max(np.mean([run["raw_mean_feature_ks"] for run in runs]), 1e-12) * 100.0
            ),
            "max_raw_feature_ks": max(flat_raw), "max_calibrated_feature_ks": max(flat_cal),
            "calibration_improved_every_seed": all(cal < raw for cal, raw in zip(flat_cal, flat_raw)),
            "all_calibrated_centers_in_support": all(run["sample_support"]["all_calibrated_centers_in_22_42"] for run in runs),
            "all_calibrated_sizes_in_training_support": all(run["sample_support"]["all_calibrated_extents_observed_in_training"] for run in runs),
        },
        "calibration_tables": str(calibration_path.resolve()),
        "comparison_grid": str(grid_path.resolve()),
        "elapsed_seconds": time.perf_counter() - eval_started,
    }
    (args.output_dir / "L36_RESULTS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"multi_seed": summary["multi_seed"],
                      "elapsed_seconds": summary["elapsed_seconds"],
                      "grid": summary["comparison_grid"],
                      "calibration_tables": summary["calibration_tables"]}, indent=2))


if __name__ == "__main__":
    main()
