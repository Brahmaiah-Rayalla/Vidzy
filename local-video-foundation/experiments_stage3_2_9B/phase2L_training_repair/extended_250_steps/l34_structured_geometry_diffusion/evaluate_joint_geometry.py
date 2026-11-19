"""Compare L34 joint geometry with marginal and factorized train-only samplers."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from calibrate_geometry_sampling import (
    BACKGROUND, COLOR_NAMES, COLORS, DATA_DIR, HERE, ROOT, SHAPE_NAMES, SIZE,
    GeometryDenoiser, calibrate_class, decode, encoded_row, geometry_metrics,
    sample_vectors,
)


def energy_distance(reference, generated):
    reference, generated = np.asarray(reference, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    cross = np.sqrt(np.maximum(((reference[:, None, :] - generated[None, :, :]) ** 2).sum(-1), 0.0))
    within_ref = np.sqrt(np.maximum(((reference[:, None, :] - reference[None, :, :]) ** 2).sum(-1), 0.0))
    within_gen = np.sqrt(np.maximum(((generated[:, None, :] - generated[None, :, :]) ** 2).sum(-1), 0.0))
    return float(2.0 * cross.mean() - within_ref.mean() - within_gen.mean())


def correlation_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std(axis=0)
    active = np.where(std > 1e-10)[0]
    if len(active) < 2:
        return np.empty((0, 0))
    return np.corrcoef(values[:, active], rowvar=False)


def correlation_gap(reference, generated):
    ref_corr, gen_corr = correlation_matrix(reference), correlation_matrix(generated)
    if ref_corr.shape != gen_corr.shape or ref_corr.size == 0:
        return 0.0
    offdiag = ~np.eye(ref_corr.shape[0], dtype=bool)
    return float(np.mean(np.abs(ref_corr[offdiag] - gen_corr[offdiag])))


def factorized_samples(class_id, target_vectors, count, seed):
    """Draw independent class-conditional marginals estimated from train rows."""
    rng = np.random.default_rng(seed)
    rows = target_vectors[class_id]
    out = np.empty((count, 5), dtype=np.float32)
    for dimension in range(3):
        out[:, dimension] = rows[rng.integers(0, len(rows), size=count), dimension]
    if class_id % 2:
        angle_rows = rows[rng.integers(0, len(rows), size=count)]
        out[:, 3:5] = angle_rows[:, 3:5]
    else:
        out[:, 3], out[:, 4] = 1.0, 0.0
    return out


def render(geometry, scale=3):
    high = Image.new("RGB", (SIZE * scale, SIZE * scale), BACKGROUND)
    draw = ImageDraw.Draw(high)
    cx, cy = geometry["cx"] * scale, geometry["cy"] * scale
    extent = geometry["extent"] * scale
    color = COLORS[geometry["color_id"]]
    if geometry["shape_id"] == 0:
        draw.ellipse((cx - extent, cy - extent, cx + extent, cy + extent), fill=color)
    else:
        radius, angle = extent / np.sqrt(3.0), geometry["angle_rad"]
        points = [(cx + radius * np.cos(angle + 2.0 * np.pi * k / 3.0),
                   cy + radius * np.sin(angle + 2.0 * np.pi * k / 3.0)) for k in range(3)]
        draw.polygon(points, fill=color)
    return high.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=HERE / "l37_joint_geometry")
    parser.add_argument("--calibration_samples_per_class", type=int, default=512)
    parser.add_argument("--evaluation_samples_per_class", type=int, default=100)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3700)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    data_manifest = DATA_DIR / "manifest.jsonl"
    rows = [json.loads(line) for line in data_manifest.read_text(encoding="utf-8").splitlines()]
    train_rows = [row for row in rows if row["file"].startswith("train_")]
    val_rows = [row for row in rows if row["file"].startswith("val_")]
    train_by_class = {class_id: np.stack([encoded_row(row) for row in train_rows
                                          if int(row["class_id"]) == class_id])
                      for class_id in range(6)}
    val_by_class = {class_id: np.stack([encoded_row(row) for row in val_rows
                                        if int(row["class_id"]) == class_id])
                    for class_id in range(6)}
    manifest_hash = hashlib.sha256(data_manifest.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started = time.perf_counter()
    run_results, first_grid = [], None

    for checkpoint_path in args.checkpoints:
        checkpoint_path = checkpoint_path.resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config_bytes = json.dumps(
            checkpoint.get("config", {}), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        model = GeometryDenoiser()
        model.load_state_dict(checkpoint["ema_model"])
        model.eval()
        alpha_bar = torch.cumprod(1.0 - torch.linspace(0.0001, 0.02, 1000), dim=0)
        model_seed = int(checkpoint["config"]["training"]["seed"])
        reference_classes, reference_vectors = sample_vectors(
            model, alpha_bar, 6, args.calibration_samples_per_class,
            args.seed + model_seed * 13, args.sampling_steps,
        )
        reference_by_class = {class_id: reference_vectors[reference_classes == class_id]
                              for class_id in range(6)}
        eval_classes, raw_vectors = sample_vectors(
            model, alpha_bar, 6, args.evaluation_samples_per_class,
            args.seed + model_seed * 17 + 1, args.sampling_steps,
        )
        calibrated_vectors = raw_vectors.copy()
        for class_id in range(6):
            selected = eval_classes == class_id
            calibrated_vectors[selected] = calibrate_class(
                class_id, raw_vectors[selected], reference_by_class, train_by_class,
                args.seed + model_seed * 31 + class_id,
            )
        factorized_by_class = {
            class_id: factorized_samples(class_id, train_by_class,
                                         args.evaluation_samples_per_class,
                                         args.seed + model_seed * 47 + class_id)
            for class_id in range(6)
        }
        methods = {"raw_l34": {}, "marginal_calibrated": {}, "factorized_train_prior": {}}
        grid_examples = []
        for class_id in range(6):
            raw_class = raw_vectors[eval_classes == class_id]
            calibrated_class = calibrated_vectors[eval_classes == class_id]
            factorized_class = factorized_by_class[class_id]
            for name, vectors in (("raw_l34", raw_class),
                                  ("marginal_calibrated", calibrated_class),
                                  ("factorized_train_prior", factorized_class)):
                generated = [decode(class_id, vector) for vector in vectors]
                per_feature = geometry_metrics(val_rows, class_id, generated)
                heldout = val_by_class[class_id]
                active_dims = (0, 1, 2) if class_id % 2 == 0 else (0, 1, 2, 3, 4)
                methods[name][str(class_id)] = {
                    "feature_ks": {feature: row["ks_distance"] for feature, row in per_feature.items()},
                    "mean_feature_ks": float(np.mean([row["ks_distance"] for row in per_feature.values()])),
                    "max_feature_ks": max(row["ks_distance"] for row in per_feature.values()),
                    "multivariate_energy_distance": energy_distance(
                        heldout[:, active_dims], vectors[:, active_dims]),
                    "correlation_matrix_mae_vs_heldout": correlation_gap(
                        heldout[:, active_dims], vectors[:, active_dims]),
                }
            if first_grid is None:
                grid_examples.append((class_id,
                                      decode(class_id, raw_class[0]),
                                      decode(class_id, calibrated_class[0]),
                                      decode(class_id, factorized_class[0])))
        if first_grid is None:
            first_grid = grid_examples
        method_summary = {}
        for name, rows_by_class in methods.items():
            feature_means = [v["mean_feature_ks"] for v in rows_by_class.values()]
            energy = [v["multivariate_energy_distance"] for v in rows_by_class.values()]
            corr = [v["correlation_matrix_mae_vs_heldout"] for v in rows_by_class.values()]
            method_summary[name] = {
                "mean_feature_ks": float(np.mean(feature_means)),
                "mean_class_max_feature_ks": float(np.mean([v["max_feature_ks"] for v in rows_by_class.values()])),
                "mean_multivariate_energy_distance": float(np.mean(energy)),
                "mean_correlation_matrix_mae_vs_heldout": float(np.mean(corr)),
                "per_class": rows_by_class,
            }
        run_results.append({
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "checkpoint_config_sha256": hashlib.sha256(checkpoint_config_bytes).hexdigest(),
            "checkpoint_seed": model_seed,
            "methods": method_summary,
        })

    method_names = ("raw_l34", "marginal_calibrated", "factorized_train_prior")
    aggregate = {}
    for name in method_names:
        aggregate[name] = {
            metric: float(np.mean([run["methods"][name][metric] for run in run_results]))
            for metric in ("mean_feature_ks", "mean_class_max_feature_ks",
                           "mean_multivariate_energy_distance",
                           "mean_correlation_matrix_mae_vs_heldout")
        }
    aggregate["factorized_train_prior"]["beats_marginal_calibration_on_joint_energy_all_seeds"] = all(
        run["methods"]["factorized_train_prior"]["mean_multivariate_energy_distance"] <
        run["methods"]["marginal_calibrated"]["mean_multivariate_energy_distance"]
        for run in run_results
    )

    tile, row_h, label_w = 112, 126, 126
    grid = Image.new("RGB", (label_w + 3 * tile, 48 + 6 * row_h), "white")
    draw = ImageDraw.Draw(grid)
    for col, title in enumerate(("L34 raw", "Marginal calibrated", "Factorized train prior")):
        draw.text((label_w + col * tile, 7), title, fill="black")
    for class_id, raw_item, calibrated_item, factorized_item in first_grid:
        y = 48 + class_id * row_h
        draw.text((4, y + 42), f"{COLOR_NAMES[class_id // 2]}\n{SHAPE_NAMES[class_id % 2]}", fill="black")
        for col, item in enumerate((raw_item, calibrated_item, factorized_item)):
            x = label_w + col * tile
            preview = render(item).resize((tile, tile), Image.Resampling.NEAREST)
            grid.paste(preview, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(50, 120, 50), width=1)
    grid_path = args.output_dir / "L37_joint_geometry_comparison.png"
    grid.save(grid_path)
    result = {
        "experiment": "L37 joint geometry evaluation: raw diffusion vs marginal vs factorized train prior",
        "dataset": {"train_rows": len(train_rows), "heldout_rows": len(val_rows),
                    "manifest_sha256": manifest_hash},
        "sampling": {"samples_per_class": args.evaluation_samples_per_class,
                     "calibration_samples_per_class": args.calibration_samples_per_class,
                     "sampling_steps": args.sampling_steps, "threads": torch.get_num_threads(),
                     "seed": args.seed},
        "inference_source_sha256": source_hash,
        "runs": run_results,
        "three_seed_aggregate": aggregate,
        "elapsed_seconds": time.perf_counter() - started,
        "comparison_grid": str(grid_path.resolve()),
        "interpretation_note": "All visible outputs use the canonical renderer. Distribution metrics evaluate geometry only; the factorized prior is a train-fitted statistical sampler, not a neural generator.",
    }
    out = args.output_dir / "L37_RESULTS.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"three_seed_aggregate": aggregate,
                      "elapsed_seconds": result["elapsed_seconds"],
                      "comparison_grid": result["comparison_grid"]}, indent=2))


if __name__ == "__main__":
    main()
