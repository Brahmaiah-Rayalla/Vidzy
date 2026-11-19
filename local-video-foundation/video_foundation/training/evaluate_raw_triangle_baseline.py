"""Repeatable, paired CPU evaluation for raw triangle diffusion checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from video_foundation.data.dataset import SyntheticImageDataset
from video_foundation.diffusion.process import DiffusionProcess
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.inference.shape_rendering import render_canonical_shape
from video_foundation.training.evaluate_size_conditioned import (
    binary_boundary,
    build_model,
    centered_triangle_metrics,
    dilate_binary,
    palette_mask,
    triangle_boundary_metrics,
)
from video_foundation.utils.experiment_metadata import build_evaluation_manifest


SIZE_NAMES = ("small", "medium", "large")
SIZE_EXTENTS_64 = (15, 29, 42)


def raw_geometry_metrics(mask: np.ndarray, target: np.ndarray) -> dict:
    """Report both visible alignment and silhouette-only metrics."""
    mask = np.asarray(mask, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(mask, target).sum())
    union = int(np.logical_or(mask, target).sum())
    ys, xs = np.where(mask)
    target_y, target_x = np.where(target)
    if len(xs) and len(target_x):
        centroid_error = float(np.hypot(xs.mean() - target_x.mean(), ys.mean() - target_y.mean()))
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
    else:
        centroid_error = None
        bbox_width = bbox_height = 0
    target_boundary = binary_boundary(target)
    pred_boundary = binary_boundary(mask)
    pred_count = int(pred_boundary.sum())
    target_count = int(target_boundary.sum())
    if pred_count and target_count:
        target_near = dilate_binary(target_boundary, 1)
        pred_near = dilate_binary(pred_boundary, 1)
        precision = float((pred_boundary & target_near).sum() / pred_count)
        recall = float((target_boundary & pred_near).sum() / target_count)
        boundary_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    else:
        boundary_f1 = 0.0
    centered_iou, area_ratio = centered_triangle_metrics(mask, target)
    return {
        "foreground_iou_unaligned": float(intersection / max(union, 1)),
        "centered_triangle_iou": centered_iou,
        "centroid_error_pixels": centroid_error,
        "foreground_area_ratio": area_ratio,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "boundary_f1_unaligned_1px": boundary_f1,
        "boundary_f1_centered_1px": triangle_boundary_metrics(mask, target, tolerance=1)["f1"],
    }


def center_mask_bbox(mask: np.ndarray, target_bbox_center: tuple[float, float]) -> np.ndarray:
    """Translate a canonical mask to match the dataset's bounding-box center convention."""
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if not len(xs):
        return mask.copy()
    current_center_x = (float(xs.min()) + float(xs.max())) / 2.0
    current_center_y = (float(ys.min()) + float(ys.max())) / 2.0
    shift_x = int(round(float(target_bbox_center[0]) - current_center_x))
    shift_y = int(round(float(target_bbox_center[1]) - current_center_y))
    height, width = mask.shape
    shifted = np.zeros_like(mask)
    dst_y0, dst_y1 = max(0, shift_y), min(height, height + shift_y)
    dst_x0, dst_x1 = max(0, shift_x), min(width, width + shift_x)
    src_y0, src_y1 = max(0, -shift_y), min(height, height - shift_y)
    src_x0, src_x1 = max(0, -shift_x), min(width, width - shift_x)
    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _rgb_from_sample(sample: torch.Tensor) -> np.ndarray:
    return ((sample.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1.0) * 127.5).round().astype(np.uint8)


def _save_sample(sample: torch.Tensor, path: Path) -> None:
    rgb = _rgb_from_sample(sample)
    Image.fromarray(rgb).save(path)
    Image.fromarray(rgb).resize(
        (rgb.shape[1] * 4, rgb.shape[0] * 4), Image.Resampling.NEAREST
    ).save(path.with_name(f"{path.stem}_preview_4x.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Prepared validation split")
    parser.add_argument("--baseline_checkpoint", required=True, help="v54 calibrated raw-diffusion baseline")
    parser.add_argument("--small_checkpoint", required=True, help="Selected small-shape checkpoint (v65)")
    parser.add_argument("--medium_large_checkpoint", required=True,
                        help="Selected medium/large checkpoint (v59)")
    parser.add_argument("--medium_checkpoint", default=None,
                        help="Optional medium-size route; defaults to --medium_large_checkpoint")
    parser.add_argument("--large_checkpoint", default=None,
                        help="Optional large-size route; defaults to --medium_large_checkpoint")
    parser.add_argument("--candidate_checkpoint", default=None,
                        help="Optional all-size checkpoint measured as a third raw-diffusion route")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", type=int, default=128,
                        help="Maximum held-out denoising examples; use 0 to evaluate the full split")
    parser.add_argument("--sampling_steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234, help="Held-out noise seed")
    parser.add_argument("--generation_seeds", type=int, nargs="+", default=(1234, 2345, 3456, 4567),
                        help="Matched generation seeds used for every checkpoint and size")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    if args.sampling_steps < 2:
        parser.error("--sampling_steps must be at least 2")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not args.generation_seeds:
        parser.error("--generation_seeds must contain at least one seed")

    evaluation_started = time.perf_counter()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    checkpoint_paths = {
        "baseline_v54": args.baseline_checkpoint,
        "selected_small_v65": args.small_checkpoint,
        "selected_medium_large_v59": args.medium_large_checkpoint,
    }
    if args.large_checkpoint:
        checkpoint_paths["selected_large"] = args.large_checkpoint
    if args.medium_checkpoint:
        checkpoint_paths["selected_medium"] = args.medium_checkpoint
    if args.candidate_checkpoint:
        checkpoint_paths["candidate"] = args.candidate_checkpoint
    checkpoint_payloads = {
        name: torch.load(path, map_location=device, weights_only=False)
        for name, path in checkpoint_paths.items()
    }
    models = {}
    configs = {}
    for name, payload in checkpoint_payloads.items():
        config = payload["config"]
        configs[name] = config
        model = build_model(config)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models[name] = model

    image_sizes = {int(config["model"]["image_size"]) for config in configs.values()}
    if len(image_sizes) != 1:
        raise ValueError(f"All compared checkpoints must use one image size; got {sorted(image_sizes)}")
    image_size = image_sizes.pop()
    baseline_uses_size = int(configs["baseline_v54"]["model"].get("num_size_classes", 0)) > 0
    route_by_size = {
        0: "selected_small_v65",
        1: "selected_medium" if args.medium_checkpoint else "selected_medium_large_v59",
        2: "selected_large" if args.large_checkpoint else "selected_medium_large_v59",
    }
    route_uses_size = {
        name: int(configs[name]["model"].get("num_size_classes", 0)) > 0
        for name in checkpoint_payloads
    }

    dataset = SyntheticImageDataset(
        args.data_dir, image_size=image_size, normalize=True, cache_in_memory=True,
        return_label=True, return_color_label=True, return_size_label=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    process = DiffusionProcess(GaussianDiffusionSchedule(
        num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear",
    ))
    sampler = DiffusionSampler(
        num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear",
    )
    heldout_generator = torch.Generator().manual_seed(args.seed)
    heldout_by_checkpoint = {
        name: {size: [] for size in SIZE_NAMES}
        for name in ("baseline_v54", "selected_route")
    }
    if args.candidate_checkpoint:
        heldout_by_checkpoint["candidate"] = {size: [] for size in SIZE_NAMES}
    heldout_counts = {size: 0 for size in SIZE_NAMES}
    target_bbox_centers_by_size = {size: [] for size in SIZE_NAMES}
    for sample_index, (clean, shapes, colors, sizes) in enumerate(loader):
        if args.samples and sample_index >= args.samples:
            break
        size_id = int(sizes.item())
        size_name = SIZE_NAMES[size_id]
        if int(shapes.item()) == 2:
            clean_rgb = _rgb_from_sample(clean[0])
            foreground = clean_rgb.max(axis=-1) > 5
            target_y, target_x = np.where(foreground)
            if len(target_x):
                target_bbox_centers_by_size[size_name].append((
                    (float(target_x.min()) + float(target_x.max())) / 2.0,
                    (float(target_y.min()) + float(target_y.max())) / 2.0,
                ))
        timestep = torch.randint(0, 1000, (1,), generator=heldout_generator)
        noise = torch.randn(clean.shape, generator=heldout_generator)
        noisy, noise = process.q_sample(clean, timestep, noise)
        _, alpha, one_minus = process.schedule.get(timestep, clean.device)
        target_v = alpha.sqrt() * noise - one_minus.sqrt() * clean
        candidate_name = route_by_size[size_id]
        candidate_size = sizes if route_uses_size[candidate_name] else None
        baseline_size = sizes if baseline_uses_size else None
        with torch.no_grad():
            baseline_v = models["baseline_v54"](noisy, timestep, shapes, colors, baseline_size)
            selected_v = models[candidate_name](noisy, timestep, shapes, colors, candidate_size)
        heldout_by_checkpoint["baseline_v54"][size_name].append(
            float((baseline_v - target_v).square().mean().item())
        )
        heldout_by_checkpoint["selected_route"][size_name].append(
            float((selected_v - target_v).square().mean().item())
        )
        if args.candidate_checkpoint:
            with torch.no_grad():
                candidate_size = sizes if route_uses_size["candidate"] else None
                candidate_v = models["candidate"](noisy, timestep, shapes, colors, candidate_size)
            heldout_by_checkpoint["candidate"][size_name].append(
                float((candidate_v - target_v).square().mean().item())
            )
        heldout_counts[size_name] += 1

    target_bbox_center_by_size = {}
    for size_name, centers in target_bbox_centers_by_size.items():
        if not centers:
            raise ValueError(
                f"No held-out triangle examples found for the {size_name} size route; "
                "cannot establish the dataset's target center convention."
            )
        target_bbox_center_by_size[size_name] = np.asarray(centers, dtype=np.float64).mean(axis=0)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    for generation_seed in args.generation_seeds:
        for size_id, size_name in enumerate(SIZE_NAMES):
            # One distinct but repeatable noise tensor per (seed, size), shared by both models.
            noise_seed = (int(generation_seed) * 1_000_003 + size_id * 97_409) % (2**63 - 1)
            noise_generator = torch.Generator().manual_seed(noise_seed)
            initial_noise = torch.randn((1, 3, image_size, image_size), generator=noise_generator)
            size_tensor = torch.tensor([size_id], dtype=torch.long)
            shape_tensor = torch.tensor([2], dtype=torch.long)  # triangle
            color_tensor = torch.tensor([2], dtype=torch.long)  # blue
            target_extent = max(1, round(SIZE_EXTENTS_64[size_id] * image_size / 64))
            canonical_mask = render_canonical_shape(
                2, ((image_size - 1) / 2.0, (image_size - 1) / 2.0), target_extent, image_size
            )
            target_bbox_center = target_bbox_center_by_size[size_name]
            target_mask = center_mask_bbox(canonical_mask, target_bbox_center)
            comparison_routes = [
                ("baseline_v54", "baseline_v54"),
                ("selected_route", route_by_size[size_id]),
            ]
            if args.candidate_checkpoint:
                comparison_routes.append(("candidate", "candidate"))
            for model_label, checkpoint_name in comparison_routes:
                model_size = size_tensor if route_uses_size[checkpoint_name] else None
                started = time.perf_counter()
                with torch.no_grad():
                    sample = sampler.sample(
                        models[checkpoint_name], batch_size=1, channels=3,
                        image_size=image_size, num_sampling_steps=args.sampling_steps,
                        device=device, initial_noise=initial_noise.clone(),
                        prediction_type="v_prediction", recompute_noise_after_clipping=True,
                        class_labels=shape_tensor, color_labels=color_tensor, size_labels=model_size,
                    )[0]
                generation_seconds = time.perf_counter() - started
                rgb = _rgb_from_sample(sample)
                mask = palette_mask(rgb)
                geometry = raw_geometry_metrics(mask, target_mask)
                artifact_path = output_dir / f"{model_label}_seed_{generation_seed}_{size_name}.png"
                _save_sample(sample, artifact_path)
                paired_rows.append({
                    "checkpoint_id": checkpoint_name,
                    "comparison_role": model_label,
                    "generation_mode": "raw_diffusion",
                    "seed": int(generation_seed),
                    "noise_seed": int(noise_seed),
                    "shape": "triangle",
                    "shape_id": 2,
                    "color": "blue",
                    "color_id": 2,
                    "size": size_name,
                    "size_id": size_id,
                    "sampling_steps": args.sampling_steps,
                    "model_image_size": image_size,
                    "target_bbox_center_pixels": [
                        float(target_bbox_center[0]), float(target_bbox_center[1])
                    ],
                    "target_centering_convention": "mean held-out triangle bounding-box center",
                    "sample_generation_seconds": round(generation_seconds, 6),
                    "sample_path": str(artifact_path),
                    "preview_4x_path": str(artifact_path.with_name(f"{artifact_path.stem}_preview_4x.png")),
                    **geometry,
                })
                print(
                    f"{model_label} seed={generation_seed} size={size_name} "
                    f"IoU={geometry['foreground_iou_unaligned']:.4f} "
                    f"centered_IoU={geometry['centered_triangle_iou']:.4f} "
                    f"boundary_F1={geometry['boundary_f1_unaligned_1px']:.4f} "
                    f"time={generation_seconds:.2f}s"
                )

    representative_seed = int(args.generation_seeds[0])
    tile_size, header_size, gap = image_size * 4, 28, 8
    grid_routes = [
        ("v54 baseline", "baseline_v54"),
        ("selected route", "selected_route"),
    ]
    if args.candidate_checkpoint:
        grid_routes.append(("candidate", "candidate"))
    grid = Image.new(
        "RGB",
        (3 * tile_size + 4 * gap,
         len(grid_routes) * (tile_size + header_size) + (len(grid_routes) + 1) * gap),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(grid)
    for row_index, (label, comparison_role) in enumerate(grid_routes):
        for column_index, size_name in enumerate(SIZE_NAMES):
            x = gap + column_index * (tile_size + gap)
            y = gap + row_index * (tile_size + header_size + gap)
            draw.text((x + 5, y + 6), f"{label} | {size_name}", fill=(240, 240, 240))
            sample_path = output_dir / f"{comparison_role}_seed_{representative_seed}_{size_name}_preview_4x.png"
            preview = Image.open(sample_path).convert("RGB")
            grid.paste(preview, (x, y + header_size))
    representative_grid_path = output_dir / f"paired_seed_{representative_seed}_raw_comparison_grid.png"
    grid.save(representative_grid_path)

    elapsed_seconds = time.perf_counter() - evaluation_started
    source_paths = [
        __file__,
        ROOT / "video_foundation" / "training" / "evaluate_size_conditioned.py",
        ROOT / "video_foundation" / "models" / "unet.py",
        ROOT / "video_foundation" / "diffusion" / "process.py",
        ROOT / "video_foundation" / "diffusion" / "sampler.py",
        ROOT / "video_foundation" / "inference" / "shape_rendering.py",
        ROOT / "video_foundation" / "data" / "dataset.py",
        ROOT / "video_foundation" / "utils" / "experiment_metadata.py",
    ]
    manifest = build_evaluation_manifest(
        task="raw_triangle_per_size_paired_baseline",
        args={**vars(args), "generation_mode": "raw_diffusion"},
        checkpoint_paths=checkpoint_paths,
        checkpoint_payloads=checkpoint_payloads,
        data_dir=args.data_dir,
        source_paths=source_paths,
        elapsed_seconds=elapsed_seconds,
        torch_version=torch.__version__,
        cpu_threads=torch.get_num_threads(),
    )
    for row in paired_rows:
        checkpoint = manifest["checkpoints"][row["checkpoint_id"]]
        row["checkpoint_path"] = checkpoint["path"]
        row["checkpoint_sha256"] = checkpoint["sha256"]
        row["checkpoint_config_sha256"] = checkpoint["config_sha256"]
        row["checkpoint_step"] = checkpoint["step"]

    metric_names = (
        "foreground_iou_unaligned", "centered_triangle_iou", "centroid_error_pixels",
        "foreground_area_ratio", "boundary_f1_unaligned_1px", "boundary_f1_centered_1px",
    )
    summary = {}
    comparison_roles = ["baseline_v54", "selected_route"]
    if args.candidate_checkpoint:
        comparison_roles.append("candidate")
    for checkpoint_id in comparison_roles:
        summary[checkpoint_id] = {}
        for size_name in SIZE_NAMES:
            group = [row for row in paired_rows
                     if row["comparison_role"] == checkpoint_id and row["size"] == size_name]
            summary[checkpoint_id][size_name] = {
                metric_name: {
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values)) if values else None,
                }
                for metric_name in metric_names
                for values in [[
                    row[metric_name] for row in group
                    if row[metric_name] is not None and math.isfinite(float(row[metric_name]))
                ]]
            }
            summary[checkpoint_id][size_name]["num_seeds"] = len(group)

    heldout_summary = {}
    for checkpoint_id, by_size in heldout_by_checkpoint.items():
        heldout_summary[checkpoint_id] = {
            size_name: {
                "v_mse": float(np.mean(values)) if values else None,
                "samples": len(values),
            }
            for size_name, values in by_size.items()
        }
    paired_metrics = {
        "schema_version": 2,
        "generation_mode": "raw_diffusion",
        "target_centering_convention": "mean held-out triangle bounding-box center by size",
        "target_bbox_center_pixels_by_size": {
            name: [float(value) for value in center]
            for name, center in target_bbox_center_by_size.items()
        },
        "generation_seeds": [int(seed) for seed in args.generation_seeds],
        "sampling_steps": args.sampling_steps,
        "sample_count_per_comparison": len(args.generation_seeds) * len(SIZE_NAMES),
        "total_generated_samples": len(paired_rows),
        "metrics_by_sample": paired_rows,
        "summary_by_checkpoint_and_size": summary,
        "heldout_v_mse_by_size": heldout_summary,
        "provenance_manifest": str(output_dir / "run_manifest.json"),
    }
    evaluation_metrics = {
        "task": "raw_triangle_per_size_paired_baseline",
        "generation_mode": "raw_diffusion",
        "target_centering_convention": "mean held-out triangle bounding-box center by size",
        "target_bbox_center_pixels_by_size": {
            name: [float(value) for value in center]
            for name, center in target_bbox_center_by_size.items()
        },
        "generation_seeds": [int(seed) for seed in args.generation_seeds],
        "heldout_seed": args.seed,
        "heldout_v_mse_by_checkpoint_and_size": heldout_summary,
        "generation_summary_by_checkpoint_and_size": summary,
        "elapsed_seconds_excluding_manifest_hashing": round(elapsed_seconds, 6),
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "validation_samples_by_size": heldout_counts,
        "representative_grid_path": str(representative_grid_path),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "paired_generation_metrics.json").write_text(
        json.dumps(paired_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(evaluation_metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved raw-only paired metrics and provenance to {output_dir}")
    print(f"Elapsed evaluation: {elapsed_seconds:.2f}s on CPU ({torch.get_num_threads()} threads)")


if __name__ == "__main__":
    main()
