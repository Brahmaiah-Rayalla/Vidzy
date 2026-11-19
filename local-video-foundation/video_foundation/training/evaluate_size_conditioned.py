"""Paired denoising and generation checks for size-conditioned checkpoints."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from video_foundation.data.dataset import SyntheticImageDataset
from video_foundation.models.unet import UNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.inference.shape_rendering import render_canonical_shape
from video_foundation.utils.experiment_metadata import build_evaluation_manifest


def build_model(config, size_embedding_scale=None):
    model_config = config["model"]
    model = UNet(
        in_channels=model_config["in_channels"], model_channels=model_config["model_channels"],
        channel_mult=model_config["channel_mult"], num_res_blocks=model_config["num_res_blocks"],
        num_classes=model_config.get("num_classes", 0),
        num_color_classes=model_config.get("num_color_classes", 0),
        num_size_classes=model_config.get("num_size_classes", 0),
        spatial_size_conditioning=model_config.get("spatial_size_conditioning", False),
        high_resolution_refinement=model_config.get("high_resolution_refinement", False),
        size_embedding_scale=(model_config.get("size_embedding_scale", 1.0)
                              if size_embedding_scale is None else size_embedding_scale),
        middle_dilation=int(model_config.get("middle_dilation", 1)),
    )
    return model


def palette_foreground_mask(image):
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    return (blue > red * 1.25) & (blue > green * 1.15) & (blue > 55)


def component_sizes(mask):
    """Return every 8-connected foreground component size, including speckles."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask)
    sizes = []
    for y, x in zip(*np.where(mask)):
        if visited[y, x]:
            continue
        visited[y, x] = True
        stack, component = [(int(y), int(x))], []
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        sizes.append(len(component))
    return sorted(sizes, reverse=True)


def palette_mask(image):
    """Largest blue component, retained for legacy shape-only metrics."""
    mask = palette_foreground_mask(image)
    sizes = component_sizes(mask)
    clean = np.zeros_like(mask)
    if sizes:
        # Reconstruct the largest component only for the legacy centered shape
        # scores. New raw metrics below explicitly retain every component.
        height, width = mask.shape
        visited = np.zeros_like(mask)
        for y, x in zip(*np.where(mask)):
            if visited[y, x]:
                continue
            visited[y, x] = True
            stack, component = [(int(y), int(x))], []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(component) == sizes[0]:
                yy, xx = zip(*component)
                clean[np.asarray(yy), np.asarray(xx)] = True
                break
    return clean


def centered_triangle_metrics(mask, target):
    """Compare a generated silhouette with the canonical triangle after center alignment."""
    aligned = center_align_mask(mask, target)
    if not np.any(mask) or not np.any(target):
        return 0.0, float(mask.sum() / max(int(target.sum()), 1))
    intersection = np.logical_and(aligned, target).sum()
    union = np.logical_or(aligned, target).sum()
    iou = float(intersection / max(int(union), 1))
    area_ratio = float(mask.sum() / max(int(target.sum()), 1))
    return iou, area_ratio


def center_align_mask(mask, target):
    """Translate a mask to the target centroid without scaling or rotation."""
    ys, xs = np.where(mask)
    target_y, target_x = np.where(target)
    if not len(xs) or not len(target_x):
        return np.zeros_like(mask, dtype=bool)
    shift_x = int(round(float(target_x.mean() - xs.mean())))
    shift_y = int(round(float(target_y.mean() - ys.mean())))
    aligned = np.zeros_like(mask, dtype=bool)
    shifted_x, shifted_y = xs + shift_x, ys + shift_y
    valid = ((shifted_x >= 0) & (shifted_x < mask.shape[1])
             & (shifted_y >= 0) & (shifted_y < mask.shape[0]))
    aligned[shifted_y[valid], shifted_x[valid]] = True
    return aligned


def binary_boundary(mask):
    """Return the one-pixel inner boundary of a binary mask using NumPy only."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(mask)
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask)
    for y_offset in range(3):
        for x_offset in range(3):
            eroded &= padded[y_offset:y_offset + height, x_offset:x_offset + width]
    return mask & ~eroded


def dilate_binary(mask, radius=1):
    """Dilate a binary mask with a square neighborhood, without wraparound."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask)
    window = 2 * radius + 1
    for y_offset in range(window):
        for x_offset in range(window):
            dilated |= padded[y_offset:y_offset + height, x_offset:x_offset + width]
    return dilated


def triangle_boundary_metrics(mask, target, tolerance=1):
    """Boundary precision/recall/F1 after centroid alignment, in pixel tolerance."""
    predicted_boundary = binary_boundary(center_align_mask(mask, target))
    target_boundary = binary_boundary(target)
    predicted_count = int(predicted_boundary.sum())
    target_count = int(target_boundary.sum())
    if predicted_count == 0 or target_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tolerance_pixels": tolerance}
    target_neighborhood = dilate_binary(target_boundary, tolerance)
    predicted_neighborhood = dilate_binary(predicted_boundary, tolerance)
    precision = float((predicted_boundary & target_neighborhood).sum() / predicted_count)
    recall = float((target_boundary & predicted_neighborhood).sum() / target_count)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tolerance_pixels": tolerance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Prepared validation image split")
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--size_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sampling_steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation_seeds", type=int, nargs="+", default=None,
                        help="Paired seeds for raw sample comparison; defaults to --seed")
    parser.add_argument("--size_embedding_scale", type=float, default=None,
                        help="Optional inference-only override for size conditioning calibration")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    generation_seeds = args.generation_seeds or [args.seed]
    evaluation_started = time.perf_counter()
    torch.set_num_threads(4)
    baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
    size_checkpoint = torch.load(args.size_checkpoint, map_location="cpu", weights_only=False)
    baseline = build_model(baseline_checkpoint["config"])
    baseline.load_state_dict(baseline_checkpoint["model_state_dict"])
    conditioned = build_model(size_checkpoint["config"], args.size_embedding_scale)
    conditioned.load_state_dict(size_checkpoint["model_state_dict"])
    baseline.eval(); conditioned.eval()
    baseline_uses_size = int(baseline_checkpoint["config"]["model"].get("num_size_classes", 0)) > 0
    image_size = size_checkpoint["config"]["model"]["image_size"]
    dataset = SyntheticImageDataset(
        args.data_dir, image_size=image_size, normalize=True,
        cache_in_memory=True, return_label=True, return_color_label=True, return_size_label=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # Validation files are commonly written grouped by size. Taking the first
    # N records therefore silently measured only the smallest size. Select a
    # deterministic, approximately balanced subset across all available sizes.
    indices_by_size = {}
    for index, size_id in enumerate(dataset.size_labels):
        indices_by_size.setdefault(int(size_id), []).append(index)
    selection_rng = np.random.default_rng(args.seed)
    for indices in indices_by_size.values():
        selection_rng.shuffle(indices)
    target_sample_count = min(args.samples, len(dataset))
    selected_indices = []
    size_ids = sorted(indices_by_size)
    while len(selected_indices) < target_sample_count:
        added = False
        for size_id in size_ids:
            if indices_by_size[size_id] and len(selected_indices) < target_sample_count:
                selected_indices.append(indices_by_size[size_id].pop())
                added = True
        if not added:
            break
    selected_index_set = set(selected_indices)
    selected_size_counts = {str(size_id): 0 for size_id in size_ids}
    selected_sample_count = 0
    process = DiffusionProcess(GaussianDiffusionSchedule(
        num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear",
    ))
    generator = torch.Generator().manual_seed(args.seed)
    baseline_by_size = [[], [], []]
    conditioned_by_size = [[], [], []]
    baseline_by_region = {"foreground": [], "background": []}
    conditioned_by_region = {"foreground": [], "background": []}
    with torch.no_grad():
        for index, (clean, shapes, colors, sizes) in enumerate(loader):
            if selected_sample_count >= len(selected_index_set):
                break
            if index not in selected_index_set:
                continue
            timestep = torch.randint(0, 1000, (1,), generator=generator)
            noise = torch.randn(clean.shape, generator=generator)
            noisy, noise = process.q_sample(clean, timestep, noise)
            _, alpha, one_minus = process.schedule.get(timestep, clean.device)
            target_v = alpha.sqrt() * noise - one_minus.sqrt() * clean
            baseline_v = baseline(
                noisy, timestep, shapes, colors, sizes if baseline_uses_size else None
            )
            conditioned_v = conditioned(noisy, timestep, shapes, colors, sizes)
            size_id = int(sizes.item())
            selected_size_counts[str(size_id)] += 1
            selected_sample_count += 1
            baseline_by_size[size_id].append((baseline_v - target_v).square().mean().item())
            conditioned_by_size[size_id].append((conditioned_v - target_v).square().mean().item())
            foreground = (clean > -0.95).any(dim=1, keepdim=True)
            for prediction, regional_losses in (
                (baseline_v, baseline_by_region),
                (conditioned_v, conditioned_by_region),
            ):
                pixel_error = (prediction - target_v).square().mean(dim=1, keepdim=True)
                regional_losses["foreground"].append(pixel_error[foreground].mean().item())
                regional_losses["background"].append(pixel_error[~foreground].mean().item())

    schedule = GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear")
    sampler = DiffusionSampler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear")
    labels = torch.full((3,), 2, dtype=torch.long)  # triangle
    colors = torch.full((3,), 2, dtype=torch.long)  # blue
    sizes = torch.arange(3, dtype=torch.long)
    os.makedirs(args.output_dir, exist_ok=True)
    shape_metrics = []
    paired_shape_metrics = []
    size_extents = (15, 29, 42)
    model_entries = (
        ("baseline", baseline, sizes if baseline_uses_size else None),
        ("size_conditioned", conditioned, sizes),
    )
    for generation_seed in generation_seeds:
        torch.manual_seed(generation_seed)
        initial_noise = torch.randn(3, 3, image_size, image_size)
        seed_grids = {}
        for model_name, model, model_sizes in model_entries:
            with torch.no_grad():
                samples = sampler.sample(
                    model, batch_size=3, channels=3, image_size=image_size,
                    num_sampling_steps=args.sampling_steps, device=torch.device("cpu"),
                    initial_noise=initial_noise.clone(), prediction_type="v_prediction",
                    recompute_noise_after_clipping=True, class_labels=labels,
                    color_labels=colors, size_labels=model_sizes,
                )
            rendered = []
            for index, sample in enumerate(samples):
                rgb = ((sample.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
                all_mask = palette_foreground_mask(rgb)
                mask = palette_mask(rgb)
                yy, xx = np.where(mask)
                bbox = (int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)) if len(xx) else (0, 0)
                extent = max(1, round(size_extents[index] * image_size / 64))
                target_mask = render_canonical_shape(
                    2, ((image_size - 1) / 2.0, (image_size - 1) / 2.0), extent, image_size
                )
                triangle_iou, area_ratio = centered_triangle_metrics(mask, target_mask)
                boundary = triangle_boundary_metrics(mask, target_mask, tolerance=1)
                component_counts = component_sizes(all_mask)
                raw_foreground_pixels = int(all_mask.sum())
                largest_component_pixels = component_counts[0] if component_counts else 0
                target_pixels = max(int(target_mask.sum()), 1)
                raw_union = int(np.logical_or(all_mask, target_mask).sum())
                raw_intersection = int(np.logical_and(all_mask, target_mask).sum())
                target_y, target_x = np.where(target_mask)
                all_y, all_x = np.where(all_mask)
                centroid_error = (
                    float(np.hypot(all_x.mean() - target_x.mean(), all_y.mean() - target_y.mean()))
                    if len(all_x) and len(target_x) else float(image_size * 2)
                )
                metric = {
                    "checkpoint": model_name,
                    "seed": generation_seed,
                    "size": ("small", "medium", "large")[index],
                    "bbox_width": bbox[0], "bbox_height": bbox[1],
                    "centered_triangle_iou": triangle_iou,
                    "foreground_area_ratio": area_ratio,
                    "boundary_precision_1px": boundary["precision"],
                    "boundary_recall_1px": boundary["recall"],
                    "boundary_f1_1px": boundary["f1"],
                    "raw_all_components_iou_unaligned": raw_intersection / max(raw_union, 1),
                    "raw_all_components_area_ratio": raw_foreground_pixels / target_pixels,
                    "raw_centroid_error_pixels": centroid_error,
                    "raw_component_count_8connected": len(component_counts),
                    "raw_largest_component_fraction": largest_component_pixels / max(raw_foreground_pixels, 1),
                    "raw_disconnected_foreground_fraction": (
                        raw_foreground_pixels - largest_component_pixels
                    ) / max(raw_foreground_pixels, 1),
                }
                paired_shape_metrics.append(metric)
                if model_name == "size_conditioned" and generation_seed == generation_seeds[0]:
                    shape_metrics.append(metric)
                print(
                    f"{model_name} seed={generation_seed} size={metric['size']} "
                    f"IoU={triangle_iou:.4f} area={area_ratio:.3f} "
                    f"boundary_f1@1px={boundary['f1']:.4f} "
                    f"raw_iou={metric['raw_all_components_iou_unaligned']:.4f} "
                    f"raw_components={metric['raw_component_count_8connected']} "
                    f"raw_noise_fraction={metric['raw_disconnected_foreground_fraction']:.3f}"
                )
                image = Image.fromarray(rgb).resize(
                    (image_size * 4, image_size * 4), Image.Resampling.NEAREST
                )
                rendered.append(image)
                image.save(os.path.join(
                    args.output_dir,
                    f"{model_name}_seed_{generation_seed}_{('small','medium','large')[index]}.png",
                ))
            seed_grids[model_name] = rendered

        paired_grid = Image.new("RGB", (image_size * 12, image_size * 8), (24, 24, 24))
        for row, model_name in enumerate(("baseline", "size_conditioned")):
            for index, image in enumerate(seed_grids[model_name]):
                paired_grid.paste(image, (index * image_size * 4, row * image_size * 4))
        paired_grid.save(os.path.join(args.output_dir, f"paired_seed_{generation_seed}.png"))

        if generation_seed == generation_seeds[0]:
            baseline_grid = Image.new("RGB", (image_size * 12, image_size * 4), (24, 24, 24))
            conditioned_grid = Image.new("RGB", (image_size * 12, image_size * 4), (24, 24, 24))
            for index, image in enumerate(seed_grids["baseline"]):
                baseline_grid.paste(image, (index * image_size * 4, 0))
            for index, image in enumerate(seed_grids["size_conditioned"]):
                conditioned_grid.paste(image, (index * image_size * 4, 0))
            baseline_grid.save(os.path.join(args.output_dir, "baseline_size_samples.png"))
            conditioned_grid.save(os.path.join(args.output_dir, "size_samples.png"))

    with open(os.path.join(args.output_dir, "generated_shape_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(shape_metrics, handle, indent=2)
    heldout_by_size = {}
    for index, name in enumerate(("small", "medium", "large")):
        base = sum(baseline_by_size[index]) / max(len(baseline_by_size[index]), 1)
        sized = sum(conditioned_by_size[index]) / max(len(conditioned_by_size[index]), 1)
        heldout_by_size[name] = {
            "baseline_v_mse": base,
            "size_conditioned_v_mse": sized,
            "samples": len(conditioned_by_size[index]),
        }
        print(f"{name}: heldout_v_mse={base:.6f}->{sized:.6f} n={len(conditioned_by_size[index])}")
    heldout_by_region = {}
    for region in ("foreground", "background"):
        base = float(np.mean(baseline_by_region[region]))
        sized = float(np.mean(conditioned_by_region[region]))
        heldout_by_region[region] = {
            "baseline_v_mse": base,
            "size_conditioned_v_mse": sized,
            "samples": len(conditioned_by_region[region]),
        }
        print(f"{region}: heldout_v_mse={base:.6f}->{sized:.6f}")

    paired_summary = {}
    for model_name in ("baseline", "size_conditioned"):
        paired_summary[model_name] = {}
        for size_name in ("small", "medium", "large"):
            group = [row for row in paired_shape_metrics
                     if row["checkpoint"] == model_name and row["size"] == size_name]
            paired_summary[model_name][size_name] = {
                metric_name: {
                    "mean": float(np.mean([row[metric_name] for row in group])) if group else 0.0,
                    "std": float(np.std([row[metric_name] for row in group])) if group else 0.0,
                }
                for metric_name in (
                    "centered_triangle_iou", "foreground_area_ratio", "boundary_f1_1px",
                    "raw_all_components_iou_unaligned", "raw_all_components_area_ratio",
                    "raw_centroid_error_pixels", "raw_component_count_8connected",
                    "raw_largest_component_fraction", "raw_disconnected_foreground_fraction",
                )
            }
    paired_generation = {
        "generation_seeds": generation_seeds,
        "num_seeds": len(generation_seeds),
        "num_samples_per_checkpoint": len(generation_seeds) * 3,
        "metrics_by_sample": paired_shape_metrics,
        "summary_by_checkpoint_and_size": paired_summary,
    }
    with open(os.path.join(args.output_dir, "paired_generation_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(paired_generation, handle, indent=2)

    elapsed_seconds = time.perf_counter() - evaluation_started
    metrics = {
        "heldout_v_mse_by_size": heldout_by_size,
        "heldout_v_mse_by_region": heldout_by_region,
        "heldout_sample_counts_by_size": selected_size_counts,
        "generated_shape_metrics": shape_metrics,
        "paired_generation_summary": paired_summary,
        "generation_seeds": generation_seeds,
    }
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    source_paths = [
        __file__,
        os.path.join(ROOT, "video_foundation", "models", "unet.py"),
        os.path.join(ROOT, "video_foundation", "diffusion", "process.py"),
        os.path.join(ROOT, "video_foundation", "diffusion", "sampler.py"),
        os.path.join(ROOT, "video_foundation", "inference", "shape_rendering.py"),
        os.path.join(ROOT, "video_foundation", "utils", "experiment_metadata.py"),
        os.path.join(ROOT, "video_foundation", "training", "train_image.py"),
    ]
    manifest = build_evaluation_manifest(
        task="raw_size_conditioned_shape_evaluation",
        args=vars(args),
        checkpoint_paths={
            "baseline": args.baseline_checkpoint,
            "size_conditioned": args.size_checkpoint,
        },
        checkpoint_payloads={
            "baseline": baseline_checkpoint,
            "size_conditioned": size_checkpoint,
        },
        data_dir=args.data_dir,
        source_paths=source_paths,
        elapsed_seconds=elapsed_seconds,
        torch_version=torch.__version__,
        cpu_threads=torch.get_num_threads(),
    )
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved evaluation metrics: {metrics_path}")
    print(f"Saved reproducibility manifest: {os.path.join(args.output_dir, 'run_manifest.json')}")
    print(f"Evaluation elapsed: {elapsed_seconds:.2f}s on CPU ({torch.get_num_threads()} threads)")


if __name__ == "__main__":
    main()
