"""Audit L31 data geometry and score generated outputs with a noise-tolerant detector."""
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EVAL = HERE / "eval_checkpoint_2500_seed42_n30"
COLORS = np.asarray([(220, 30, 30), (30, 80, 220), (230, 200, 30)], dtype=np.float32) / 255.0
BACKGROUND = np.asarray((245, 245, 245), dtype=np.float32) / 255.0
SHAPES = ["circle", "triangle"]


def largest_component(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            queue = deque([(y, x)])
            seen[y, x] = True
            component = []
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
            if len(component) > len(best):
                best = component
    component_mask = np.zeros_like(mask, dtype=bool)
    if best:
        ys, xs = zip(*best)
        component_mask[np.asarray(ys), np.asarray(xs)] = True
    return component_mask


def radial_shape(mask):
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return -1
    cy, cx = ys.mean(), xs.mean()
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    radii = []
    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        best = 0
        for radius in range(2, 32):
            x, y = int(round(cx + dx * radius)), int(round(cy + dy * radius))
            if 0 <= x < 64 and 0 <= y < 64 and mask[y, x]:
                best = radius
            elif best > 0 and radius > best + 2:
                break
        radii.append(best)
    radii = np.asarray(radii, dtype=np.float64)
    if radii.mean() < 3:
        return -1
    harmonic3 = np.abs(np.sum(radii * np.exp(-3j * angles))) / np.sum(radii)
    radial_cv = radii.std() / radii.mean()
    score = harmonic3 * 0.6 + min(1.0, radial_cv / 0.4) * 0.4
    return int(score > 0.15)


def classify(image):
    """Blur high-frequency noise, segment palette-colored components, then classify contour."""
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    blurred = np.asarray(Image.fromarray(image_u8).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    distance_background = np.linalg.norm(blurred - BACKGROUND, axis=2)
    color_distances = np.stack([np.linalg.norm(blurred - color, axis=2) for color in COLORS], axis=2)
    nearest_color = np.argmin(color_distances, axis=2)
    nearest_distance = np.min(color_distances, axis=2)
    eligible = (nearest_distance + 0.02 < distance_background) & (nearest_distance < 0.48)
    candidates = []
    for color_id in range(3):
        component = largest_component(eligible & (nearest_color == color_id))
        candidates.append((int(component.sum()), color_id, component))
    area, color_id, component = max(candidates, key=lambda value: value[0])
    if area < 20:
        return -1, -1, int(area)
    shape_id = radial_shape(component)
    joint_id = color_id * 2 + shape_id if shape_id >= 0 else -1
    return joint_id, color_id, shape_id


def image_array(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def summarize(rows):
    total = max(1, len(rows))
    return {
        "samples": len(rows),
        "joint_accuracy": sum(row["joint_correct"] for row in rows) / total,
        "color_accuracy": sum(row["color_correct"] for row in rows) / total,
        "shape_accuracy": sum(row["shape_correct"] for row in rows) / total,
        "per_class_joint_accuracy": {
            str(cid): sum(row["joint_correct"] for row in rows if row["target_class"] == cid)
            / max(1, sum(row["target_class"] == cid for row in rows))
            for cid in range(6)
        },
        "confusion_matrix": [
            [sum(row["predicted_class"] == pred and row["target_class"] == target for row in rows) for pred in range(6)]
            for target in range(6)
        ],
    }


manifest_rows = [json.loads(line) for line in (DATA / "manifest.jsonl").open(encoding="utf-8")]
training_areas = {}
for shape_id, shape_name in enumerate(SHAPES):
    values = []
    for row in manifest_rows:
        if row["file"].startswith("train_") and row["shape"] == shape_id:
            size = row["size"]
            values.append(np.pi * size**2 if shape_id == 0 else np.sqrt(3.0) * size**2 / 4.0)
    training_areas[shape_name] = {"mean_analytic_area_px2": float(np.mean(values)), "n": len(values)}

clean_rows = [row for row in manifest_rows if row["file"].startswith("val_")]
clean_accuracy = {"total": len(clean_rows), "correct": 0}
for row in clean_rows:
    predicted, _, _ = classify(image_array(DATA / f"{row['file']}.png"))
    clean_accuracy["correct"] += int(predicted == row["cid"])
clean_accuracy["accuracy"] = clean_accuracy["correct"] / max(1, clean_accuracy["total"])

noise_calibration = {}
noise_rng = np.random.default_rng(1742)
for sigma in (0.05, 0.10, 0.15, 0.20):
    correct = 0
    for row in clean_rows:
        clean = image_array(DATA / f"{row['file']}.png")
        noisy = np.clip(clean + noise_rng.normal(0.0, sigma, clean.shape), 0.0, 1.0)
        predicted, _, _ = classify(noisy)
        correct += int(predicted == row["cid"])
    noise_calibration[str(sigma)] = {"accuracy": correct / len(clean_rows), "total": len(clean_rows)}

sample_metrics = json.loads((EVAL / "per_sample_metrics.json").read_text(encoding="utf-8"))
generated = {}
scored_by_run = {}
for run_name, samples in sample_metrics.items():
    scored = []
    for row in samples:
        predicted, predicted_color, predicted_shape = classify(image_array(row["image_path"]))
        target = row["class_id"]
        scored.append({
            "target_class": target,
            "predicted_class": predicted,
            "predicted_color": predicted_color,
            "predicted_shape": predicted_shape,
            "joint_correct": predicted == target,
            "color_correct": predicted_color == target // 2,
            "shape_correct": predicted_shape == target % 2,
            "image_path": row["image_path"],
        })
    generated[run_name] = summarize(scored)
    scored_by_run[run_name] = scored

baseline_rows = scored_by_run["stage3_l27_migrated_baseline"]
candidate_rows = scored_by_run["l31_step_2500"]
paired_rng = np.random.default_rng(4242)
paired_comparison = {}
for metric in ("joint_correct", "shape_correct", "color_correct"):
    base_values = np.asarray([row[metric] for row in baseline_rows], dtype=np.float64)
    candidate_values = np.asarray([row[metric] for row in candidate_rows], dtype=np.float64)
    improvements = int(np.sum((candidate_values == 1) & (base_values == 0)))
    regressions = int(np.sum((candidate_values == 0) & (base_values == 1)))
    paired_delta = candidate_values - base_values
    bootstrap_indices = paired_rng.integers(0, len(paired_delta), size=(10000, len(paired_delta)))
    bootstrap_means = paired_delta[bootstrap_indices].mean(axis=1)
    paired_comparison[metric] = {
        "baseline_accuracy": float(base_values.mean()),
        "candidate_accuracy": float(candidate_values.mean()),
        "paired_delta": float(paired_delta.mean()),
        "paired_improvements": improvements,
        "paired_regressions": regressions,
        "bootstrap_95pct_interval": [float(x) for x in np.quantile(bootstrap_means, [0.025, 0.975])],
    }
per_class_comparison = {}
for class_id in range(6):
    base_class = [row for row in baseline_rows if row["target_class"] == class_id]
    candidate_class = [row for row in candidate_rows if row["target_class"] == class_id]
    per_class_comparison[str(class_id)] = {
        "baseline_joint_accuracy": float(np.mean([row["joint_correct"] for row in base_class])),
        "candidate_joint_accuracy": float(np.mean([row["joint_correct"] for row in candidate_class])),
    }

result = {
    "detector": {
        "method": "RGB Gaussian blur 1.2px; nearest palette-vs-background mask; largest 8-connected component; radial 3-fold shape score",
        "clean_heldout_calibration": clean_accuracy,
        "additive_rgb_noise_calibration": noise_calibration,
    },
    "training_data_analytic_area": training_areas,
    "generated_outputs_reclassified": generated,
    "paired_comparison": paired_comparison,
    "per_class_joint_comparison": per_class_comparison,
    "limitations": "Synthetic additive-noise calibration is not identical to diffusion artifacts; read scores together with the saved image grid.",
}
output_path = EVAL / "failure_audit_robust_metrics.json"
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
