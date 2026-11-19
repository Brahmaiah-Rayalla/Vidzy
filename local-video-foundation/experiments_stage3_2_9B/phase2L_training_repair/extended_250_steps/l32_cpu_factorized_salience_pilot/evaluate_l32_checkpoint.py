"""Re-evaluate an L32 EMA checkpoint with a configurable denoising-step budget."""
import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
STAGE27_MODELS = REPO_ROOT / "experiments_stage3_2_7" / "video_foundation"
sys.path.insert(0, str(REPO_ROOT))
from video_foundation.utils.experiment_metadata import build_evaluation_manifest
sys.path.insert(0, str(STAGE27_MODELS))
from models.conditional_unet import ConditionalUNet

COLORS = [(220, 30, 30), (30, 80, 220), (230, 200, 30)]
COLOR_NAMES = ["red", "blue", "yellow"]
SHAPE_NAMES = ["circle", "triangle"]
BACKGROUND = (245, 245, 245)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=HERE / "checkpoint_step_01200.pt")
parser.add_argument("--output_dir", type=Path, default=HERE / "eval_48_steps")
parser.add_argument("--sampling_steps", type=int, default=48)
parser.add_argument("--eval_per_class", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()
if min(args.sampling_steps, args.eval_per_class, args.threads) < 1:
    parser.error("sampling steps, eval samples, and threads must be positive")

started = time.perf_counter()
torch.set_num_threads(args.threads)
args.output_dir.mkdir(parents=True, exist_ok=True)
data_dir = HERE / "data"
sample_dir = args.output_dir / "samples"
sample_dir.mkdir(parents=True, exist_ok=True)
betas = torch.linspace(0.0001, 0.02, 1000)
alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
timesteps = torch.linspace(999, 0, args.sampling_steps + 1).long()
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model_config = checkpoint["config"]["architecture"]
model = ConditionalUNet(
    in_channels=3, model_channels=model_config["base_channels"],
    channel_mult=model_config["channel_mult"], num_res_blocks=model_config["res_blocks"],
    attention_resolutions=[], dropout=0.0, num_timesteps=1000,
    text_dim=32, n_heads=4, use_cross_attn=False,
)
model.conv_in = torch.nn.Conv2d(model_config["input_channels"], model_config["base_channels"], 3, padding=1)
model.load_state_dict(checkpoint["ema_model"], strict=True)
model.eval()


def condition_maps(class_ids):
    result = torch.zeros(len(class_ids), 5, 64, 64)
    for index, class_id in enumerate(class_ids):
        result[index, class_id // 2].fill_(1.0)
        result[index, 3 + class_id % 2].fill_(1.0)
    return result


def forward(image, timestep, class_ids):
    context = torch.zeros(len(class_ids), 1, 32)
    mask = torch.ones(len(class_ids), 1)
    inputs = torch.cat((image, condition_maps(class_ids)), dim=1)
    return model(inputs, timestep, context, mask)


def largest_component(mask):
    seen = np.zeros_like(mask, dtype=bool)
    best = []
    height, width = mask.shape
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
    output = np.zeros_like(mask, dtype=bool)
    if best:
        ys, xs = zip(*best)
        output[np.asarray(ys), np.asarray(xs)] = True
    return output


def robust_classify(image):
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    blurred = np.asarray(Image.fromarray(image_u8).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    bg = np.asarray(BACKGROUND, dtype=np.float32) / 255.0
    color_prototypes = np.asarray(COLORS, dtype=np.float32) / 255.0
    d_bg = np.linalg.norm(blurred - bg, axis=2)
    d_colors = np.stack([np.linalg.norm(blurred - color, axis=2) for color in color_prototypes], axis=2)
    nearest = np.argmin(d_colors, axis=2)
    d_nearest = np.min(d_colors, axis=2)
    eligible = (d_nearest + 0.02 < d_bg) & (d_nearest < 0.48)
    candidates = []
    for color_id in range(3):
        component = largest_component(eligible & (nearest == color_id))
        candidates.append((int(component.sum()), color_id, component))
    area, color_id, component = max(candidates, key=lambda item: item[0])
    if area < 20:
        return -1, -1
    ys, xs = np.where(component)
    cy, cx = ys.mean(), xs.mean()
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    radii = []
    for angle in angles:
        best = 0
        dx, dy = np.cos(angle), np.sin(angle)
        for radius in range(2, 32):
            x, y = int(round(cx + dx * radius)), int(round(cy + dy * radius))
            if 0 <= x < 64 and 0 <= y < 64 and component[y, x]:
                best = radius
            elif best and radius > best + 2:
                break
        radii.append(best)
    radii = np.asarray(radii, dtype=np.float64)
    harmonic = np.abs(np.sum(radii * np.exp(-3j * angles))) / max(np.sum(radii), 1e-8)
    radial_cv = radii.std() / max(radii.mean(), 1e-8)
    shape_id = int(harmonic * 0.6 + min(1.0, radial_cv / 0.4) * 0.4 > 0.15)
    return color_id * 2 + shape_id, color_id


def paired_stats(base, candidate):
    rng = np.random.default_rng(8401)
    result = {}
    for key in ("joint_correct", "shape_correct", "color_correct"):
        a = np.asarray([row[key] for row in base], dtype=float)
        b = np.asarray([row[key] for row in candidate], dtype=float)
        delta = b - a
        indices = rng.integers(0, len(delta), size=(10000, len(delta)))
        means = delta[indices].mean(1)
        result[key] = {
            "baseline_accuracy": float(a.mean()),
            "candidate_accuracy": float(b.mean()),
            "paired_delta": float(delta.mean()),
            "improved_pairs": int(np.sum((b == 1) & (a == 0))),
            "regressed_pairs": int(np.sum((b == 0) & (a == 1))),
            "bootstrap_95pct_interval": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        }
    return result


def summarize(rows):
    count = max(1, len(rows))
    return {
        "samples": len(rows),
        "joint_accuracy": sum(row["joint_correct"] for row in rows) / count,
        "shape_accuracy": sum(row["shape_correct"] for row in rows) / count,
        "color_accuracy": sum(row["color_correct"] for row in rows) / count,
        "per_class_joint_accuracy": {
            str(cid): sum(row["joint_correct"] for row in rows if row["class_id"] == cid)
            / max(1, sum(row["class_id"] == cid for row in rows))
            for cid in range(6)
        },
    }


samples = []
with torch.no_grad():
    for class_id in range(6):
        for sample_id in range(args.eval_per_class):
            seed = 50000 + args.seed * 1000000 + class_id * 1000 + sample_id
            image = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(seed))
            for index in range(len(timesteps) - 1):
                t, t_prev = timesteps[index], timesteps[index + 1]
                eps = forward(image, torch.tensor([int(t)], dtype=torch.long), [class_id])
                alpha_t, alpha_prev = alphas_cumprod[t], alphas_cumprod[t_prev]
                predicted_x0 = (image - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
                predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
                image = alpha_prev.sqrt() * predicted_x0 + (1.0 - alpha_prev).sqrt() * eps
            eps_zero = forward(image, torch.zeros(1, dtype=torch.long), [class_id])
            alpha_zero = alphas_cumprod[0]
            image = ((image - (1.0 - alpha_zero).sqrt() * eps_zero) / alpha_zero.sqrt()).clamp(-1.0, 1.0)
            array = ((image.clamp(-1, 1) + 1) * 0.5)[0].permute(1, 2, 0).numpy()
            output_path = sample_dir / f"class_{class_id}_sample_{sample_id:03d}.png"
            Image.fromarray((array * 255).astype(np.uint8)).save(output_path)
            predicted_class, predicted_color = robust_classify(array)
            samples.append({
                "class_id": class_id,
                "sample_id": sample_id,
                "seed": seed,
                "predicted_class": predicted_class,
                "predicted_color": predicted_color,
                "joint_correct": predicted_class == class_id,
                "shape_correct": predicted_class >= 0 and predicted_class % 2 == class_id % 2,
                "color_correct": predicted_color == class_id // 2,
                "image_path": str(output_path.resolve()),
            })
        print(f"48-step generation class {class_id + 1}/6 complete", flush=True)

old_12_step = json.loads((HERE / "per_sample_metrics.json").read_text(encoding="utf-8"))
comparison = paired_stats(old_12_step, samples)
result = {
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_step": checkpoint["step"],
    "sampling_steps": args.sampling_steps,
    "seed": args.seed,
    "samples_per_class": args.eval_per_class,
    "generation": summarize(samples),
    "paired_vs_same_checkpoint_12_step_samples": comparison,
    "elapsed_generation_seconds": round(time.perf_counter() - started, 2),
    "cpu_threads": torch.get_num_threads(),
    "note": "Same EMA checkpoint and same noise seeds as the 12-step evaluation; only inference step count changes.",
    "sampler": "DDIM eta=0, x0 clipped to [-1, 1] at each step, followed by final t=0 x0 reconstruction",
}
(args.output_dir / "L32_SAMPLING_STEP_COMPARISON.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
(args.output_dir / "per_sample_metrics.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

tile, label_w, row_h = 192, 100, 220
grid = Image.new("RGB", (label_w + 6 * tile, 60 + 6 * row_h), "white")
draw = ImageDraw.Draw(grid)
draw.text((label_w, 8), f"Same L32 EMA checkpoint: 12 steps (left) vs {args.sampling_steps} steps (right)", fill="black")
for class_id in range(6):
    y = 60 + class_id * row_h
    draw.text((4, y + tile // 2 - 14), f"{COLOR_NAMES[class_id // 2]}\n{SHAPE_NAMES[class_id % 2]}", fill="black")
    for group, group_rows in enumerate((old_12_step, samples)):
        for column in range(3):
            row = next(item for item in group_rows if item["class_id"] == class_id and item["sample_id"] == column)
            path = row.get("image_path", row.get("image_path"))
            x = label_w + group * 3 * tile + column * tile
            image = Image.open(path).convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
            grid.paste(image, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(30, 140, 50) if row["joint_correct"] else (180, 50, 50), width=2)
grid.save(args.output_dir / "sampling_steps_grid.png")

manifest = build_evaluation_manifest(
    task="l32_sampling_step_ablation",
    args={"checkpoint": str(args.checkpoint.resolve()), "sampling_steps": args.sampling_steps,
          "eval_per_class": args.eval_per_class, "seed": args.seed, "threads": args.threads,
          "noise_seed_formula": "50000 + seed*1000000 + class_id*1000 + sample_id"},
    checkpoint_paths={"l32_ema": args.checkpoint},
    checkpoint_payloads={"l32_ema": checkpoint},
    data_dir=data_dir,
    source_paths=[Path(__file__), HERE / "train_l32_cpu_pilot.py",
                  STAGE27_MODELS / "models" / "conditional_unet.py",
                  REPO_ROOT / "video_foundation" / "utils" / "experiment_metadata.py"],
    elapsed_seconds=result["elapsed_generation_seconds"],
    torch_version=torch.__version__,
    cpu_threads=torch.get_num_threads(),
)
(args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
