"""Evaluate an existing L31 checkpoint against its migrated L27 baseline.

This is evaluation-only: it never constructs an optimizer or changes weights.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
L12 = REPO_ROOT / "experiments_stage3_2_9B" / "phase2L_training_repair" / "extended_250_steps" / "l12_gate_attribution"
L27 = REPO_ROOT / "experiments_stage3_2_9B" / "phase2L_training_repair" / "extended_250_steps" / "l27_structured_validation_text_mapping"
DATA = HERE / "data"
DEFAULT_CHECKPOINT = HERE / "ckpt_2500.pt"
DEFAULT_OUT = HERE / "eval_checkpoint_2500_seed42_n30"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
parser.add_argument("--eval_per_class", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=8,
                    help="CPU inference batch size; smaller values reduce peak memory")
args = parser.parse_args()
if args.eval_per_class < 1 or args.threads < 1 or args.batch_size < 1:
    parser.error("--eval_per_class, --threads, and --batch_size must be positive")

started = time.perf_counter()
torch.set_num_threads(args.threads)
sys.path.insert(0, str(L12))
sys.path.insert(0, str(REPO_ROOT))
from video_foundation.utils.experiment_metadata import build_evaluation_manifest
import l12_common
from l12_common import build_diffusion_schedule
from models.conditional_unet import ConditionalUNet

COLORS = [(220, 30, 30), (30, 80, 220), (230, 200, 30)]
COLOR_NAMES = ["red", "blue", "yellow"]
SHAPE_NAMES = ["circle", "triangle"]
ac, ts = build_diffusion_schedule()
betas = torch.linspace(0.0001, 0.02, 1000)
abars = torch.cumprod(1 - betas, 0)


def color_of(arr):
    bg = np.array([245, 245, 245]) / 255.0
    mask = np.abs(arr - bg).max(2) > 0.15
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return -1, -1
    pixels = arr[mask]
    distances = [np.abs(pixels - np.array(COLORS[k]) / 255.0).mean(1).mean() for k in range(3)]
    return int(np.argmin(distances)), 1


def radial_shape(arr):
    """Independent radial-boundary classifier calibrated on clean validation data."""
    bg = np.array([245, 245, 245]) / 255.0
    mask = np.abs(arr - bg).max(2) > 0.15
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
    radii = np.asarray(radii, dtype=float)
    if radii.mean() < 3:
        return -1
    k3 = np.abs(np.sum(radii * np.exp(-3j * angles))) / np.sum(radii)
    cv = radii.std() / radii.mean()
    score = k3 * 0.6 + min(1.0, cv / 0.4) * 0.4
    return 1 if score > 0.15 else 0


def classify(arr):
    color, _ = color_of(arr)
    shape = radial_shape(arr)
    joint = color * 2 + shape if color >= 0 and shape >= 0 else -1
    return joint, color, shape


class SplitConv9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_rgb = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.conv_cond = torch.nn.Conv2d(6, 64, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv_rgb(x[:, :3]) + self.conv_cond(x[:, 3:])


def make_model():
    model = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=1000,
        text_dim=256,
        n_heads=4,
        use_cross_attn=True,
    )
    model.conv_in = SplitConv9()
    return model


def install_l27_baseline(model):
    old_checkpoint = torch.load(L27 / "ckpt_10000.pt", map_location="cpu", weights_only=False)
    old = old_checkpoint["su"]
    with torch.no_grad():
        model.conv_in.conv_rgb.weight.copy_(old["conv_in.conv_rgb.weight"])
        model.conv_in.conv_rgb.bias.copy_(old["conv_in.conv_rgb.bias"])
        old_cond = old["conv_in.conv_cond.weight"]
        for class_id in range(6):
            color_id = class_id // 2
            shape_offset = 0.001 * ((class_id % 2) * 2 - 1)
            model.conv_in.conv_cond.weight[:, class_id].copy_(old_cond[:, color_id] + shape_offset)
    model.load_state_dict({key: value for key, value in old.items() if not key.startswith("conv_in.")}, strict=False)
    for name, parameter in model.named_parameters():
        if "cross_attn" in name:
            parameter.requires_grad_(False)
    return old_checkpoint


def forward(model, image, timestep, class_ids):
    batch = image.shape[0]
    condition = torch.zeros(batch, 6, 64, 64)
    for index, class_id in enumerate(class_ids):
        condition[index, class_id] = 1
    tokens = torch.zeros(batch, 64, 256)
    token_mask = torch.ones(batch, 64)
    return model(torch.cat([image, condition], 1), timestep, tokens, token_mask)


def val_loss(model):
    model.eval()
    losses = {class_id: [] for class_id in range(6)}
    generator = torch.Generator().manual_seed(args.seed + 999)
    with torch.no_grad():
        for start in range(0, len(validation_rows), args.batch_size):
            rows = validation_rows[start:start + args.batch_size]
            images = torch.stack([load_image(row) for row in rows])
            timesteps = torch.full((len(rows),), 500, dtype=torch.long)
            noise = torch.randn(len(rows), 3, 64, 64, generator=generator)
            alpha = abars[timesteps].view(-1, 1, 1, 1)
            noisy = alpha.sqrt() * images + (1 - alpha).sqrt() * noise
            predictions = forward(model, noisy, timesteps, [row["cid"] for row in rows])
            per_sample = (predictions - noise).square().flatten(1).mean(1).tolist()
            for row, loss in zip(rows, per_sample):
                losses[row["cid"]].append(float(loss))
    model.train()
    by_class = {str(k): float(np.mean(v)) for k, v in losses.items()}
    return float(np.mean([value for values in losses.values() for value in values])), by_class


def evaluate_generation(model, label):
    model.eval()
    stats = {
        "total": 0,
        "joint_correct": 0,
        "color_correct": 0,
        "shape_correct": 0,
        "per_class": {str(class_id): {"correct": 0, "total": 0} for class_id in range(6)},
    }
    sample_rows = []
    image_dir = args.output_dir / label
    image_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for class_id in range(6):
            class_samples = list(range(args.eval_per_class))
            for start in range(0, len(class_samples), args.batch_size):
                sample_ids = class_samples[start:start + args.batch_size]
                seeds = [50000 + args.seed * 1000000 + class_id * 1000 + sample_id for sample_id in sample_ids]
                image = torch.cat([
                    torch.randn(1, 3, 64, 64,
                                generator=torch.Generator().manual_seed(seed))
                    for seed in seeds
                ], dim=0)
                subsequence = ts[::2]
                for index in range(len(subsequence) - 1):
                    timestep, next_timestep = subsequence[index], subsequence[index + 1]
                    prediction = forward(
                        model,
                        image,
                        torch.full((len(sample_ids),), int(timestep), dtype=torch.long),
                        [class_id] * len(sample_ids),
                    )
                    alpha_t, alpha_next = ac[timestep], ac[next_timestep]
                    image = (
                        torch.sqrt(alpha_next)
                        * (image - torch.sqrt(1 - alpha_t) * prediction)
                        / torch.sqrt(alpha_t)
                        + torch.sqrt(1 - alpha_next) * prediction
                    )
                arrays = ((image.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()
                for sample_id, seed, array in zip(sample_ids, seeds, arrays):
                    image_path = image_dir / f"class_{class_id}_sample_{sample_id:03d}.png"
                    Image.fromarray((array * 255).astype("uint8")).save(image_path)
                    predicted_class, predicted_color, predicted_shape = classify(array)
                    joint_ok = predicted_class == class_id
                    color_ok = predicted_color == class_id // 2
                    shape_ok = predicted_shape == class_id % 2
                    stats["total"] += 1
                    stats["joint_correct"] += int(joint_ok)
                    stats["color_correct"] += int(color_ok)
                    stats["shape_correct"] += int(shape_ok)
                    stats["per_class"][str(class_id)]["total"] += 1
                    stats["per_class"][str(class_id)]["correct"] += int(joint_ok)
                    sample_rows.append({
                        "class_id": class_id,
                        "target_color": COLOR_NAMES[class_id // 2],
                        "target_shape": SHAPE_NAMES[class_id % 2],
                        "sample_id": sample_id,
                        "seed": seed,
                        "predicted_color_id": predicted_color,
                        "predicted_shape_id": predicted_shape,
                        "predicted_joint_class_id": predicted_class,
                        "color_correct": color_ok,
                        "shape_correct": shape_ok,
                        "joint_correct": joint_ok,
                        "image_path": str(image_path.resolve()),
                    })
            print(f"  {label}: class {class_id + 1}/6 evaluated", flush=True)
    total = max(1, stats["total"])
    stats["joint_accuracy"] = stats["joint_correct"] / total
    stats["color_accuracy"] = stats["color_correct"] / total
    stats["shape_accuracy"] = stats["shape_correct"] / total
    stats["per_class_accuracy"] = {
        class_id: value["correct"] / max(value["total"], 1)
        for class_id, value in stats["per_class"].items()
    }
    model.train()
    return stats, sample_rows


def save_grid(rows_by_label, name):
    tile, label_height, columns = 64, 20, 6
    rows = []
    for label, sample_rows in rows_by_label:
        canvas = Image.new("RGB", (columns * tile, 6 * (tile + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        for class_id in range(6):
            for col in range(columns):
                row_index = class_id * columns + col
                sample = sample_rows[row_index]
                image = Image.open(sample["image_path"]).convert("RGB")
                x, y = col * tile, class_id * (tile + label_height)
                canvas.paste(image, (x, y + label_height))
                draw.text((x + 2, y + 3), f"c{class_id} s{col}", fill="black")
        rows.append((label, canvas))
    output = Image.new("RGB", (columns * tile, len(rows) * 6 * (tile + label_height)), "white")
    draw = ImageDraw.Draw(output)
    for index, (label, canvas) in enumerate(rows):
        y = index * canvas.height
        output.paste(canvas, (0, y))
        draw.text((2, y + 2), label, fill="black")
    output.save(args.output_dir / name)


def load_image(row):
    array = np.asarray(Image.open(DATA / f"{row['file']}.png")).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1) * 2 - 1


args.output_dir.mkdir(parents=True, exist_ok=True)
print("Loading fixed L31 validation split and checkpoints...", flush=True)
dataset_rows = [json.loads(line) for line in (DATA / "manifest.jsonl").open(encoding="utf-8")]
validation_rows = [row for row in dataset_rows if row["file"].startswith("val_")]
if len(validation_rows) != 300:
    raise RuntimeError(f"Expected 300 fixed validation examples, found {len(validation_rows)}")

baseline_model = make_model()
baseline_checkpoint = install_l27_baseline(baseline_model)
candidate_model = make_model()
candidate_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
candidate_model.load_state_dict(candidate_checkpoint["su"], strict=True)
for name, parameter in candidate_model.named_parameters():
    if "cross_attn" in name:
        parameter.requires_grad_(False)

print("Calculating batched held-out diffusion losses...", flush=True)
baseline_v_loss, baseline_v_by_class = val_loss(baseline_model)
candidate_v_loss, candidate_v_by_class = val_loss(candidate_model)
print("Generating matched baseline samples...", flush=True)
baseline_stats, baseline_samples = evaluate_generation(baseline_model, "stage3_l27_migrated_baseline")
print("Generating step-2500 candidate samples...", flush=True)
candidate_stats, candidate_samples = evaluate_generation(candidate_model, "l31_step_2500")

metrics = {
    "task": "L31 high-salience shape/color pilot, matched Stage3 baseline evaluation",
    "evaluation_seed": args.seed,
    "samples_per_class": args.eval_per_class,
    "classes": [{"class_id": i, "color": COLOR_NAMES[i // 2], "shape": SHAPE_NAMES[i % 2]} for i in range(6)],
    "validation_v_noise_mse": {
        "stage3_l27_migrated_baseline": {"mean": baseline_v_loss, "per_class": baseline_v_by_class},
        "l31_step_2500": {"mean": candidate_v_loss, "per_class": candidate_v_by_class},
    },
    "generation": {
        "stage3_l27_migrated_baseline": baseline_stats,
        "l31_step_2500": candidate_stats,
    },
    "generation_elapsed_seconds": round(time.perf_counter() - started, 2),
    "training_checkpoint": str(args.checkpoint.resolve()),
    "training_checkpoint_step": candidate_checkpoint.get("step"),
    "cpu_threads": torch.get_num_threads(),
}
(args.output_dir / "L31_MATCHED_HELDOUT_EVALUATION.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(args.output_dir / "per_sample_metrics.json").write_text(
    json.dumps({"stage3_l27_migrated_baseline": baseline_samples, "l31_step_2500": candidate_samples}, indent=2),
    encoding="utf-8",
)
save_grid([("L27 migrated baseline", baseline_samples), ("L31 step 2500", candidate_samples)], "matched_generation_grid.png")

baseline_checkpoint_path = L27 / "ckpt_10000.pt"
baseline_manifest_payload = dict(baseline_checkpoint)
baseline_manifest_payload.setdefault("step", 10000)
baseline_manifest_payload["config"] = {
    "model": {"in_channels": 3, "model_channels": 64, "channel_mult": [1, 2, 2, 2], "num_res_blocks": 2, "num_condition_classes": 3, "image_size": 64},
    "conditioning": {"type": "onehot3_color", "classes": COLOR_NAMES},
    "training": {"steps": 10000, "checkpoint_step": 10000, "batch_size": 4, "learning_rate": 1e-4, "weight_decay": 0.01},
}
source_paths = [
    Path(__file__),
    HERE / "l31_full.py",
    L27 / "l27_full.py",
    L12 / "l12_common.py",
    REPO_ROOT / "experiments_stage3_2_7" / "video_foundation" / "models" / "conditional_unet.py",
    REPO_ROOT / "video_foundation" / "utils" / "experiment_metadata.py",
]
manifest = build_evaluation_manifest(
    task="l31_high_salience_shape_color_pilot_matched_eval",
    args={
        "checkpoint": str(args.checkpoint.resolve()),
        "baseline_checkpoint": str(baseline_checkpoint_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "eval_per_class": args.eval_per_class,
        "seed": args.seed,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "generation_mode": "class_conditioned_diffusion",
        "paired_generation_seed_formula": "50000 + seed*1000000 + class_id*1000 + sample_id",
    },
    checkpoint_paths={"stage3_l27_baseline": baseline_checkpoint_path, "l31_pilot": args.checkpoint},
    checkpoint_payloads={"stage3_l27_baseline": baseline_manifest_payload, "l31_pilot": candidate_checkpoint},
    data_dir=DATA,
    source_paths=source_paths,
    elapsed_seconds=metrics["generation_elapsed_seconds"],
    torch_version=torch.__version__,
    cpu_threads=torch.get_num_threads(),
)
(args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(json.dumps({
    "baseline": baseline_stats,
    "candidate": candidate_stats,
    "evaluation_seconds": metrics["generation_elapsed_seconds"],
    "output_dir": str(args.output_dir.resolve()),
}, indent=2))
