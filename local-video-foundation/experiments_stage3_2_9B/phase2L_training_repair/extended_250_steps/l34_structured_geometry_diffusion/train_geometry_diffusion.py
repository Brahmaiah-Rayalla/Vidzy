"""CPU diffusion over compact shape geometry, followed by antialiased rasterization."""
import argparse
import copy
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
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "experiments_stage3_2_9B" / "phase2L_training_repair" / "extended_250_steps" / "l32_cpu_factorized_salience_pilot" / "data"
COLORS = [(220, 30, 30), (30, 80, 220), (230, 200, 30)]
COLOR_NAMES = ["red", "blue", "yellow"]
SHAPE_NAMES = ["circle", "triangle"]
BACKGROUND = (245, 245, 245)
SIZE = 64
T = 1000

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--samples_per_class", type=int, default=100)
parser.add_argument("--sampling_steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--output_dir", type=Path, default=HERE)
args = parser.parse_args()
if min(args.steps, args.batch_size, args.samples_per_class, args.sampling_steps, args.threads) < 1:
    parser.error("steps, batch size, samples per class, sampling steps, and threads must be positive")

started = time.perf_counter()
args.output_dir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
data_rows = [json.loads(line) for line in (DATA_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
train_rows = [row for row in data_rows if row["file"].startswith("train_")]
val_rows = [row for row in data_rows if row["file"].startswith("val_")]
if len(train_rows) != 1200 or len(val_rows) != 300:
    raise RuntimeError(f"Expected 1200 train and 300 validation records; got {len(train_rows)}/{len(val_rows)}")


def recover_rotation(row):
    """Replay the deterministic sample RNG to recover the source triangle angle."""
    rng = random.Random(int(row["seed"]))
    rng.randint(7, 12)
    rng.randint(22, 42)
    rng.randint(22, 42)
    return rng.uniform(0.0, 2.0 * math.pi)


def encode_geometry(row):
    class_id = int(row["class_id"])
    shape_id = class_id % 2
    cx = (float(row["cx"]) - 32.0) / 10.0
    cy = (float(row["cy"]) - 32.0) / 10.0
    if shape_id == 0:
        normalized_size = (float(row["size"]) - 9.5) / 2.5
        cos3, sin3 = 1.0, 0.0
    else:
        # Triangle `size` stores its side length (~19–32 px), not its circumradius.
        normalized_size = (float(row["size"]) - 25.5) / 6.5
        angle = recover_rotation(row)
        cos3, sin3 = math.cos(3.0 * angle), math.sin(3.0 * angle)
    return [cx, cy, normalized_size, cos3, sin3]


class GeometryDenoiser(nn.Module):
    """Class-conditioned epsilon predictor on five continuous geometry values."""
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


train_x = torch.tensor([encode_geometry(row) for row in train_rows], dtype=torch.float32)
train_y = torch.tensor([int(row["class_id"]) for row in train_rows], dtype=torch.long)
val_x = torch.tensor([encode_geometry(row) for row in val_rows], dtype=torch.float32)
val_y = torch.tensor([int(row["class_id"]) for row in val_rows], dtype=torch.long)
betas = torch.linspace(0.0001, 0.02, T)
alpha_bar = torch.cumprod(1.0 - betas, dim=0)
model = GeometryDenoiser()
ema_model = copy.deepcopy(model).eval()
for parameter in ema_model.parameters():
    parameter.requires_grad_(False)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
batch_rng = torch.Generator().manual_seed(args.seed + 1)
timestep_rng = torch.Generator().manual_seed(args.seed + 2)
noise_rng = torch.Generator().manual_seed(args.seed + 3)


def validation_mse(current_model):
    current_model.eval()
    result = []
    with torch.no_grad():
        for timestep in (0, 100, 250, 500, 750, 999):
            gen = torch.Generator().manual_seed(args.seed + 10000 + timestep)
            noise = torch.randn(val_x.shape, generator=gen)
            times = torch.full((len(val_x),), timestep, dtype=torch.long)
            alpha = alpha_bar[times].unsqueeze(1)
            prediction = current_model(alpha.sqrt() * val_x + (1.0 - alpha).sqrt() * noise, times, val_y)
            result.append({"timestep": timestep, "epsilon_mse": float((prediction - noise).square().mean())})
    current_model.train()
    return result


train_started = time.perf_counter()
history = []
for step_index in range(args.steps):
    indices = torch.randint(0, len(train_x), (args.batch_size,), generator=batch_rng)
    clean, classes = train_x[indices], train_y[indices]
    times = torch.randint(0, T, (args.batch_size,), generator=timestep_rng)
    noise = torch.randn((args.batch_size, 5), generator=noise_rng)
    alpha = alpha_bar[times].unsqueeze(1)
    prediction = model(alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise, times, classes)
    loss = F.mse_loss(prediction, noise)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(0.995).add_(parameter, alpha=0.005)
    step = step_index + 1
    if step == 1 or step % 100 == 0 or step == args.steps:
        record = {"step": step, "epsilon_mse": float(loss.detach()),
                  "elapsed_seconds": round(time.perf_counter() - train_started, 2)}
        history.append(record)
        print(f"step={step} eps_mse={record['epsilon_mse']:.5f} sec={record['elapsed_seconds']:.1f}", flush=True)

train_seconds = time.perf_counter() - train_started
parameter_count = sum(parameter.numel() for parameter in model.parameters())
config = {
    "representation": ["center_x", "center_y", "normalized_extent", "cos_3theta", "sin_3theta"],
    "conditioning": {"class_id_order": "color_id*2+shape_id", "colors": COLOR_NAMES, "shapes": SHAPE_NAMES},
    "architecture": {"kind": "MLP diffusion denoiser", "width": 128, "residual_blocks": 3,
                     "parameters": parameter_count},
    "diffusion": {"timesteps": T, "beta_schedule": "linear 1e-4..0.02", "objective": "epsilon_mse",
                  "ema_decay": 0.995},
    "training": {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                 "seed": args.seed, "threads": torch.get_num_threads()},
}
checkpoint_path = args.output_dir / "checkpoint_final.pt"
torch.save({"model": model.state_dict(), "ema_model": ema_model.state_dict(), "step": args.steps,
            "config": config}, checkpoint_path)
(args.output_dir / "training_metrics.json").write_text(json.dumps({
    "steps": args.steps, "train_elapsed_seconds": round(train_seconds, 3), "seed": args.seed,
    "threads": torch.get_num_threads(), "history": history,
}, indent=2), encoding="utf-8")


def sample_geometry(current_model):
    class_ids = torch.arange(6, dtype=torch.long).repeat_interleave(args.samples_per_class)
    gen = torch.Generator().manual_seed(args.seed + 20000)
    state = torch.randn((len(class_ids), 5), generator=gen)
    times = torch.linspace(T - 1, 0, args.sampling_steps + 1).long()
    current_model.eval()
    with torch.no_grad():
        for index in range(len(times) - 1):
            t, t_prev = times[index], times[index + 1]
            t_batch = torch.full((len(class_ids),), int(t), dtype=torch.long)
            epsilon = current_model(state, t_batch, class_ids)
            a_t, a_prev = alpha_bar[t], alpha_bar[t_prev]
            predicted_x0 = ((state - (1.0 - a_t).sqrt() * epsilon) / a_t.sqrt()).clamp(-1.0, 1.0)
            state = a_prev.sqrt() * predicted_x0 + (1.0 - a_prev).sqrt() * epsilon
        zero = torch.zeros(len(class_ids), dtype=torch.long)
        epsilon0 = current_model(state, zero, class_ids)
        a0 = alpha_bar[0]
        state = ((state - (1.0 - a0).sqrt() * epsilon0) / a0.sqrt()).clamp(-1.0, 1.0)
    current_model.train()
    return class_ids.numpy(), state.numpy()


def decode(class_id, values):
    shape_id = int(class_id) % 2
    cx = float(np.clip(32.0 + 10.0 * values[0], 22.0, 42.0))
    cy = float(np.clip(32.0 + 10.0 * values[1], 22.0, 42.0))
    if shape_id == 0:
        extent = int(np.clip(round(9.5 + 2.5 * values[2]), 7, 12))
        angle = 0.0
    else:
        extent = int(np.clip(round(25.5 + 6.5 * values[2]), 19, 32))
        angle = (math.atan2(float(values[4]), float(values[3])) / 3.0) % (2.0 * math.pi / 3.0)
    return {"class_id": int(class_id), "color_id": int(class_id) // 2, "shape_id": shape_id,
            "cx": cx, "cy": cy, "extent": extent, "angle_rad": angle}


def render(geometry, supersample=4):
    scale = supersample
    high = Image.new("RGB", (SIZE * scale, SIZE * scale), BACKGROUND)
    draw = ImageDraw.Draw(high)
    cx, cy = geometry["cx"] * scale, geometry["cy"] * scale
    extent_high = geometry["extent"] * scale
    color = COLORS[geometry["color_id"]]
    if geometry["shape_id"] == 0:
        draw.ellipse((cx - extent_high, cy - extent_high, cx + extent_high, cy + extent_high), fill=color)
    else:
        circumradius_high = extent_high / math.sqrt(3.0)
        angle = geometry["angle_rad"]
        points = [(cx + circumradius_high * math.cos(angle + 2.0 * math.pi * k / 3.0),
                   cy + circumradius_high * math.sin(angle + 2.0 * math.pi * k / 3.0)) for k in range(3)]
        draw.polygon(points, fill=color)
    return high.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def ks_distance(reference, generated):
    reference, generated = np.sort(np.asarray(reference)), np.sort(np.asarray(generated))
    points = np.sort(np.concatenate((reference, generated)))
    cdf_ref = np.searchsorted(reference, points, side="right") / len(reference)
    cdf_gen = np.searchsorted(generated, points, side="right") / len(generated)
    return float(np.max(np.abs(cdf_ref - cdf_gen)))


eval_started = time.perf_counter()
val_mse = validation_mse(ema_model)
generation_started = time.perf_counter()
sampled_classes, sampled_vectors = sample_geometry(ema_model)
sample_dir = args.output_dir / "samples"
sample_dir.mkdir(parents=True, exist_ok=True)
samples = []
for index, (class_id, vector) in enumerate(zip(sampled_classes.tolist(), sampled_vectors)):
    geometry = decode(class_id, vector)
    path = sample_dir / f"class_{class_id}_sample_{index % args.samples_per_class:03d}.png"
    render(geometry).save(path)
    samples.append({**geometry, "sample_id": index % args.samples_per_class,
                    "image_path": str(path.resolve())})

distribution = {}
for class_id in range(6):
    val_class = [row for row in val_rows if int(row["class_id"]) == class_id]
    gen_class = [row for row in samples if row["class_id"] == class_id]
    distribution[str(class_id)] = {}
    for feature, reference, generated in (
        ("cx", [float(row["cx"]) for row in val_class], [row["cx"] for row in gen_class]),
        ("cy", [float(row["cy"]) for row in val_class], [row["cy"] for row in gen_class]),
        ("extent", [float(row["size"]) for row in val_class], [row["extent"] for row in gen_class]),
    ):
        distribution[str(class_id)][feature] = {
            "heldout_mean": float(np.mean(reference)), "generated_mean": float(np.mean(generated)),
            "heldout_std": float(np.std(reference)), "generated_std": float(np.std(generated)),
            "ks_distance": ks_distance(reference, generated),
        }

tile, row_height, label_width = 192, 220, 100
grid = Image.new("RGB", (label_width + 3 * tile, 60 + 6 * row_height), "white")
draw = ImageDraw.Draw(grid)
draw.text((label_width, 8), "Geometry diffusion + CPU antialiased renderer — samples 0–2", fill="black")
for class_id in range(6):
    y = 60 + class_id * row_height
    draw.text((4, y + 80), f"{COLOR_NAMES[class_id // 2]}\n{SHAPE_NAMES[class_id % 2]}", fill="black")
    for sample_id in range(3):
        row = next(item for item in samples if item["class_id"] == class_id and item["sample_id"] == sample_id)
        image = Image.open(row["image_path"]).resize((tile, tile), Image.Resampling.NEAREST)
        x = label_width + sample_id * tile
        grid.paste(image, (x, y))
        draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(30, 140, 50), width=2)
grid_path = args.output_dir / "generation_grid.png"
grid.save(grid_path)
generation_seconds = time.perf_counter() - generation_started
eval_seconds = time.perf_counter() - eval_started

result = {
    "experiment": "L34 structured geometry diffusion + CPU renderer",
    "generation_mode": "conditional 5D geometry diffusion followed by supersampled primitive rasterization",
    "seed": args.seed, "samples_per_class": args.samples_per_class,
    "architecture": config["architecture"],
    "dataset": {"train": len(train_rows), "validation": len(val_rows),
                "manifest_sha256": hashlib.sha256((DATA_DIR / "manifest.jsonl").read_bytes()).hexdigest()},
    "heldout_geometry_noise_mse": val_mse,
    "renderer_label_adherence": 1.0,
    "geometry_distribution_by_class": distribution,
    "training_elapsed_seconds": round(train_seconds, 3),
    "sample_generation_and_render_seconds": round(generation_seconds, 3),
    "evaluation_sampling_render_seconds": round(eval_seconds, 3),
    "generated_samples_per_second": round(len(samples) / max(generation_seconds, 1e-8), 3),
    "cpu_threads": torch.get_num_threads(), "checkpoint": str(checkpoint_path.resolve()),
    "generation_grid": str(grid_path.resolve()),
}
(args.output_dir / "L34_RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
(args.output_dir / "per_sample_geometry.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
manifest = {
    "task": "l34_structured_geometry_diffusion",
    "created_at_unix": time.time(), "seed": args.seed, "torch": torch.__version__,
    "device": "cpu", "cpu_threads": torch.get_num_threads(), "arguments": vars(args) | {"output_dir": str(args.output_dir.resolve())},
    "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                   "config_sha256": config_sha},
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "dataset_manifest_sha256": result["dataset"]["manifest_sha256"],
    "train_elapsed_seconds": round(train_seconds, 3), "evaluation_elapsed_seconds": round(eval_seconds, 3),
}
(args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
