"""Train and evaluate a compact, factorized color/shape diffusion pilot on CPU."""
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
STAGE27_MODELS = REPO_ROOT / "experiments_stage3_2_7" / "video_foundation"
L27_DIR = REPO_ROOT / "experiments_stage3_2_9B" / "phase2L_training_repair" / "extended_250_steps" / "l27_structured_validation_text_mapping"
L31_DIR = REPO_ROOT / "experiments_stage3_2_9B" / "phase2L_training_repair" / "extended_250_steps" / "l31_high_salience_shape_conditioning"
sys.path.insert(0, str(REPO_ROOT))
from video_foundation.utils.experiment_metadata import build_evaluation_manifest
sys.path.insert(0, str(STAGE27_MODELS))
from models.conditional_unet import ConditionalUNet, ConditionalResidualBlock

COLORS = [(220, 30, 30), (30, 80, 220), (230, 200, 30)]
COLOR_NAMES = ["red", "blue", "yellow"]
SHAPE_NAMES = ["circle", "triangle"]
BACKGROUND = (245, 245, 245)
TIMESTEPS = 1000

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--eval_per_class", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--output_dir", type=Path, default=HERE)
parser.add_argument("--condition_film", action="store_true",
                    help="Inject factorized color/shape embeddings into every residual block through the time embedding")
parser.add_argument("--condition_adagn", action="store_true",
                    help="Apply zero-initialized factorized color/shape scale-shift after residual GroupNorm")
parser.add_argument("--shape_head", action="store_true",
                    help="Train an auxiliary silhouette head and report raw vs mask-projected samples separately")
parser.add_argument("--shape_head_weight", type=float, default=0.1)
parser.add_argument("--mask_projection_threshold", type=float, default=0.5,
                    help="Threshold for converting the learned silhouette logits to an output mask")
parser.add_argument("--mask_head_audit_only", action="store_true",
                    help="Evaluate a trained silhouette head on held-out noisy images without generating samples")
parser.add_argument("--warm_start_checkpoint", type=Path, default=None,
                    help="Optional compact pilot checkpoint whose EMA weights initialize this run")
parser.add_argument("--evaluation_only_checkpoint", type=Path, default=None,
                    help="Load this checkpoint and evaluate it without optimizer updates")
parser.add_argument("--baseline_metrics", type=Path, default=None,
                    help="Optional per-sample JSON used for a fixed-noise paired comparison")
parser.add_argument("--sampling_steps", type=int, default=12)
args = parser.parse_args()
if min(args.steps, args.batch_size, args.eval_per_class, args.threads, args.sampling_steps) < 1:
    parser.error("steps, batch size, samples per class, threads, and sampling steps must be positive")
if args.evaluation_only_checkpoint is not None:
    evaluation_payload = torch.load(args.evaluation_only_checkpoint, map_location="cpu", weights_only=False)
    args.condition_film = bool(evaluation_payload["config"]["architecture"].get("factorized_timestep_conditioning", False))
    args.condition_adagn = bool(evaluation_payload["config"]["architecture"].get("factorized_adagn_conditioning", False))
    args.shape_head = bool(evaluation_payload["config"]["architecture"].get("auxiliary_shape_head", False))
if args.mask_head_audit_only and args.evaluation_only_checkpoint is None:
    parser.error("--mask_head_audit_only requires --evaluation_only_checkpoint")

started = time.perf_counter()
args.output_dir.mkdir(parents=True, exist_ok=True)
data_dir = args.output_dir / "data"
image_dir = args.output_dir / "samples"
raw_image_dir = args.output_dir / "samples_raw"
data_dir.mkdir(parents=True, exist_ok=True)
image_dir.mkdir(parents=True, exist_ok=True)
raw_image_dir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
generator = torch.Generator().manual_seed(args.seed)
timestep_generator = torch.Generator().manual_seed(args.seed + 1)
betas = torch.linspace(0.0001, 0.02, TIMESTEPS)
alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
sampling_timesteps = torch.linspace(TIMESTEPS - 1, 0, args.sampling_steps + 1).long()


def make_data_row(split, class_id, sample_id, seed):
    rng = random.Random(seed)
    color_id, shape_id = class_id // 2, class_id % 2
    # Choose one latent radius and map triangle side to equal analytic area.
    latent_radius = rng.randint(7, 12)
    triangle_side = int(round(latent_radius * math.sqrt(4 * math.pi / math.sqrt(3))))
    cx, cy = rng.randint(22, 42), rng.randint(22, 42)
    scale = 4
    canvas = Image.new("RGB", (64 * scale, 64 * scale), BACKGROUND)
    mask = Image.new("L", (64 * scale, 64 * scale), 0)
    draw, mask_draw = ImageDraw.Draw(canvas), ImageDraw.Draw(mask)
    if shape_id == 0:
        radius = latent_radius * scale
        box = ((cx * scale - radius, cy * scale - radius), (cx * scale + radius, cy * scale + radius))
        draw.ellipse(box, fill=COLORS[color_id])
        mask_draw.ellipse(box, fill=255)
        extent = latent_radius
        analytic_area = math.pi * latent_radius**2
    else:
        circumradius = triangle_side / math.sqrt(3)
        rotation = rng.uniform(0, 2 * math.pi)
        points = [
            ((cx + circumradius * math.cos(rotation + 2 * math.pi * k / 3)) * scale,
             (cy + circumradius * math.sin(rotation + 2 * math.pi * k / 3)) * scale)
            for k in range(3)
        ]
        draw.polygon(points, fill=COLORS[color_id])
        mask_draw.polygon(points, fill=255)
        extent = triangle_side
        analytic_area = math.sqrt(3) * triangle_side**2 / 4
    canvas = canvas.resize((64, 64), Image.Resampling.LANCZOS)
    mask = mask.resize((64, 64), Image.Resampling.LANCZOS)
    stem = f"{split}_{class_id}_{sample_id:04d}"
    image_path, mask_path = data_dir / f"{stem}.png", data_dir / f"{stem}_mask.png"
    canvas.save(image_path)
    mask.save(mask_path)
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return {
        "file": stem,
        "class_id": class_id,
        "color_id": color_id,
        "shape_id": shape_id,
        "cx": cx,
        "cy": cy,
        "size": extent,
        "latent_radius": latent_radius,
        "analytic_area_px2": analytic_area,
        "seed": seed,
        "sha256": image_hash,
    }


manifest_path = data_dir / "manifest.jsonl"
if manifest_path.exists():
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
else:
    rows = []
    sequence = 0
    for split, per_class in (("train", 200), ("val", 50)):
        for class_id in range(6):
            for sample_id in range(per_class):
                rows.append(make_data_row(split, class_id, sample_id, 82000 + sequence))
                sequence += 1
    manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

train_rows = [row for row in rows if row["file"].startswith("train_")]
val_rows = [row for row in rows if row["file"].startswith("val_")]
if len(train_rows) != 1200 or len(val_rows) != 300:
    raise RuntimeError(f"Expected 1200 train/300 validation examples; found {len(train_rows)}/{len(val_rows)}")


def load_tensor(row):
    image = np.asarray(Image.open(data_dir / f"{row['file']}.png").convert("RGB"), dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(data_dir / f"{row['file']}_mask.png").convert("L"), dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image).permute(2, 0, 1) * 2.0 - 1.0
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)
    return image_tensor, mask_tensor


def condition_maps(class_ids):
    batch = len(class_ids)
    color_map = torch.zeros(batch, 3, 64, 64)
    shape_map = torch.zeros(batch, 2, 64, 64)
    for index, class_id in enumerate(class_ids):
        color_map[index, class_id // 2].fill_(1.0)
        shape_map[index, class_id % 2].fill_(1.0)
    return torch.cat((color_map, shape_map), dim=1)


def build_model():
    # A compact 2D U-Net: RGB+3-color+2-shape inputs, no unused cross-attention.
    model = ConditionalUNet(
        in_channels=3,
        model_channels=24,
        channel_mult=[1, 2, 3],
        num_res_blocks=1,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=TIMESTEPS,
        text_dim=32,
        n_heads=4,
        use_cross_attn=False,
    )
    model.conv_in = torch.nn.Conv2d(8, 24, 3, padding=1)
    warm = None
    if args.warm_start_checkpoint is not None:
        warm = torch.load(args.warm_start_checkpoint, map_location="cpu", weights_only=False)
        warm_arch = warm["config"].get("architecture", {})
        if warm_arch.get("factorized_timestep_conditioning", False):
            model.timestep_embed = ClassConditionedTimeEmbedding(model.timestep_embed, 24)
        if warm_arch.get("factorized_adagn_conditioning", False):
            install_adagn_conditioning(model)
        if warm_arch.get("auxiliary_shape_head", False):
            model.silhouette_head = torch.nn.Conv2d(24, 1, 3, padding=1)
        model.load_state_dict(warm["ema_model"], strict=True)
        print(f"warm-started from {args.warm_start_checkpoint} step={warm['step']}", flush=True)
    if args.condition_film and not isinstance(model.timestep_embed, ClassConditionedTimeEmbedding):
        model.timestep_embed = ClassConditionedTimeEmbedding(model.timestep_embed, 24)
    if args.condition_adagn and not any(isinstance(module, FactorizedAdaGNBlock) for module in model.modules()):
        install_adagn_conditioning(model)
    if args.shape_head and not hasattr(model, "silhouette_head"):
        model.silhouette_head = torch.nn.Conv2d(24, 1, 3, padding=1)
    return model


class ClassConditionedTimeEmbedding(torch.nn.Module):
    """Add factorized labels to every U-Net residual block's timestep features."""
    def __init__(self, timestep_embedding, dimension):
        super().__init__()
        self.timestep_embedding = timestep_embedding
        self.color_embedding = torch.nn.Embedding(3, dimension)
        self.shape_embedding = torch.nn.Embedding(2, dimension)
        torch.nn.init.zeros_(self.color_embedding.weight)
        torch.nn.init.zeros_(self.shape_embedding.weight)
        self._class_ids = None

    def set_class_ids(self, class_ids):
        self._class_ids = torch.as_tensor(class_ids, dtype=torch.long)

    def forward(self, timesteps):
        if self._class_ids is None or len(self._class_ids) != len(timesteps):
            raise RuntimeError("Factorized shape/color labels must be set before the U-Net forward pass")
        labels = self._class_ids.to(timesteps.device)
        return (self.timestep_embedding(timesteps)
                + self.color_embedding(labels // 2)
                + self.shape_embedding(labels % 2))


class FactorizedAdaGNBlock(ConditionalResidualBlock):
    """Residual block with factorized class scale/shift applied after GroupNorm."""
    def __init__(self, base_block):
        torch.nn.Module.__init__(self)
        self.timestep_proj = base_block.timestep_proj
        self.conv1, self.norm1 = base_block.conv1, base_block.norm1
        self.conv2, self.norm2 = base_block.conv2, base_block.norm2
        self.skip_connection = base_block.skip_connection
        self.use_cross_attn = base_block.use_cross_attn
        if self.use_cross_attn:
            self.cross_attn = base_block.cross_attn
        self.dropout = base_block.dropout
        channels = self.conv2.out_channels
        self.color_affine1 = torch.nn.Embedding(3, channels * 2)
        self.shape_affine1 = torch.nn.Embedding(2, channels * 2)
        self.color_affine2 = torch.nn.Embedding(3, channels * 2)
        self.shape_affine2 = torch.nn.Embedding(2, channels * 2)
        for embedding in (self.color_affine1, self.shape_affine1,
                          self.color_affine2, self.shape_affine2):
            torch.nn.init.zeros_(embedding.weight)
        self._class_ids = None

    def set_class_ids(self, class_ids):
        self._class_ids = torch.as_tensor(class_ids, dtype=torch.long)

    def _affine(self, color_embedding, shape_embedding, device):
        if self._class_ids is None:
            raise RuntimeError("Factorized labels must be set before the U-Net forward pass")
        labels = self._class_ids.to(device)
        scale, shift = (color_embedding(labels // 2) + shape_embedding(labels % 2)).chunk(2, dim=1)
        return scale[:, :, None, None], shift[:, :, None, None]

    def forward(self, x, timestep_emb, text_emb, text_mask=None):
        h = self.norm1(self.conv1(x))
        scale, shift = self._affine(self.color_affine1, self.shape_affine1, x.device)
        h = F.silu(h * (1.0 + scale) + shift)
        h = h + self.timestep_proj(timestep_emb)[:, :, None, None]
        if self.use_cross_attn:
            h = self.cross_attn(h, text_emb, text_mask)
        h = self.norm2(self.conv2(h))
        scale, shift = self._affine(self.color_affine2, self.shape_affine2, x.device)
        h = h * (1.0 + scale) + shift
        h = self.dropout(h)
        h = F.silu(h)
        return h + self.skip_connection(x)


def install_adagn_conditioning(model):
    for path in ("down_blocks", "up_blocks"):
        for module_list in getattr(model, path):
            for index, module in enumerate(module_list):
                if isinstance(module, ConditionalResidualBlock) and not isinstance(module, FactorizedAdaGNBlock):
                    module_list[index] = FactorizedAdaGNBlock(module)
    if not isinstance(model.middle_block, FactorizedAdaGNBlock):
        model.middle_block = FactorizedAdaGNBlock(model.middle_block)


model = build_model()
ema_model = copy.deepcopy(model).eval()
for parameter in ema_model.parameters():
    parameter.requires_grad_(False)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
empty_context = torch.zeros(args.batch_size, 1, 32)
empty_mask = torch.ones(args.batch_size, 1)


def predict(current_model, noisy, timesteps, class_ids):
    cond = condition_maps(class_ids)[:len(class_ids)]
    context = empty_context[:len(class_ids)]
    mask = empty_mask[:len(class_ids)]
    for module in current_model.modules():
        if isinstance(module, (ClassConditionedTimeEmbedding, FactorizedAdaGNBlock)):
            module.set_class_ids(class_ids)
    inputs = torch.cat((noisy, cond), dim=1)
    if hasattr(current_model, "silhouette_head"):
        epsilon, features = current_model(inputs, timesteps, context, mask, return_features=True)
        return epsilon, current_model.silhouette_head(features["low_res"])
    return current_model(inputs, timesteps, context, mask)


def save_json_atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def save_checkpoint(step, train_seconds, history):
    config = {
        "architecture": {"base_channels": 24, "channel_mult": [1, 2, 3], "res_blocks": 1,
                         "input_channels": 8, "output_channels": 3, "cross_attention": False,
                         "factorized_timestep_conditioning": args.condition_film,
                         "factorized_adagn_conditioning": args.condition_adagn,
                         "auxiliary_shape_head": args.shape_head,
                         "trainable_parameters": parameter_count},
        "conditioning": {"factorized_color_shape": True, "color_classes": COLOR_NAMES, "shape_classes": SHAPE_NAMES},
        "loss": {"epsilon_foreground_weight": 5.0, "low_noise_reconstruction_weight": 0.05,
                 "low_noise_silhouette_weight": 0.03, "auxiliary_shape_head_weight": args.shape_head_weight,
                 "auxiliary_shape_head_components": "balanced BCE + 0.5 Dice + 0.25 3x3 boundary-band L1",
                 "low_noise_timestep_max": 250},
        "training": {"steps": args.steps, "step": step, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate, "weight_decay": 1e-4,
                     "seed": args.seed, "timestep_seed": args.seed + 1,
                     "threads": torch.get_num_threads(), "ema_decay": 0.995,
                     "warm_start_checkpoint": str(args.warm_start_checkpoint.resolve()) if args.warm_start_checkpoint else None},
        "data_manifest": str(manifest_path.resolve()),
    }
    checkpoint_path = args.output_dir / f"checkpoint_step_{step:05d}.pt"
    torch.save({"model": model.state_dict(), "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "config": config}, checkpoint_path)
    save_json_atomic(args.output_dir / "training_metrics.json", {
        "completed_steps": step,
        "elapsed_train_seconds": round(train_seconds, 3),
        "seed": args.seed,
        "threads": torch.get_num_threads(),
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
    })
    return checkpoint_path, config


def train():
    model.train()
    history = []
    train_started = time.perf_counter()
    order = list(range(len(train_rows)))
    random.Random(args.seed).shuffle(order)
    batch_count = math.ceil(len(order) / args.batch_size)
    save_steps = sorted(set((args.steps // 3, (2 * args.steps) // 3, args.steps)))
    latest_checkpoint, latest_config = None, None
    for step_index in range(args.steps):
        if step_index % batch_count == 0 and step_index > 0:
            random.Random(args.seed + step_index).shuffle(order)
        start = (step_index % batch_count) * args.batch_size
        selected = [train_rows[order[(start + offset) % len(order)]] for offset in range(args.batch_size)]
        loaded = [load_tensor(row) for row in selected]
        images = torch.stack([item[0] for item in loaded])
        foreground = torch.stack([item[1] for item in loaded])
        class_ids = [row["class_id"] for row in selected]
        timesteps = torch.randint(0, TIMESTEPS, (len(selected),), generator=timestep_generator)
        noise = torch.randn(len(selected), 3, 64, 64, generator=generator)
        alpha = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        noisy = alpha.sqrt() * images + (1.0 - alpha).sqrt() * noise
        prediction_result = predict(model, noisy, timesteps, class_ids)
        if isinstance(prediction_result, tuple):
            prediction, mask_prediction = prediction_result
        else:
            prediction, mask_prediction = prediction_result, None
        epsilon_error = (prediction - noise).square().mean(dim=1, keepdim=True)
        weights = 1.0 + 4.0 * foreground
        per_sample_epsilon = (epsilon_error * weights).flatten(1).sum(1) / weights.flatten(1).sum(1)
        epsilon_loss = per_sample_epsilon.mean()
        foreground_loss = (epsilon_error * foreground).sum() / foreground.sum().clamp_min(1.0)
        background_loss = (epsilon_error * (1.0 - foreground)).sum() / (1.0 - foreground).sum().clamp_min(1.0)

        low_noise = timesteps <= 250
        reconstruction_loss = epsilon_loss.new_zeros(())
        silhouette_loss = epsilon_loss.new_zeros(())
        shape_head_loss = epsilon_loss.new_zeros(())
        if low_noise.any():
            x0 = (noisy[low_noise] - (1.0 - alpha[low_noise]).sqrt() * prediction[low_noise]) / alpha[low_noise].sqrt()
            target = images[low_noise]
            mask = foreground[low_noise]
            recon_pixel = (x0 - target).abs().mean(dim=1, keepdim=True)
            recon_weight = 1.0 + 4.0 * mask
            reconstruction_loss = (recon_pixel * recon_weight).sum() / recon_weight.sum().clamp_min(1.0)
            background_rgb = x0.new_tensor([2.0 * channel / 255.0 - 1.0 for channel in BACKGROUND]).view(1, 3, 1, 1)
            foreground_logits = ((x0 - background_rgb).abs().amax(dim=1, keepdim=True) - 0.12) * 8.0
            silhouette_loss = F.binary_cross_entropy_with_logits(foreground_logits, mask)
            if mask_prediction is not None:
                direct_logits = mask_prediction[low_noise]
                direct_mask = mask
                pixel_bce = F.binary_cross_entropy_with_logits(direct_logits, direct_mask, reduction="none")
                pixel_weights = 1.0 + 4.0 * direct_mask
                balanced_bce = (pixel_bce * pixel_weights).sum() / pixel_weights.sum().clamp_min(1.0)
                probabilities = torch.sigmoid(direct_logits)
                intersection = (probabilities * direct_mask).sum(dim=(1, 2, 3))
                denominator = probabilities.sum(dim=(1, 2, 3)) + direct_mask.sum(dim=(1, 2, 3))
                dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
                predicted_boundary = (F.max_pool2d(probabilities, 3, stride=1, padding=1)
                                      - (1.0 - F.max_pool2d(1.0 - probabilities, 3, stride=1, padding=1)))
                binary_target = (direct_mask > 0.5).to(direct_mask.dtype)
                target_boundary = (F.max_pool2d(binary_target, 3, stride=1, padding=1)
                                   - (1.0 - F.max_pool2d(1.0 - binary_target, 3, stride=1, padding=1)))
                boundary_loss = (predicted_boundary - target_boundary).abs().mean()
                shape_head_loss = balanced_bce + 0.5 * dice_loss + 0.25 * boundary_loss

        loss = epsilon_loss + 0.05 * reconstruction_loss + 0.03 * silhouette_loss
        if mask_prediction is not None:
            loss = loss + args.shape_head_weight * shape_head_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                ema_parameter.mul_(0.995).add_(parameter, alpha=0.005)

        step = step_index + 1
        if step % 100 == 0 or step == 1:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "epsilon_loss": float(epsilon_loss.detach()),
                "foreground_epsilon_mse": float(foreground_loss.detach()),
                "background_epsilon_mse": float(background_loss.detach()),
                "low_noise_rgb_l1": float(reconstruction_loss.detach()),
                "low_noise_silhouette_bce": float(silhouette_loss.detach()),
                "auxiliary_shape_head_loss": float(shape_head_loss.detach()),
                "elapsed_seconds": round(time.perf_counter() - train_started, 2),
            }
            history.append(record)
            print(f"step={step} loss={record['loss']:.4f} fg={record['foreground_epsilon_mse']:.4f} "
                  f"bg={record['background_epsilon_mse']:.4f} sec={record['elapsed_seconds']:.1f}", flush=True)
        if step in save_steps:
            latest_checkpoint, latest_config = save_checkpoint(step, time.perf_counter() - train_started, history)
            print(f"checkpoint={latest_checkpoint.name} saved", flush=True)
    return latest_checkpoint, latest_config, history, time.perf_counter() - train_started


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
    result = np.zeros_like(mask, dtype=bool)
    if best:
        ys, xs = zip(*best)
        result[np.asarray(ys), np.asarray(xs)] = True
    return result


def robust_classify(image):
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    blurred = np.asarray(Image.fromarray(image_u8).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    bg = np.asarray(BACKGROUND, dtype=np.float32) / 255.0
    d_bg = np.linalg.norm(blurred - bg, axis=2)
    colors = np.asarray(COLORS, dtype=np.float32) / 255.0
    d_colors = np.stack([np.linalg.norm(blurred - color, axis=2) for color in colors], axis=2)
    nearest = np.argmin(d_colors, axis=2)
    d_nearest = np.min(d_colors, axis=2)
    valid = (d_nearest + 0.02 < d_bg) & (d_nearest < 0.48)
    components = [(int((comp := largest_component(valid & (nearest == cid))).sum()), cid, comp) for cid in range(3)]
    area, color_id, component = max(components, key=lambda item: item[0])
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
    cv = radii.std() / max(radii.mean(), 1e-8)
    shape_id = int(harmonic * 0.6 + min(1.0, cv / 0.4) * 0.4 > 0.15)
    return color_id * 2 + shape_id, color_id


def save_grid(sample_rows, path, title=None):
    tile, label_width, row_height = 192, 100, 220
    canvas = Image.new("RGB", (label_width + 6 * tile, 60 + 6 * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    grid_title = title or ("CPU diffusion + learned silhouette projection" if args.shape_head else "CPU diffusion samples")
    draw.text((label_width, 8), f"{grid_title} — samples 0–2", fill="black")
    for class_id in range(6):
        y = 60 + class_id * row_height
        draw.text((5, y + tile // 2 - 14), f"{COLOR_NAMES[class_id // 2]}\n{SHAPE_NAMES[class_id % 2]}", fill="black")
        for sample_id in range(3):
            row = next(item for item in sample_rows if item["class_id"] == class_id and item["sample_id"] == sample_id)
            image = Image.open(row["image_path"]).convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
            x = label_width + sample_id * tile
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(30, 140, 50) if row["joint_correct"] else (180, 50, 50), width=2)
    canvas.save(path)


def generate_and_measure(current_model):
    current_model.eval()
    samples = []
    with torch.no_grad():
        for class_id in range(6):
            for sample_id in range(args.eval_per_class):
                seed = 50000 + args.seed * 1000000 + class_id * 1000 + sample_id
                noise = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(seed))
                image = noise
                for index in range(len(sampling_timesteps) - 1):
                    timestep, timestep_prev = sampling_timesteps[index], sampling_timesteps[index + 1]
                    time_batch = torch.tensor([int(timestep)], dtype=torch.long)
                    prediction_result = predict(current_model, image, time_batch, [class_id])
                    prediction = prediction_result[0] if isinstance(prediction_result, tuple) else prediction_result
                    alpha_t, alpha_prev = alphas_cumprod[timestep], alphas_cumprod[timestep_prev]
                    predicted_x0 = (image - (1.0 - alpha_t).sqrt() * prediction) / alpha_t.sqrt()
                    predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
                    image = alpha_prev.sqrt() * predicted_x0 + (1.0 - alpha_prev).sqrt() * prediction
                # The last loop update reaches x_t at t=0. Recover the clean x0 estimate
                # explicitly; returning x_t leaves the final beta_0 noise in every sample.
                zero_timestep = torch.zeros(1, dtype=torch.long)
                zero_result = predict(current_model, image, zero_timestep, [class_id])
                if isinstance(zero_result, tuple):
                    zero_prediction, zero_mask_logits = zero_result
                else:
                    zero_prediction, zero_mask_logits = zero_result, None
                alpha_zero = alphas_cumprod[0]
                image = ((image - (1.0 - alpha_zero).sqrt() * zero_prediction) / alpha_zero.sqrt()).clamp(-1.0, 1.0)
                raw_array = ((image.clamp(-1, 1) + 1) * 0.5)[0].permute(1, 2, 0).numpy()
                raw_path = raw_image_dir / f"class_{class_id}_sample_{sample_id:03d}.png"
                Image.fromarray((raw_array * 255).astype(np.uint8)).save(raw_path)
                if zero_mask_logits is not None:
                    alpha_mask = (torch.sigmoid(zero_mask_logits) >= args.mask_projection_threshold).to(image.dtype)
                    color = torch.tensor(COLORS[class_id // 2], dtype=image.dtype).view(1, 3, 1, 1) / 255.0 * 2.0 - 1.0
                    background = image.new_tensor(BACKGROUND).view(1, 3, 1, 1) / 255.0 * 2.0 - 1.0
                    projected = alpha_mask * color + (1.0 - alpha_mask) * background
                    array = ((projected.clamp(-1, 1) + 1) * 0.5)[0].permute(1, 2, 0).numpy()
                else:
                    array = raw_array
                path = image_dir / f"class_{class_id}_sample_{sample_id:03d}.png"
                Image.fromarray((array * 255).astype(np.uint8)).save(path)
                predicted_class, predicted_color = robust_classify(array)
                raw_predicted_class, raw_predicted_color = robust_classify(raw_array)
                samples.append({
                    "class_id": class_id,
                    "sample_id": sample_id,
                    "seed": seed,
                    "predicted_class": predicted_class,
                    "predicted_color": predicted_color,
                    "joint_correct": predicted_class == class_id,
                    "shape_correct": predicted_class >= 0 and predicted_class % 2 == class_id % 2,
                    "color_correct": predicted_color == class_id // 2,
                    "image_path": str(path.resolve()),
                    "generation_mode": "learned_silhouette_projection" if zero_mask_logits is not None else "raw_diffusion",
                    "raw_predicted_class": raw_predicted_class,
                    "raw_predicted_color": raw_predicted_color,
                    "raw_joint_correct": raw_predicted_class == class_id,
                    "raw_shape_correct": raw_predicted_class >= 0 and raw_predicted_class % 2 == class_id % 2,
                    "raw_color_correct": raw_predicted_color == class_id // 2,
                    "raw_image_path": str(raw_path.resolve()),
                })
            print(f"generation class {class_id + 1}/6 complete", flush=True)
    current_model.train()
    return samples


def summarize(samples):
    n = max(1, len(samples))
    return {
        "samples": len(samples),
        "joint_accuracy": sum(row["joint_correct"] for row in samples) / n,
        "shape_accuracy": sum(row["shape_correct"] for row in samples) / n,
        "color_accuracy": sum(row["color_correct"] for row in samples) / n,
        "per_class_joint_accuracy": {
            str(cid): sum(row["joint_correct"] for row in samples if row["class_id"] == cid)
            / max(1, sum(row["class_id"] == cid for row in samples))
            for cid in range(6)
        },
    }


def paired_stats(baseline, candidate):
    rng = np.random.default_rng(9137)
    result = {}
    for key in ("joint_correct", "shape_correct", "color_correct"):
        old = np.asarray([row[key] for row in baseline], dtype=np.float64)
        new = np.asarray([row[key] for row in candidate], dtype=np.float64)
        delta = new - old
        indices = rng.integers(0, len(delta), size=(10000, len(delta)))
        means = delta[indices].mean(axis=1)
        result[key] = {
            "baseline_accuracy": float(old.mean()),
            "candidate_accuracy": float(new.mean()),
            "paired_delta": float(delta.mean()),
            "improved_pairs": int(np.sum((new == 1) & (old == 0))),
            "regressed_pairs": int(np.sum((new == 0) & (old == 1))),
            "bootstrap_95pct_interval": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        }
    return result


def val_noise_mse(current_model):
    current_model.eval()
    mse = []
    val_gen = torch.Generator().manual_seed(args.seed + 9000)
    with torch.no_grad():
        for start in range(0, len(val_rows), args.batch_size):
            batch = val_rows[start:start + args.batch_size]
            images = torch.stack([load_tensor(row)[0] for row in batch])
            class_ids = [row["class_id"] for row in batch]
            timesteps = torch.full((len(batch),), 500, dtype=torch.long)
            noise = torch.randn(len(batch), 3, 64, 64, generator=val_gen)
            alpha = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            prediction = predict(current_model, alpha.sqrt() * images + (1 - alpha).sqrt() * noise, timesteps, class_ids)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            mse.extend((prediction - noise).square().flatten(1).mean(1).tolist())
    current_model.train()
    return float(np.mean(mse))


def audit_detector():
    correct = 0
    noise_correct = {"0.10": 0, "0.20": 0}
    rng = np.random.default_rng(997)
    for row in val_rows:
        arr = np.asarray(Image.open(data_dir / f"{row['file']}.png").convert("RGB"), dtype=np.float32) / 255.0
        clean_class, _ = robust_classify(arr)
        correct += int(clean_class == row["class_id"])
        for sigma in (0.10, 0.20):
            noisy = np.clip(arr + rng.normal(0.0, sigma, arr.shape), 0.0, 1.0)
            noisy_class, _ = robust_classify(noisy)
            noise_correct[f"{sigma:.2f}"] += int(noisy_class == row["class_id"])
    return {
        "clean_accuracy": correct / len(val_rows),
        "gaussian_noise_accuracy": {key: value / len(val_rows) for key, value in noise_correct.items()},
        "n": len(val_rows),
        "detector": "Gaussian blur 1.2 px, nearest palette-vs-background, largest connected component, radial 3-fold score",
    }


def audit_mask_head(current_model):
    if not hasattr(current_model, "silhouette_head"):
        raise RuntimeError("The selected checkpoint has no auxiliary silhouette head")
    current_model.eval()
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    with torch.no_grad():
        for timestep in (0, 100, 250):
            accum = {threshold: {"iou": [], "area_ratio": [], "boundary_f1": []} for threshold in thresholds}
            audit_gen = torch.Generator().manual_seed(args.seed + 18000 + timestep)
            for start in range(0, len(val_rows), args.batch_size):
                batch = val_rows[start:start + args.batch_size]
                loaded = [load_tensor(row) for row in batch]
                images = torch.stack([item[0] for item in loaded])
                masks = torch.stack([item[1] for item in loaded])
                class_ids = [row["class_id"] for row in batch]
                times = torch.full((len(batch),), timestep, dtype=torch.long)
                noise = torch.randn(len(batch), 3, 64, 64, generator=audit_gen)
                alpha = alphas_cumprod[times].view(-1, 1, 1, 1)
                _, logits = predict(current_model, alpha.sqrt() * images + (1.0 - alpha).sqrt() * noise, times, class_ids)
                probabilities = torch.sigmoid(logits)
                target = masks > 0.5
                target_boundary = (F.max_pool2d(target.float(), 3, stride=1, padding=1)
                                   - (1.0 - F.max_pool2d((~target).float(), 3, stride=1, padding=1))) > 0
                target_boundary_dilated = F.max_pool2d(target_boundary.float(), 3, stride=1, padding=1) > 0
                for threshold in thresholds:
                    predicted = probabilities >= threshold
                    intersection = (predicted & target).sum(dim=(1, 2, 3)).float()
                    union = (predicted | target).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
                    area = predicted.sum(dim=(1, 2, 3)).float()
                    target_area = target.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
                    pred_boundary = (F.max_pool2d(predicted.float(), 3, stride=1, padding=1)
                                     - (1.0 - F.max_pool2d((~predicted).float(), 3, stride=1, padding=1))) > 0
                    pred_boundary_dilated = F.max_pool2d(pred_boundary.float(), 3, stride=1, padding=1) > 0
                    precision = (pred_boundary & target_boundary_dilated).sum(dim=(1, 2, 3)).float() / pred_boundary.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
                    recall = (target_boundary & pred_boundary_dilated).sum(dim=(1, 2, 3)).float() / target_boundary.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
                    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
                    accum[threshold]["iou"].extend((intersection / union).tolist())
                    accum[threshold]["area_ratio"].extend((area / target_area).tolist())
                    accum[threshold]["boundary_f1"].extend(f1.tolist())
            results[str(timestep)] = {
                str(threshold): {key: float(np.mean(values)) for key, values in metrics.items()}
                for threshold, metrics in accum.items()
            }
    current_model.train()
    return {
        "checkpoint": str(args.evaluation_only_checkpoint.resolve()),
        "validation_samples": len(val_rows),
        "timesteps": [0, 100, 250],
        "thresholds": thresholds,
        "metrics_by_timestep_and_threshold": results,
        "note": "Thresholds are calibrated on held-out synthetic masks; generated-sample projection is a separate distribution and is not included here.",
    }


def main():
    print(f"dataset train={len(train_rows)} val={len(val_rows)} params={parameter_count} threads={args.threads}", flush=True)
    if args.evaluation_only_checkpoint is not None:
        checkpoint_path = args.evaluation_only_checkpoint.resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        config, history, train_seconds = checkpoint["config"], [], 0.0
        initial_validation = None if args.mask_head_audit_only else val_noise_mse(ema_model)
    else:
        initial_validation = val_noise_mse(ema_model)
        print(f"initial compact model val_noise_mse={initial_validation:.6f}", flush=True)
        checkpoint_path, config, history, train_seconds = train()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.mask_head_audit_only:
        report = audit_mask_head(ema_model)
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        audit_path = args.output_dir / "MASK_HEAD_VALIDATION_AUDIT.json"
        save_json_atomic(audit_path, report)
        print(json.dumps(report, indent=2), flush=True)
        return
    ema_model.load_state_dict(checkpoint["ema_model"])
    final_validation = val_noise_mse(ema_model)
    eval_started = time.perf_counter()
    generated_samples = generate_and_measure(ema_model)
    eval_seconds = time.perf_counter() - eval_started
    detector_audit = audit_detector()
    paired_comparison = None
    baseline_metrics = args.baseline_metrics
    if baseline_metrics is None and args.warm_start_checkpoint is not None:
        baseline_metrics = args.warm_start_checkpoint.resolve().parent / "per_sample_metrics.json"
    if baseline_metrics is not None:
        prior_metrics = baseline_metrics.resolve()
        if prior_metrics.exists():
            baseline_samples = json.loads(prior_metrics.read_text(encoding="utf-8"))
            raw_candidates = [{
                "class_id": row["class_id"], "sample_id": row["sample_id"], "seed": row["seed"],
                "joint_correct": row["raw_joint_correct"], "shape_correct": row["raw_shape_correct"],
                "color_correct": row["raw_color_correct"],
            } for row in generated_samples]
            paired_comparison = {
                "raw_diffusion_vs_baseline": paired_stats(baseline_samples, raw_candidates),
                "reported_output_vs_baseline": paired_stats(baseline_samples, generated_samples),
                "reported_output_mode": "learned_silhouette_projection" if args.shape_head else "raw_diffusion",
            }
    npz_path = args.output_dir / "per_sample_metrics.json"
    npz_path.write_text(json.dumps(generated_samples, indent=2), encoding="utf-8")
    save_grid(generated_samples, args.output_dir / "generation_grid.png")
    raw_samples = [{
        "class_id": row["class_id"], "sample_id": row["sample_id"],
        "joint_correct": row["raw_joint_correct"], "shape_correct": row["raw_shape_correct"],
        "color_correct": row["raw_color_correct"], "image_path": row["raw_image_path"],
    } for row in generated_samples]
    save_grid(raw_samples, args.output_dir / "generation_grid_raw.png", title="Raw epsilon-diffusion samples")
    result = {
        "experiment": "CPU shape/color diffusion with optional auxiliary silhouette head",
        "evaluation_only": args.evaluation_only_checkpoint is not None,
        "seed": args.seed,
        "samples_per_class": args.eval_per_class,
        "dataset": {"train": len(train_rows), "validation": len(val_rows),
                    "analytic_area_mean_by_shape": {
                        shape: float(np.mean([row["analytic_area_px2"] for row in train_rows if row["shape_id"] == i]))
                        for i, shape in enumerate(SHAPE_NAMES)
                    }},
        "architecture": config["architecture"],
        "loss_config": config["loss"],
        "validation_noise_mse": {"initial": initial_validation, "final_ema": final_validation},
        "generation": summarize(generated_samples),
        "raw_generation": summarize(raw_samples),
        "generation_modes": {"raw": "epsilon diffusion with clipped DDIM and final x0 reconstruction",
                             "reported_output": (f"thresholded learned silhouette projection at {args.mask_projection_threshold:.3f}"
                                                 if args.shape_head else "raw diffusion")},
        "paired_vs_baseline": paired_comparison,
        "detector_calibration": detector_audit,
        "train_elapsed_seconds": round(train_seconds, 2),
        "evaluation_elapsed_seconds": round(eval_seconds, 2),
        "total_elapsed_seconds": round(time.perf_counter() - started, 2),
        "cpu_threads": torch.get_num_threads(),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    save_json_atomic(args.output_dir / "L32_RESULTS.json", result)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    l27_path = L27_DIR / "ckpt_10000.pt"
    l27_payload = torch.load(l27_path, map_location="cpu", weights_only=False)
    l27_payload = dict(l27_payload)
    l27_payload.setdefault("step", 10000)
    l27_payload["config"] = {"model": "legacy 64-channel U-Net", "conditioning": "color one-hot migrated to six classes"}
    source_files = [Path(__file__), STAGE27_MODELS / "models" / "conditional_unet.py",
                    REPO_ROOT / "video_foundation" / "utils" / "experiment_metadata.py"]
    checkpoints = {"l32_pilot": checkpoint_path, "stage3_l27_reference": l27_path}
    payloads = {"l32_pilot": checkpoint_payload, "stage3_l27_reference": l27_payload}
    if args.warm_start_checkpoint is not None:
        warm_payload = torch.load(args.warm_start_checkpoint, map_location="cpu", weights_only=False)
        checkpoints["l32_warm_start"] = args.warm_start_checkpoint
        payloads["l32_warm_start"] = warm_payload
    manifest = build_evaluation_manifest(
        task="l32_cpu_factorized_salience_pilot",
        args={"steps": args.steps, "batch_size": args.batch_size, "eval_per_class": args.eval_per_class,
              "seed": args.seed, "threads": args.threads, "learning_rate": args.learning_rate,
              "generation_mode": "raw epsilon diffusion and optional learned silhouette projection; clipped x0 DDIM with final t=0 reconstruction",
              "sampling_steps": len(sampling_timesteps) - 1,
              "evaluation_only": args.evaluation_only_checkpoint is not None,
              "condition_film": args.condition_film,
              "condition_adagn": args.condition_adagn,
              "shape_head": args.shape_head,
              "shape_head_weight": args.shape_head_weight,
              "mask_projection_threshold": args.mask_projection_threshold},
        checkpoint_paths=checkpoints,
        checkpoint_payloads=payloads,
        data_dir=data_dir,
        source_paths=source_files,
        elapsed_seconds=eval_seconds,
        torch_version=torch.__version__,
        cpu_threads=torch.get_num_threads(),
    )
    save_json_atomic(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
