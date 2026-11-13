"""
Detailed sampling diagnostics to identify where values diverge.
"""

import torch
import numpy as np
from video_foundation.models.unet import UNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.checkpoint import CheckpointManager
from video_foundation.utils.reproducibility import set_seed, get_device

print("="*60)
print("Sampling Diagnostics")
print("="*60)

device = get_device()

# Load checkpoint
print("\n1. Loading checkpoint...")
checkpoint_path = "checkpoints/checkpoint_final.pt"
checkpoint_data = torch.load(checkpoint_path, map_location=device)
config = checkpoint_data['config']

# Reconstruct model
model = UNet(
    in_channels=config['model']['in_channels'],
    model_channels=config['model']['model_channels'],
    channel_mult=config['model']['channel_mult'],
    num_res_blocks=config['model']['num_res_blocks'],
    attention_resolutions=config['model']['attention_resolutions'],
    dropout=config['model']['dropout'],
    num_timesteps=config['diffusion']['num_timesteps']
)
model.load_state_dict(checkpoint_data['model_state_dict'])
model = model.to(device)
model.eval()

# Create diffusion sampler
diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)

set_seed(42)

# Custom sampling with diagnostics
batch_size = 2
channels = 3
image_size = 64
num_sampling_steps = 50

print(f"\n2. Running diagnostic sampling...")
print(f"   Batch size: {batch_size}")
print(f"   Sampling steps: {num_sampling_steps}")

# Start from noise
x = torch.randn(batch_size, channels, image_size, image_size, device=device)
print(f"   Initial noise range: [{x.min():.4f}, {x.max():.4f}]")
print(f"   Initial noise mean: {x.mean():.4f}")
print(f"   Initial noise std: {x.std():.4f}")

# Create timesteps
total_timesteps = diffusion.schedule.num_timesteps
timesteps = torch.linspace(total_timesteps - 1, 0, num_sampling_steps, dtype=torch.long, device=device)

print(f"\n3. Timestep sequence:")
print(f"   First timestep: {timesteps[0]}")
print(f"   Last timestep: {timesteps[-1]}")
print(f"   Step range: {timesteps[0] - timesteps[-1]}")

# Sampling loop with diagnostics
diagnostics = []

for i, t in enumerate(timesteps):
    t_batch = t.expand(batch_size)
    predicted_noise = model(x, t_batch)

    # Get schedule values
    _, alpha_bar_t, _ = diffusion.schedule.get(t_batch, device)

    if i < num_sampling_steps - 1:
        t_prev = timesteps[i + 1]
        t_prev_batch = t_prev.expand(batch_size)
        _, alpha_bar_t_prev, _ = diffusion.schedule.get(t_prev_batch, device)
    else:
        alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

    # Predict x_0
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    x_0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

    # Compute x_{t-1}
    sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
    sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)
    x = sqrt_alpha_bar_t_prev * x_0_pred + sqrt_one_minus_alpha_bar_t_prev * predicted_noise

    # Record diagnostics every 10 steps
    if i % 10 == 0 or i == num_sampling_steps - 1:
        diag = {
            'step': i,
            'timestep': t.item(),
            'x_min': x.min().item(),
            'x_max': x.max().item(),
            'x_mean': x.mean().item(),
            'x_std': x.std().item(),
            'x_0_pred_min': x_0_pred.min().item(),
            'x_0_pred_max': x_0_pred.max().item(),
            'x_0_pred_mean': x_0_pred.mean().item(),
            'x_0_pred_std': x_0_pred.std().item(),
            'alpha_bar_t': alpha_bar_t[0,0,0,0].item(),
            'alpha_bar_t_prev': alpha_bar_t_prev[0,0,0,0].item(),
            'has_nan': torch.isnan(x).any().item(),
            'has_inf': torch.isinf(x).any().item(),
        }
        diagnostics.append(diag)

print(f"\n4. Diagnostics Summary:")
print(f"{'Step':>6} {'T':>6} {'x_min':>10} {'x_max':>10} {'x_mean':>10} {'x_0_min':>10} {'x_0_max':>10} {'alpha_t':>10} {'alpha_prev':>10}")
print("-" * 100)

for d in diagnostics:
    print(f"{d['step']:>6} {d['timestep']:>6} {d['x_min']:>10.4f} {d['x_max']:>10.4f} {d['x_mean']:>10.4f} {d['x_0_pred_min']:>10.4f} {d['x_0_pred_max']:>10.4f} {d['alpha_bar_t']:>10.6f} {d['alpha_bar_t_prev']:>10.6f}")

print(f"\n5. Final Output:")
print(f"   Range: [{x.min():.4f}, {x.max():.4f}]")
print(f"   Mean: {x.mean():.4f}")
print(f"   Std: {x.std():.4f}")
print(f"   Has NaN: {torch.isnan(x).any().item()}")
print(f"   Has Inf: {torch.isinf(x).any().item()}")

print(f"\n6. Analysis:")
print(f"   Expected range: [-1.0, 1.0]")
print(f"   Actual range: [{x.min():.4f}, {x.max():.4f}]")

if x.min() < -1.5 or x.max() > 1.5:
    print(f"   WARNING: Values outside expected range detected!")
    print(f"   First divergence step:")
    for d in diagnostics:
        if d['x_min'] < -1.5 or d['x_max'] > 1.5:
            print(f"     Step {d['step']} (t={d['timestep']}): x=[{d['x_min']:.4f}, {d['x_max']:.4f}]")
            print(f"       x_0_pred=[{d['x_0_pred_min']:.4f}, {d['x_0_pred_max']:.4f}]")
            print(f"       alpha_bar_t={d['alpha_bar_t']:.6f}, alpha_bar_t_prev={d['alpha_bar_t_prev']:.6f}")
            break
else:
    print(f"   [OK] Values within expected range")
