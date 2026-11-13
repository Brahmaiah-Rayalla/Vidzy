"""
Detailed sampling diagnostics to identify the reverse diffusion defect.
"""

import torch
import numpy as np
from video_foundation.models.unet import UNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device
import yaml

# Load config
with open('configs/image_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = get_device()
set_seed(42)

print("="*70)
print("DETAILED SAMPLING DIAGNOSTICS")
print("="*70)

# Load model
print("\nLoading model...")
checkpoint_data = torch.load('checkpoints/checkpoint_final.pt', map_location=device)
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

# Simulate the full reverse diffusion process with detailed logging
print("\nSimulating reverse diffusion with detailed logging...")
batch_size = 1
x = torch.randn((batch_size, 3, 64, 64), device=device)
total_timesteps = diffusion.schedule.num_timesteps
num_sampling_steps = 20
timesteps = torch.linspace(total_timesteps - 1, 0, num_sampling_steps, dtype=torch.long, device=device)

print(f"\nInitial noise: mean={x.mean():.4f}, std={x.std():.4f}")

for i, t in enumerate(timesteps):
    t_batch = t.expand(batch_size)

    with torch.no_grad():
        predicted_noise = model(x, t_batch)

    # Get schedule values
    _, alpha_bar_t, _ = diffusion.schedule.get(t_batch, device)

    if i < num_sampling_steps - 1:
        t_prev = timesteps[i + 1]
        t_prev_batch = t_prev.expand(batch_size)
        _, alpha_bar_t_prev, _ = diffusion.schedule.get(t_prev_batch, device)
    else:
        alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

    # Predict x_0
    x_0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
    x_0_pred_clipped = torch.clamp(x_0_pred, -1.0, 1.0)

    # DDIM update
    sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
    sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)

    x_new = sqrt_alpha_bar_t_prev * x_0_pred_clipped + sqrt_one_minus_alpha_bar_t_prev * predicted_noise

    print(f"\nStep {i} (t={t.item()}):")
    print(f"  x_t: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"  predicted_noise: mean={predicted_noise.mean():.4f}, std={predicted_noise.std():.4f}")
    print(f"  x_0_pred (before clip): mean={x_0_pred.mean():.4f}, std={x_0_pred.std():.4f}")
    print(f"  x_0_pred (after clip): mean={x_0_pred_clipped.mean():.4f}, std={x_0_pred_clipped.std():.4f}")
    print(f"  x_{t_prev.item()}: mean={x_new.mean():.4f}, std={x_new.std():.4f}")
    print(f"  alpha_bar_t: {alpha_bar_t.item():.6f}")
    print(f"  alpha_bar_t_prev: {alpha_bar_t_prev.item():.6f}")

    # Check if clipping is happening
    clipped_fraction = (x_0_pred.abs() > 1.0).float().mean().item()
    if clipped_fraction > 0.01:
        print(f"  WARNING: {clipped_fraction*100:.2f}% of x_0_pred values were clipped!")

    x = x_new

print(f"\nFinal result: mean={x.mean():.4f}, std={x.std():.4f}")

# Now test WITHOUT clipping to see if that's the issue
print("\n" + "="*70)
print("TESTING WITHOUT CLIPPING")
print("="*70)

x = torch.randn((batch_size, 3, 64, 64), device=device)
print(f"\nInitial noise: mean={x.mean():.4f}, std={x.std():.4f}")

for i, t in enumerate(timesteps):
    t_batch = t.expand(batch_size)

    with torch.no_grad():
        predicted_noise = model(x, t_batch)

    # Get schedule values
    _, alpha_bar_t, _ = diffusion.schedule.get(t_batch, device)

    if i < num_sampling_steps - 1:
        t_prev = timesteps[i + 1]
        t_prev_batch = t_prev.expand(batch_size)
        _, alpha_bar_t_prev, _ = diffusion.schedule.get(t_prev_batch, device)
    else:
        alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

    # Predict x_0 WITHOUT clipping
    x_0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

    # DDIM update
    sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
    sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)

    x = sqrt_alpha_bar_t_prev * x_0_pred + sqrt_one_minus_alpha_bar_t_prev * predicted_noise

    if i % 5 == 0 or i == num_sampling_steps - 1:
        print(f"\nStep {i} (t={t.item()}):")
        print(f"  x_t: mean={x.mean():.4f}, std={x.std():.4f}")
        print(f"  x_0_pred: mean={x_0_pred.mean():.4f}, std={x_0_pred.std():.4f}")
        print(f"  x_0_pred range: [{x_0_pred.min():.4f}, {x_0_pred.max():.4f}]")

print(f"\nFinal result (no clipping): mean={x.mean():.4f}, std={x.std():.4f}")
print(f"Final result range: [{x.min():.4f}, {x.max():.4f}]")
