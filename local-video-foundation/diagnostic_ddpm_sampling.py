"""
Test DDPM sampling vs DDIM to see if the sampling method is the issue.
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
print("DDPM vs DDIM SAMPLING COMPARISON")
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

# Test DDIM with 20 steps
print("\n" + "="*70)
print("DDIM Sampling (20 steps)")
print("="*70)
set_seed(42)
with torch.no_grad():
    samples_ddim = diffusion.sample(
        model=model,
        batch_size=4,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        num_sampling_steps=20,
        device=device,
        clip_x0=True
    )

print(f"DDIM samples: mean={samples_ddim.mean():.4f}, std={samples_ddim.std():.4f}")
print(f"DDIM range: [{samples_ddim.min():.4f}, {samples_ddim.max():.4f}]")

# Test DDIM with 50 steps
print("\n" + "="*70)
print("DDIM Sampling (50 steps)")
print("="*70)
set_seed(42)
with torch.no_grad():
    samples_ddim_50 = diffusion.sample(
        model=model,
        batch_size=4,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        num_sampling_steps=50,
        device=device,
        clip_x0=True
    )

print(f"DDIM (50 steps): mean={samples_ddim_50.mean():.4f}, std={samples_ddim_50.std():.4f}")
print(f"DDIM (50 steps) range: [{samples_ddim_50.min():.4f}, {samples_ddim_50.max():.4f}]")

# Test DDIM with 100 steps
print("\n" + "="*70)
print("DDIM Sampling (100 steps)")
print("="*70)
set_seed(42)
with torch.no_grad():
    samples_ddim_100 = diffusion.sample(
        model=model,
        batch_size=4,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        num_sampling_steps=100,
        device=device,
        clip_x0=True
    )

print(f"DDIM (100 steps): mean={samples_ddim_100.mean():.4f}, std={samples_ddim_100.std():.4f}")
print(f"DDIM (100 steps) range: [{samples_ddim_100.min():.4f}, {samples_ddim_100.max():.4f}]")

# Test DDPM with reduced steps for comparison
print("\n" + "="*70)
print("DDPM Sampling (first 100 timesteps only - for comparison)")
print("="*70)
print("Testing partial DDPM trajectory to compare with DDIM...")

# Manually run DDPM for first 100 steps
batch_size = 4
x = torch.randn((batch_size, 3, 64, 64), device=device)
timesteps = torch.arange(999, 899, -1, device=device, dtype=torch.long)

for i, t in enumerate(timesteps):
    t_batch = t.expand(batch_size)

    with torch.no_grad():
        predicted_noise = model(x, t_batch)

    # Get schedule values
    alpha, alpha_bar, one_minus_alpha_bar = diffusion.schedule.get(t_batch, device)
    beta = 1 - alpha

    sqrt_alpha = torch.sqrt(alpha)
    sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)

    # Predict x_0
    x_0_pred = (x - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
    x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

    # Get previous alpha_bar
    if t > 0:
        alpha_bar_prev = diffusion.schedule.alphas_cumprod[t - 1].to(device)
    else:
        alpha_bar_prev = torch.tensor(1.0).to(device)

    sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)

    # Compute posterior mean
    posterior_mean = (sqrt_alpha_bar_prev * beta / one_minus_alpha_bar) * x_0_pred + \
                   (sqrt_alpha * (1 - alpha_bar_prev) / one_minus_alpha_bar) * x

    # Compute posterior variance
    if t > 0:
        posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
    else:
        posterior_variance = torch.tensor(0.0).to(device)

    # Sample from posterior
    if t > 0:
        noise = torch.randn_like(x)
        x = posterior_mean + torch.sqrt(posterior_variance) * noise
    else:
        x = posterior_mean

print(f"DDPM (100 steps): mean={x.mean():.4f}, std={x.std():.4f}")
print(f"DDPM (100 steps) range: [{x.min():.4f}, {x.max():.4f}]")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"DDIM (20 steps): std={samples_ddim.std():.4f}")
print(f"DDIM (50 steps): std={samples_ddim_50.std():.4f}")
print(f"DDIM (100 steps): std={samples_ddim_100.std():.4f}")
print(f"DDPM (100 steps): std={x.std():.4f}")
print(f"\nTraining data std: 0.5508")
print(f"Target std: ~0.55")
