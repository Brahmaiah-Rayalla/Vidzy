"""
Detailed DDPM sampling diagnostics to understand value explosion.
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
print("DETAILED DDPM SAMPLING DIAGNOSTICS")
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

# Run DDPM with detailed logging
print("\nRunning DDPM with detailed logging...")
batch_size = 1
x = torch.randn((batch_size, 3, 64, 64), device=device)
timesteps = torch.arange(999, -1, -10, device=device, dtype=torch.long)  # Every 10th step for speed

print(f"\nInitial noise: mean={x.mean():.4f}, std={x.std():.4f}")

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
    x_0_pred_clipped = torch.clamp(x_0_pred, -1.0, 1.0)

    # Get previous alpha_bar
    if t > 0:
        alpha_bar_prev = diffusion.schedule.alphas_cumprod[t - 1].to(device)
    else:
        alpha_bar_prev = torch.tensor(1.0).to(device)

    sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)

    # Compute posterior mean
    posterior_mean = (sqrt_alpha_bar_prev * beta / one_minus_alpha_bar) * x_0_pred_clipped + \
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

    if i % 10 == 0 or i == len(timesteps) - 1:
        print(f"\nStep {i} (t={t.item()}):")
        print(f"  x_t: mean={x.mean():.4f}, std={x.std():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")
        print(f"  predicted_noise: mean={predicted_noise.mean():.4f}, std={predicted_noise.std():.4f}")
        print(f"  x_0_pred (before clip): mean={x_0_pred.mean():.4f}, std={x_0_pred.std():.4f}, range=[{x_0_pred.min():.4f}, {x_0_pred.max():.4f}]")
        print(f"  x_0_pred (after clip): mean={x_0_pred_clipped.mean():.4f}, std={x_0_pred_clipped.std():.4f}")
        print(f"  posterior_mean: mean={posterior_mean.mean():.4f}, std={posterior_mean.std():.4f}")
        print(f"  posterior_variance: {posterior_variance.item():.6f}")
        print(f"  sqrt(posterior_variance): {torch.sqrt(posterior_variance).item():.6f}")

        # Check if x_0_pred is being clipped
        clipped_fraction = (x_0_pred.abs() > 1.0).float().mean().item()
        if clipped_fraction > 0.01:
            print(f"  WARNING: {clipped_fraction*100:.2f}% of x_0_pred values were clipped!")

print(f"\nFinal result: mean={x.mean():.4f}, std={x.std():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")

# Now test WITHOUT x_0 clipping in DDPM
print("\n" + "="*70)
print("DDPM WITHOUT X_0 CLIPPING")
print("="*70)

x = torch.randn((batch_size, 3, 64, 64), device=device)
print(f"\nInitial noise: mean={x.mean():.4f}, std={x.std():.4f}")

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

    # Predict x_0 WITHOUT clipping
    x_0_pred = (x - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar

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

    if i % 10 == 0 or i == len(timesteps) - 1:
        print(f"\nStep {i} (t={t.item()}):")
        print(f"  x_t: mean={x.mean():.4f}, std={x.std():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")
        print(f"  x_0_pred: mean={x_0_pred.mean():.4f}, std={x_0_pred.std():.4f}, range=[{x_0_pred.min():.4f}, {x_0_pred.max():.4f}]")

print(f"\nFinal result (no clipping): mean={x.mean():.4f}, std={x.std():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")
