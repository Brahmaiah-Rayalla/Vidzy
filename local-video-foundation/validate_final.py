"""
Final validation using DDPM sampling with the fix.
"""

import torch
from video_foundation.models.unet import UNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device
import os
from PIL import Image
import yaml

# Load config
with open('configs/image_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = get_device()
set_seed(42)

print("="*70)
print("FINAL VALIDATION WITH DDPM SAMPLING")
print("="*70)

# Load checkpoint
print("\nLoading checkpoint...")
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

# Generate samples with fast DDPM (every 10th step for speed)
print("\nGenerating 16 samples with fast DDPM (100 steps)...")
batch_size = 16
x = torch.randn((batch_size, 3, 64, 64), device=device)
timesteps = torch.arange(999, -1, -10, device=device, dtype=torch.long)

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

    # Predict x_0 (no clipping during sampling)
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

# Final clipping
samples = torch.clamp(x, -1.0, 1.0)

print(f"DDPM range: [{samples.min():.4f}, {samples.max():.4f}]")
print(f"DDPM mean: {samples.mean():.4f}")
print(f"DDPM std: {samples.std():.4f}")

# Save
os.makedirs("outputs/validation", exist_ok=True)
samples_save = (samples + 1) / 2.0
samples_save = torch.clamp(samples_save, 0, 1)
samples_save = (samples_save * 255).byte()
from torchvision.utils import make_grid
grid = make_grid(samples_save, nrow=4, padding=2)
grid = grid.permute(1, 2, 0).cpu().numpy()
grid = Image.fromarray(grid)
grid.save("outputs/validation/ddpm_16_samples_final.png")
print("Saved: outputs/validation/ddpm_16_samples_final.png")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
print(f"Training data std: 0.5508")
print(f"Generated std: {samples.std():.4f}")
print(f"Target std: ~0.55")
print(f"\nStd ratio: {samples.std() / 0.5508:.2%}")
print(f"Quality: {'GOOD' if samples.std() > 0.4 else 'POOR'}")
