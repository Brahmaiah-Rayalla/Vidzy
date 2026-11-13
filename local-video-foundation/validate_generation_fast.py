"""
Comprehensive validation of image generation quality (fast version).
"""

import torch
import numpy as np
from PIL import Image
from video_foundation.models.unet import UNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.checkpoint import CheckpointManager
from video_foundation.utils.reproducibility import set_seed, get_device
import os

print("="*60)
print("Stage 1.7 - Image Generation Validation (Fast)")
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

# Generate 64 images with DDIM (fast - 50 steps)
print("\n2. Generating 64 images with DDIM (with clipping)...")
set_seed(42)
with torch.no_grad():
    samples_ddim = diffusion.sample(
        model=model,
        batch_size=64,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        num_sampling_steps=50,
        device=device,
        clip_x0=True
    )

print(f"   DDIM raw range: [{samples_ddim.min():.4f}, {samples_ddim.max():.4f}]")
print(f"   DDIM mean: {samples_ddim.mean():.4f}")
print(f"   DDIM std: {samples_ddim.std():.4f}")

# Generate 16 images with DDPM (smaller batch for speed)
print("\n3. Generating 16 images with DDPM (reduced timesteps for speed)...")
set_seed(42)
with torch.no_grad():
    samples_ddpm = diffusion.sample_ddpm(
        model=model,
        batch_size=16,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        device=device
    )

print(f"   DDPM raw range: [{samples_ddpm.min():.4f}, {samples_ddpm.max():.4f}]")
print(f"   DDPM mean: {samples_ddpm.mean():.4f}")
print(f"   DDPM std: {samples_ddpm.std():.4f}")

# Save contact sheets
def save_contact_sheet(samples, path, nrow=8):
    """Save a contact sheet of samples."""
    samples = (samples + 1) / 2.0
    samples = torch.clamp(samples, 0, 1)
    samples = (samples * 255).byte()
    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=nrow, padding=2)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    grid = Image.fromarray(grid)
    grid.save(path)
    print(f"   Saved: {path}")

os.makedirs("outputs/validation", exist_ok=True)
save_contact_sheet(samples_ddim, "outputs/validation/ddim_64_samples.png", nrow=8)
save_contact_sheet(samples_ddpm, "outputs/validation/ddpm_16_samples.png", nrow=4)

# Detailed statistics
print("\n4. Detailed Statistics:")
print("\n   DDIM Statistics (64 samples):")
print(f"   Raw range: [{samples_ddim.min():.4f}, {samples_ddim.max():.4f}]")
print(f"   Mean: {samples_ddim.mean():.4f}")
print(f"   Std: {samples_ddim.std():.4f}")
print(f"   Median: {samples_ddim.median():.4f}")

# Per-image statistics
ddim_per_image_std = []
for i in range(64):
    img_std = samples_ddim[i].std().item()
    ddim_per_image_std.append(img_std)

print(f"   Per-image std - Mean: {np.mean(ddim_per_image_std):.4f}")
print(f"   Per-image std - Min: {np.min(ddim_per_image_std):.4f}")
print(f"   Per-image std - Max: {np.max(ddim_per_image_std):.4f}")

# Diversity metric (sample subset for speed)
ddim_diversity = []
for i in range(16):
    for j in range(i+1, 16):
        diff = (samples_ddim[i] - samples_ddim[j]).abs().mean().item()
        ddim_diversity.append(diff)

print(f"   Average pairwise difference (first 16): {np.mean(ddim_diversity):.4f}")

print("\n   DDPM Statistics (16 samples):")
print(f"   Raw range: [{samples_ddpm.min():.4f}, {samples_ddpm.max():.4f}]")
print(f"   Mean: {samples_ddpm.mean():.4f}")
print(f"   Std: {samples_ddpm.std():.4f}")
print(f"   Median: {samples_ddpm.median():.4f}")

# Per-image statistics
ddpm_per_image_std = []
for i in range(16):
    img_std = samples_ddpm[i].std().item()
    ddpm_per_image_std.append(img_std)

print(f"   Per-image std - Mean: {np.mean(ddpm_per_image_std):.4f}")
print(f"   Per-image std - Min: {np.min(ddpm_per_image_std):.4f}")
print(f"   Per-image std - Max: {np.max(ddpm_per_image_std):.4f}")

# Diversity metric
ddpm_diversity = []
for i in range(16):
    for j in range(i+1, 16):
        diff = (samples_ddpm[i] - samples_ddpm[j]).abs().mean().item()
        ddpm_diversity.append(diff)

print(f"   Average pairwise difference: {np.mean(ddpm_diversity):.4f}")

# Check for clipping
print("\n5. Clipping Analysis:")
ddim_clipped_low = (samples_ddim < -0.999).float().mean().item() * 100
ddim_clipped_high = (samples_ddim > 0.999).float().mean().item() * 100
print(f"   DDIM - Values at lower bound: {ddim_clipped_low:.2f}%")
print(f"   DDIM - Values at upper bound: {ddim_clipped_high:.2f}%")

ddpm_clipped_low = (samples_ddpm < -0.999).float().mean().item() * 100
ddpm_clipped_high = (samples_ddpm > 0.999).float().mean().item() * 100
print(f"   DDPM - Values at lower bound: {ddpm_clipped_low:.2f}%")
print(f"   DDPM - Values at upper bound: {ddpm_clipped_high:.2f}%")

# RGB conversion check
print("\n6. RGB Conversion Analysis:")
ddim_rgb = (samples_ddim + 1) / 2.0
ddim_rgb = torch.clamp(ddim_rgb, 0, 1)
print(f"   DDIM RGB range: [{ddim_rgb.min():.4f}, {ddim_rgb.max():.4f}]")
print(f"   DDIM RGB mean: {ddim_rgb.mean():.4f}")

ddpm_rgb = (samples_ddpm + 1) / 2.0
ddpm_rgb = torch.clamp(ddpm_rgb, 0, 1)
print(f"   DDPM RGB range: [{ddpm_rgb.min():.4f}, {ddim_rgb.max():.4f}]")
print(f"   DDPM RGB mean: {ddpm_rgb.mean():.4f}")

print("\n7. Contact sheets saved:")
print("   - outputs/validation/ddim_64_samples.png")
print("   - outputs/validation/ddpm_16_samples.png")
print("\n   Please visually inspect these grids to assess:")
print("   - Are recognizable geometric shapes visible?")
print("   - Do samples show diversity or collapse?")
print("   - Are there severe artifacts or noise patterns?")
