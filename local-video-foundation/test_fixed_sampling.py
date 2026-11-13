"""
Test the fixed DDIM sampling with x_0 clipping.
"""

import torch
from video_foundation.models.unet import UNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.checkpoint import CheckpointManager
from video_foundation.utils.reproducibility import set_seed, get_device
from PIL import Image
import numpy as np

print("="*60)
print("Testing Fixed DDIM Sampling")
print("="*60)

device = get_device()

# Load checkpoint
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

print("\n1. Testing DDIM with clipping (clip_x0=True)...")
set_seed(42)
with torch.no_grad():
    samples_ddim_clip = diffusion.sample(
        model=model,
        batch_size=4,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        num_sampling_steps=50,
        device=device,
        clip_x0=True
    )

print(f"   Range: [{samples_ddim_clip.min():.4f}, {samples_ddim_clip.max():.4f}]")
print(f"   Mean: {samples_ddim_clip.mean():.4f}")
print(f"   Std: {samples_ddim_clip.std():.4f}")
print(f"   Has NaN: {torch.isnan(samples_ddim_clip).any().item()}")
print(f"   Has Inf: {torch.isinf(samples_ddim_clip).any().item()}")

print("\n2. Testing DDIM without clipping (clip_x0=False)...")
set_seed(42)
with torch.no_grad():
    samples_ddim_no_clip = diffusion.sample(
        model=model,
        batch_size=4,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        num_sampling_steps=50,
        device=device,
        clip_x0=False
    )

print(f"   Range: [{samples_ddim_no_clip.min():.4f}, {samples_ddim_no_clip.max():.4f}]")
print(f"   Mean: {samples_ddim_no_clip.mean():.4f}")
print(f"   Std: {samples_ddim_no_clip.std():.4f}")
print(f"   Has NaN: {torch.isnan(samples_ddim_no_clip).any().item()}")
print(f"   Has Inf: {torch.isinf(samples_ddim_no_clip).any().item()}")

print("\n3. Testing DDPM sampling...")
set_seed(42)
with torch.no_grad():
    samples_ddpm = diffusion.sample_ddpm(
        model=model,
        batch_size=4,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        device=device
    )

print(f"   Range: [{samples_ddpm.min():.4f}, {samples_ddpm.max():.4f}]")
print(f"   Mean: {samples_ddpm.mean():.4f}")
print(f"   Std: {samples_ddpm.std():.4f}")
print(f"   Has NaN: {torch.isnan(samples_ddpm).any().item()}")
print(f"   Has Inf: {torch.isinf(samples_ddpm).any().item()}")

# Save comparison grids
def save_grid(samples, path, nrow=2):
    samples = (samples + 1) / 2.0
    samples = torch.clamp(samples, 0, 1)
    samples = (samples * 255).byte()
    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=nrow, padding=2)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    grid = Image.fromarray(grid)
    grid.save(path)

print("\n4. Saving comparison grids...")
save_grid(samples_ddim_clip, "outputs/ddim_with_clipping.png", nrow=2)
save_grid(samples_ddim_no_clip, "outputs/ddim_without_clipping.png", nrow=2)
save_grid(samples_ddpm, "outputs/ddpm_sampling.png", nrow=2)

print("   Saved:")
print("   - outputs/ddim_with_clipping.png")
print("   - outputs/ddim_without_clipping.png")
print("   - outputs/ddpm_sampling.png")

print("\n5. Summary:")
print(f"   DDIM with clipping: [{samples_ddim_clip.min():.4f}, {samples_ddim_clip.max():.4f}]")
print(f"   DDIM without clipping: [{samples_ddim_no_clip.min():.4f}, {samples_ddim_no_clip.max():.4f}]")
print(f"   DDPM: [{samples_ddpm.min():.4f}, {samples_ddpm.max():.4f}]")

if samples_ddim_clip.min() >= -1.1 and samples_ddim_clip.max() <= 1.1:
    print(f"\n   [OK] DDIM with clipping produces values in expected range")
else:
    print(f"\n   WARNING: DDIM with clipping still outside expected range")

if samples_ddpm.min() >= -1.1 and samples_ddpm.max() <= 1.1:
    print(f"   [OK] DDPM produces values in expected range")
else:
    print(f"   WARNING: DDPM still outside expected range")
