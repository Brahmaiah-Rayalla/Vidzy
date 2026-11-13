"""
Minimal validation - generate 4 samples quickly.
"""

import torch
import numpy as np
from PIL import Image
from video_foundation.models.unet import UNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device
import os

device = get_device()

# Load checkpoint
print("Loading checkpoint...")
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

# Generate 4 samples with DDIM (20 steps for speed)
print("Generating 4 samples with DDIM (20 steps)...")
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

print(f"DDIM range: [{samples_ddim.min():.4f}, {samples_ddim.max():.4f}]")
print(f"DDIM mean: {samples_ddim.mean():.4f}")
print(f"DDIM std: {samples_ddim.std():.4f}")

# Save
os.makedirs("outputs/validation", exist_ok=True)
samples = (samples_ddim + 1) / 2.0
samples = torch.clamp(samples, 0, 1)
samples = (samples * 255).byte()
from torchvision.utils import make_grid
grid = make_grid(samples, nrow=2, padding=2)
grid = grid.permute(1, 2, 0).cpu().numpy()
grid = Image.fromarray(grid)
grid.save("outputs/validation/ddim_4_samples.png")
print("Saved: outputs/validation/ddim_4_samples.png")
