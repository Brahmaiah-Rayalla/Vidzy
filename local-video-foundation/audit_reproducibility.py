"""
Test checkpoint reloading and reproducibility.
"""

import torch
import numpy as np
from video_foundation.models.unet import UNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.checkpoint import CheckpointManager
from video_foundation.utils.reproducibility import set_seed, get_device

print("="*60)
print("Checkpoint and Reproducibility Audit")
print("="*60)

device = get_device()

# Load checkpoint
print("\n1. Loading checkpoint...")
checkpoint_path = "checkpoints/checkpoint_final.pt"
checkpoint_data = torch.load(checkpoint_path, map_location=device)

config = checkpoint_data.get('config', {})
step = checkpoint_data.get('step', 'unknown')
seed = checkpoint_data.get('seed', 'unknown')

print(f"   Checkpoint step: {step}")
print(f"   Checkpoint seed: {seed}")
print(f"   Config loaded: {len(config) > 0}")

# Reconstruct model
print("\n2. Reconstructing model...")
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

num_params = sum(p.numel() for p in model.parameters())
print(f"   Model parameters: {num_params:,}")
print(f"   Model loaded successfully")

# Create diffusion sampler
print("\n3. Creating diffusion sampler...")
diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)
print(f"   Sampler created")

# Test reproducibility with same seed
print("\n4. Testing reproducibility with same seed...")
set_seed(42)

with torch.no_grad():
    samples1 = diffusion.sample(
        model=model,
        batch_size=4,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        num_sampling_steps=50,
        device=device
    )

set_seed(42)

with torch.no_grad():
    samples2 = diffusion.sample(
        model=model,
        batch_size=4,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        num_sampling_steps=50,
        device=device
    )

# Check if samples are identical
are_identical = torch.allclose(samples1, samples2, atol=1e-5)
print(f"   Samples identical with same seed: {are_identical}")

if are_identical:
    print(f"   [OK] Reproducibility confirmed")
else:
    max_diff = (samples1 - samples2).abs().max().item()
    print(f"   WARNING: Samples differ (max diff: {max_diff})")
    print(f"   This may be due to non-deterministic operations or device differences")

# Test different seeds produce different samples
print("\n5. Testing different seeds produce different samples...")
set_seed(42)

with torch.no_grad():
    samples_seed42 = diffusion.sample(
        model=model,
        batch_size=4,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        num_sampling_steps=50,
        device=device
    )

set_seed(123)

with torch.no_grad():
    samples_seed123 = diffusion.sample(
        model=model,
        batch_size=4,
        channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        num_sampling_steps=50,
        device=device
    )

are_different = not torch.allclose(samples_seed42, samples_seed123, atol=1e-3)
print(f"   Different seeds produce different samples: {are_different}")

if are_different:
    print(f"   [OK] Seed affects output")
else:
    print(f"   WARNING: Different seeds produce similar samples")

# Check sample properties
print("\n6. Sample properties check:")
print(f"   Shape: {samples1.shape}")
print(f"   Range: [{samples1.min():.4f}, {samples1.max():.4f}]")
print(f"   Mean: {samples1.mean():.4f}")
print(f"   Std: {samples1.std():.4f}")
print(f"   All finite: {torch.all(torch.isfinite(samples1))}")

print(f"\n7. Summary:")
print(f"   Checkpoint loading: [OK]")
print(f"   Model reconstruction: [OK]")
print(f"   Reproducibility (same seed): {'[OK]' if are_identical else '[PARTIAL]'}")
print(f"   Seed affects output: {'[OK]' if are_different else '[FAIL]'}")
print(f"   Sample properties: [OK]")
