"""
Audit video diffusion mathematics for Stage 3.2.3 Task 2.

Verifies noise distribution, timestep sampling, and loss calculation.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess
from video_foundation.training.train_video import SyntheticVideoDataset


def audit_diffusion_math(data_dir):
    """Audit diffusion mathematics with deterministic test tensors."""
    print("=" * 80)
    print("DIFFUSION MATHEMATICS AUDIT")
    print("=" * 80)

    # Load dataset
    print("\nLoading dataset...")
    dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=16, image_size=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # Test with deterministic tensors
    print("\nTesting with deterministic tensors...")
    batch_size = 1
    num_frames = 16
    image_size = 64

    # Create deterministic clean video
    clean_video = torch.randn(batch_size * num_frames, 3, image_size, image_size)
    timesteps = torch.tensor([500]).expand(batch_size * num_frames)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    noisy_video, noise = diffusion.q_sample(clean_video, timesteps)

    print("\n1. NOISE DISTRIBUTION:")
    print(f"   Noise shape: {noise.shape}")
    print(f"   Noise mean: {noise.mean().item():.6f}")
    print(f"   Noise std: {noise.std().item():.6f}")
    print(f"   Noise min: {noise.min().item():.6f}")
    print(f"   Noise max: {noise.max().item():.6f}")
    print(f"   Expected: mean ~0.0, std ~1.0")
    print(f"   Status: {'[PASS]' if abs(noise.mean().item()) < 0.1 and 0.9 < noise.std().item() < 1.1 else '[FAIL]'}")

    print("\n2. TENSOR SHAPES:")
    print(f"   Clean video: {clean_video.shape}")
    print(f"   Noise: {noise.shape}")
    print(f"   Noisy video: {noisy_video.shape}")
    print(f"   Timesteps: {timesteps.shape}")
    print(f"   Status: {'[PASS]' if clean_video.shape == noise.shape == noisy_video.shape else '[FAIL]'}")

    print("\n3. ALPHA/CUMULATIVE ALPHA:")
    print(f"   Alpha at t=500: {schedule.alphas[500]:.6f}")
    print(f"   Cumulative alpha at t=500: {schedule.alphas_cumprod[500]:.6f}")
    print(f"   Sqrt cumulative alpha: {torch.sqrt(schedule.alphas_cumprod[500]):.6f}")
    print(f"   Sqrt(1 - cumulative alpha): {torch.sqrt(1 - schedule.alphas_cumprod[500]):.6f}")

    # Verify q_sample formula manually
    print("\n4. VERIFY Q_SAMPLE FORMULA:")
    alpha_cumprod = schedule.alphas_cumprod[500]
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

    manual_noisy = sqrt_alpha_cumprod * clean_video + sqrt_one_minus_alpha_cumprod * noise
    match = torch.allclose(noisy_video, manual_noisy, atol=1e-5)
    print(f"   Manual calculation matches q_sample: {'[PASS]' if match else '[FAIL]'}")

    print("\n5. LOSS CALCULATION:")
    # Simulate prediction
    predicted_noise = torch.randn_like(noise)
    loss = nn.functional.mse_loss(predicted_noise, noise)
    print(f"   MSE loss: {loss.item():.6f}")
    print(f"   Expected: ~1.0 for random prediction vs random noise")
    print(f"   Status: {'[PASS]' if 0.9 < loss.item() < 1.1 else '[FAIL]'}")

    # Test with actual data
    print("\n6. TESTING WITH ACTUAL DATA:")
    batch = next(iter(dataloader))
    videos = batch['video'].to('cpu')
    batch_size = videos.shape[0]

    videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
    timesteps = torch.randint(0, 1000, (batch_size * num_frames,))

    torch.manual_seed(42)
    noisy_videos, actual_noise = diffusion.q_sample(videos_flat, timesteps)

    print(f"   Actual noise mean: {actual_noise.mean().item():.6f}")
    print(f"   Actual noise std: {actual_noise.std().item():.6f}")
    print(f"   Noisy video mean: {noisy_videos.mean().item():.6f}")
    print(f"   Noisy video std: {noisy_videos.std().item():.6f}")

    # Zero prediction baseline
    zero_pred = torch.zeros_like(actual_noise)
    zero_loss = nn.functional.mse_loss(zero_pred, actual_noise)
    print(f"   Zero-prediction loss: {zero_loss.item():.6f}")
    print(f"   Expected: ~1.0")
    print(f"   Status: {'[PASS]' if 0.9 < zero_loss.item() < 1.1 else '[FAIL]'}")

    print("\n" + "=" * 80)

    return {
        'noise_mean': float(noise.mean().item()),
        'noise_std': float(noise.std().item()),
        'shape_match': clean_video.shape == noise.shape == noisy_video.shape,
        'q_sample_formula_match': match,
        'zero_pred_loss': float(zero_loss.item()),
        'actual_noise_mean': float(actual_noise.mean().item()),
        'actual_noise_std': float(actual_noise.std().item())
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit diffusion mathematics")
    parser.add_argument("--data_dir", type=str, default="data/controlled_video", help="Data directory")
    args = parser.parse_args()

    results = audit_diffusion_math(args.data_dir)

    with open("diffusion_math_audit.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to diffusion_math_audit.json")


if __name__ == "__main__":
    main()
