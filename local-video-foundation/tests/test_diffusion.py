"""
Tests for the diffusion process.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess
from video_foundation.diffusion.sampler import DiffusionSampler


def test_schedule_computation():
    """Test that the noise schedule is computed correctly."""
    print("Testing schedule computation...")

    schedule = GaussianDiffusionSchedule(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="linear"
    )

    # Check that betas are in the correct range
    assert schedule.betas[0] >= 0.0001, "First beta should be >= beta_start"
    assert schedule.betas[-1] <= 0.02, "Last beta should be <= beta_end"

    # Check that alphas are in (0, 1]
    assert torch.all(schedule.alphas > 0), "Alphas should be positive"
    assert torch.all(schedule.alphas <= 1), "Alphas should be <= 1"

    # Check that alpha_bar is monotonically decreasing
    assert torch.all(schedule.alphas_cumprod[:-1] >= schedule.alphas_cumprod[1:]), \
        "Alpha_bar should be monotonically decreasing"

    print("[PASS] Schedule computation test passed")


def test_forward_diffusion():
    """Test that forward diffusion adds noise correctly."""
    print("Testing forward diffusion...")

    schedule = GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    process = DiffusionProcess(schedule)

    # Create dummy clean image
    x_0 = torch.randn(2, 3, 64, 64)
    timesteps = torch.tensor([0, 999])  # First and last timesteps

    # Forward diffusion
    x_t, noise = process.q_sample(x_0, timesteps)

    # Check shapes
    assert x_t.shape == x_0.shape, f"Expected shape {x_0.shape}, got {x_t.shape}"
    assert noise.shape == x_0.shape, f"Expected shape {x_0.shape}, got {noise.shape}"

    # Check that at t=0, x_t is close to x_0
    assert torch.allclose(x_t[0], x_0[0], atol=0.1), "At t=0, x_t should be close to x_0"

    # Check that at high t, x_t is very noisy
    assert not torch.allclose(x_t[1], x_0[1], atol=0.1), "At high t, x_t should be noisy"

    print("[PASS] Forward diffusion test passed")


def test_training_loss():
    """Test that training loss is computed correctly."""
    print("Testing training loss...")

    from video_foundation.models.unet import UNet

    schedule = GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    process = DiffusionProcess(schedule)

    model = UNet(
        in_channels=3,
        model_channels=32,  # Smaller for faster test
        channel_mult=[1, 2],
        num_res_blocks=1,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=1000
    )

    # Create dummy clean image
    x_0 = torch.randn(2, 3, 64, 64)
    timesteps = torch.randint(0, 1000, (2,))

    # Compute loss
    loss = process.p_losses(model, x_0, timesteps)

    # Check that loss is finite
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss > 0, "Loss should be positive"

    print("[PASS] Training loss test passed")


def test_ddim_sampling():
    """Test that DDIM sampling generates images."""
    print("Testing DDIM sampling...")

    from video_foundation.models.unet import UNet

    schedule = GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    process = DiffusionProcess(schedule)

    model = UNet(
        in_channels=3,
        model_channels=32,
        channel_mult=[1, 2],
        num_res_blocks=1,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=1000
    )

    # Sample images
    samples = process.ddim_sample_loop(
        model=model,
        batch_size=2,
        channels=3,
        image_size=64,
        num_sampling_steps=10  # Small number for fast test
    )

    # Check shape
    assert samples.shape == (2, 3, 64, 64), f"Expected shape (2, 3, 64, 64), got {samples.shape}"

    # Check that samples are finite
    assert torch.all(torch.isfinite(samples)), "Samples should be finite"

    print("[PASS] DDIM sampling test passed")


if __name__ == "__main__":
    test_schedule_computation()
    test_forward_diffusion()
    test_training_loss()
    test_ddim_sampling()

    print("\n" + "="*60)
    print("All diffusion tests passed!")
    print("="*60)
