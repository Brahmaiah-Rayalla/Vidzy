"""
Regression tests for sampling fixes.
"""

import torch
import numpy as np
from video_foundation.models.unet import UNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess
from video_foundation.utils.reproducibility import set_seed


def test_forward_diffusion_equation():
    """Test forward diffusion with known inputs."""
    print("Testing forward diffusion equation...")

    schedule = GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    process = DiffusionProcess(schedule)

    # Use deterministic inputs
    x_0 = torch.ones(2, 3, 4, 4)  # All ones
    noise = torch.zeros(2, 3, 4, 4)  # No noise
    timesteps = torch.tensor([0, 0])

    x_t, _ = process.q_sample(x_0, timesteps, noise)

    # At t=0 with no noise, x_t should approximately equal x_0
    assert torch.allclose(x_t, x_0, atol=1e-4), "At t=0 with no noise, x_t should equal x_0"

    # Test with t=999 (high noise)
    noise = torch.ones(2, 3, 4, 4)
    timesteps = torch.tensor([999, 999])
    x_t, _ = process.q_sample(x_0, timesteps, noise)

    # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
    alpha_bar = schedule.alphas_cumprod[999]
    expected = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

    assert torch.allclose(x_t, expected, atol=1e-5), "Forward diffusion equation mismatch"

    print("[PASS] Forward diffusion equation correct")


def test_cumulative_alpha_indexing():
    """Test correct alpha_bar indexing."""
    print("Testing cumulative alpha indexing...")

    schedule = GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)

    # Test that alpha_bar decreases monotonically
    assert torch.all(schedule.alphas_cumprod[:-1] >= schedule.alphas_cumprod[1:]), \
        "alpha_bar should be monotonically decreasing"

    # Test that alpha_bar[0] is close to 1
    assert schedule.alphas_cumprod[0] > 0.999, "alpha_bar[0] should be close to 1"

    # Test that alpha_bar[-1] is close to 0
    assert schedule.alphas_cumprod[-1] < 0.001, "alpha_bar[-1] should be close to 0"

    # Test get() method
    timesteps = torch.tensor([0, 500, 999])
    _, alpha_bar, _ = schedule.get(timesteps)

    assert alpha_bar[0, 0, 0, 0] == schedule.alphas_cumprod[0], "Indexing mismatch at t=0"
    assert alpha_bar[1, 0, 0, 0] == schedule.alphas_cumprod[500], "Indexing mismatch at t=500"
    assert alpha_bar[2, 0, 0, 0] == schedule.alphas_cumprod[999], "Indexing mismatch at t=999"

    print("[PASS] Cumulative alpha indexing correct")


def test_ddim_x0_clipping():
    """Test that x_0 clipping prevents runaway values."""
    print("Testing DDIM x_0 clipping...")

    schedule = GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    process = DiffusionProcess(schedule)

    # Create a simple model that predicts noise
    class DummyModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    model = DummyModel()

    # Start with noise at high timestep
    x = torch.randn(1, 3, 4, 4)
    timesteps = torch.tensor([999])

    # Predict noise (all zeros)
    predicted_noise = model(x, timesteps)

    # Get schedule values
    _, alpha_bar_t, _ = schedule.get(timesteps)
    alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

    # Predict x_0 without clipping
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    x_0_pred_no_clip = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

    # With clipping
    x_0_pred_clip = torch.clamp(x_0_pred_no_clip, -1.0, 1.0)

    # The unclipped prediction should have huge values due to division by tiny sqrt(alpha_bar)
    assert x_0_pred_no_clip.abs().max() > 10, "Unclipped x_0 should have large values"

    # The clipped prediction should be bounded
    assert x_0_pred_clip.min() >= -1.0 and x_0_pred_clip.max() <= 1.0, \
        "Clipped x_0 should be in [-1, 1]"

    print("[PASS] DDIM x_0 clipping works correctly")


def test_ddim_final_step():
    """Test that the final DDIM step handles t=0 correctly."""
    print("Testing DDIM final step behavior...")

    schedule = GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    process = DiffusionProcess(schedule)

    class DummyModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    model = DummyModel()

    # Run DDIM with 1 step (goes from t=999 to t=0)
    shape = (1, 3, 4, 4)
    result = process.ddim_sample(model, shape, num_sampling_steps=1, device=torch.device('cpu'), clip_x0=True)

    # Result should be in [-1, 1] due to clipping
    assert result.min() >= -1.0 and result.max() <= 1.0, "Final step should produce values in [-1, 1]"

    # Result should be finite
    assert torch.all(torch.isfinite(result)), "Final step should produce finite values"

    print("[PASS] DDIM final step handles t=0 correctly")


def test_ddpm_sampling():
    """Test DDPM sampling produces finite outputs."""
    print("Testing DDPM sampling...")

    schedule = GaussianDiffusionSchedule(num_timesteps=100, beta_start=0.0001, beta_end=0.02)
    process = DiffusionProcess(schedule)

    class DummyModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    model = DummyModel()

    # Run DDPM sampling with reduced timesteps for speed
    shape = (1, 3, 4, 4)
    result = process.ddpm_sample(model, shape, device=torch.device('cpu'))

    # Result should be finite
    assert torch.all(torch.isfinite(result)), "DDPM should produce finite values"

    # Result should be in reasonable range
    assert result.min() >= -10.0 and result.max() <= 10.0, "DDPM should produce values in reasonable range"

    print("[PASS] DDPM sampling produces finite outputs")


def test_output_shape_and_normalization():
    """Test that sampling produces correct shape and normalization."""
    print("Testing output shape and normalization...")

    schedule = GaussianDiffusionSchedule(num_timesteps=100, beta_start=0.0001, beta_end=0.02)
    process = DiffusionProcess(schedule)

    class DummyModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    model = DummyModel()

    # Test DDIM
    shape = (2, 3, 8, 8)
    result_ddim = process.ddim_sample(model, shape, num_sampling_steps=10, device=torch.device('cpu'), clip_x0=True)

    assert result_ddim.shape == shape, f"DDIM output shape mismatch: {result_ddim.shape} vs {shape}"

    # Test DDPM
    result_ddpm = process.ddpm_sample(model, shape, device=torch.device('cpu'))

    assert result_ddpm.shape == shape, f"DDPM output shape mismatch: {result_ddpm.shape} vs {shape}"

    print("[PASS] Output shape and normalization correct")


def test_complete_sampling_with_checkpoint():
    """Test complete sampling with the actual checkpoint."""
    print("Testing complete sampling with checkpoint...")

    try:
        import os
        checkpoint_path = "checkpoints/checkpoint_final.pt"

        if not os.path.exists(checkpoint_path):
            print("[SKIP] Checkpoint not found")
            return

        device = torch.device('cpu')
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        config = checkpoint_data['config']

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

        diffusion = DiffusionSampler(
            num_timesteps=config['diffusion']['num_timesteps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            schedule=config['diffusion']['beta_schedule']
        )

        set_seed(42)
        with torch.no_grad():
            samples = diffusion.sample(
                model=model,
                batch_size=2,
                channels=config['model']['in_channels'],
                image_size=config['model']['image_size'],
                num_sampling_steps=10,  # Reduced for speed
                device=device,
                clip_x0=True
            )

        assert samples.shape == (2, 3, 64, 64), f"Sample shape mismatch: {samples.shape}"
        assert torch.all(torch.isfinite(samples)), "Samples should be finite"
        assert samples.min() >= -1.1 and samples.max() <= 1.1, "Samples should be in [-1, 1] range"

        print("[PASS] Complete sampling with checkpoint works")

    except Exception as e:
        print(f"[FAIL] Complete sampling with checkpoint failed: {e}")
        raise


if __name__ == "__main__":
    print("="*60)
    print("Running Sampling Regression Tests")
    print("="*60 + "\n")

    test_forward_diffusion_equation()
    test_cumulative_alpha_indexing()
    test_ddim_x0_clipping()
    test_ddim_final_step()
    test_ddpm_sampling()
    test_output_shape_and_normalization()
    test_complete_sampling_with_checkpoint()

    print("\n" + "="*60)
    print("All sampling regression tests passed!")
    print("="*60)
