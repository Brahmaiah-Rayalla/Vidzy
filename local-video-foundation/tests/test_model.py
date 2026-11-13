"""
Tests for the U-Net model.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.models.unet import UNet


def test_model_output_shape():
    """Test that the model produces output with the correct shape."""
    print("Testing model output shape...")

    model = UNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=1000
    )

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))

    # Forward pass
    output = model(x, timesteps)

    # Check output shape
    assert output.shape == (batch_size, 3, 64, 64), \
        f"Expected output shape (4, 3, 64, 64), got {output.shape}"

    print("[PASS] Model output shape test passed")


def test_model_initialization():
    """Test that the model initializes correctly."""
    print("Testing model initialization...")

    model = UNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=1000
    )

    # Check that parameters are not all zeros
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model should have parameters"

    # Check that parameters are not all the same (random initialization)
    first_param = next(iter(model.parameters()))
    assert not torch.allclose(first_param, torch.zeros_like(first_param)), \
        "Parameters should be randomly initialized, not zeros"

    print("[PASS] Model initialization test passed")


def test_model_gradient_flow():
    """Test that gradients flow through the model."""
    print("Testing gradient flow...")

    model = UNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=1000
    )

    # Create dummy input
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    timesteps = torch.randint(0, 1000, (2,))

    # Forward pass
    output = model(x, timesteps)

    # Compute loss
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None, "Input should have gradients"
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), \
        "Gradients should not be zero"

    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} should have gradients"

    print("[PASS] Gradient flow test passed")


if __name__ == "__main__":
    test_model_output_shape()
    test_model_initialization()
    test_model_gradient_flow()

    print("\n" + "="*60)
    print("All model tests passed!")
    print("="*60)
