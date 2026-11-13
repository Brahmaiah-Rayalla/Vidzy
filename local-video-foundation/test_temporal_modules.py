"""
Test temporal modules in VideoUNet.

Tests:
- Temporal module tensor shapes
- Gradient flow through temporal modules
- Frame order changes temporal output
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import torch
import torch.nn as nn
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet


def test_temporal_module_shapes():
    """Test that temporal modules have correct tensor shapes."""
    print("Testing temporal module tensor shapes...")

    # Create models
    spatial_unet = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        text_dim=256,
        n_heads=4
    )

    text_encoder = TextEncoder(
        vocab_size=42,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512
    )

    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    print(f"Temporal parameters: {video_unet.temporal_params:,}")

    # Test inputs
    batch_size = 2
    num_frames = 16
    x = torch.randn(batch_size, 3, num_frames, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)

    # Forward pass
    with torch.no_grad():
        output = video_unet(x, timesteps, text_emb, text_mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, f"Output shape mismatch"

    print("Temporal module shape test PASSED")


def test_temporal_gradient_flow():
    """Test that gradients flow through temporal modules."""
    print("\nTesting temporal module gradient flow...")

    # Create models
    spatial_unet = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        text_dim=256,
        n_heads=4
    )

    text_encoder = TextEncoder(
        vocab_size=42,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512
    )

    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    # Test inputs
    x = torch.randn(1, 3, 16, 64, 64, requires_grad=True)
    timesteps = torch.randint(0, 1000, (1,))
    text_emb = torch.randn(1, 32, 256)
    text_mask = torch.ones(1, 32)

    # Forward pass
    output = video_unet(x, timesteps, text_emb, text_mask)

    # Compute loss
    loss = output.mean()

    # Backward pass
    loss.backward()

    # Check gradients in temporal modules
    temporal_params_with_grad = 0
    temporal_params_total = 0

    for name, param in video_unet.named_parameters():
        if 'temporal' in name:
            temporal_params_total += 1
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    temporal_params_with_grad += 1
                    print(f"  Gradient in {name}: {param.grad.abs().sum().item():.6f}")

    print(f"\nTemporal parameters with gradients: {temporal_params_with_grad}/{temporal_params_total}")

    assert temporal_params_with_grad > 0, "No gradients in temporal modules"

    print("Temporal gradient flow test PASSED")


def test_frame_order_sensitivity():
    """Test that changing frame order changes temporal output."""
    print("\nTesting frame order sensitivity...")

    # Create models
    spatial_unet = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        text_dim=256,
        n_heads=4
    )

    text_encoder = TextEncoder(
        vocab_size=42,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512
    )

    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    # Create inputs with different frame order
    x = torch.randn(1, 3, 16, 64, 64)
    x_reversed = torch.flip(x, dims=[2])  # Reverse frame order

    timesteps = torch.randint(0, 1000, (1,))
    text_emb = torch.randn(1, 32, 256)
    text_mask = torch.ones(1, 32)

    # Forward pass with original order
    with torch.no_grad():
        output_normal = video_unet(x, timesteps, text_emb, text_mask)

    # Forward pass with reversed order
    with torch.no_grad():
        output_reversed = video_unet(x_reversed, timesteps, text_emb, text_mask)

    # Check if outputs are different
    diff = (output_normal - output_reversed).abs().mean().item()
    print(f"Mean absolute difference between normal and reversed: {diff:.6f}")

    assert diff > 0.01, "Frame order does not affect output (temporal modules not working)"

    print("Frame order sensitivity test PASSED")


def test_comparison_with_baseline():
    """Compare temporal vs baseline (frame-independent) model."""
    print("\nComparing temporal vs baseline models...")

    # Create baseline (no temporal)
    spatial_unet_baseline = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        text_dim=256,
        n_heads=4
    )

    text_encoder_baseline = TextEncoder(
        vocab_size=42,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512
    )

    video_unet_baseline = VideoUNet(
        spatial_unet=spatial_unet_baseline,
        text_encoder=text_encoder_baseline,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=False
    )

    # Create temporal model
    spatial_unet_temporal = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        text_dim=256,
        n_heads=4
    )

    text_encoder_temporal = TextEncoder(
        vocab_size=42,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512
    )

    video_unet_temporal = VideoUNet(
        spatial_unet=spatial_unet_temporal,
        text_encoder=text_encoder_temporal,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    # Copy weights from baseline to temporal for fair comparison
    video_unet_temporal.spatial_unet.load_state_dict(video_unet_baseline.spatial_unet.state_dict())
    video_unet_temporal.text_encoder.load_state_dict(video_unet_baseline.text_encoder.state_dict())

    # Test inputs
    x = torch.randn(1, 3, 16, 64, 64)
    timesteps = torch.randint(0, 1000, (1,))
    text_emb = torch.randn(1, 32, 256)
    text_mask = torch.ones(1, 32)

    # Forward pass baseline
    with torch.no_grad():
        output_baseline = video_unet_baseline(x, timesteps, text_emb, text_mask)

    # Forward pass temporal
    with torch.no_grad():
        output_temporal = video_unet_temporal(x, timesteps, text_emb, text_mask)

    print(f"Baseline parameters: {video_unet_baseline.temporal_params:,}")
    print(f"Temporal parameters: {video_unet_temporal.temporal_params:,}")
    print(f"Output difference (mean abs): {(output_baseline - output_temporal).abs().mean().item():.6f}")

    print("Comparison test PASSED")


if __name__ == "__main__":
    test_temporal_module_shapes()
    test_temporal_gradient_flow()
    test_frame_order_sensitivity()
    test_comparison_with_baseline()
    print("\n" + "="*50)
    print("All temporal module tests PASSED")
    print("="*50)
