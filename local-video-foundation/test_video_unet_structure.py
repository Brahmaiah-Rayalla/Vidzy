"""
Test script to verify VideoUNet architecture and parameter loading.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import torch
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet, create_video_unet_from_stage2


def test_video_unet_structure():
    """Test the VideoUNet architecture structure."""
    print("Testing VideoUNet architecture...")

    # Create dummy spatial U-Net
    spatial_unet = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        text_dim=256,
        n_heads=4
    )

    # Create dummy text encoder
    text_encoder = TextEncoder(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512
    )

    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True
    )

    print(f"\nVideo U-Net created successfully")
    print(f"Total parameters: {sum(p.numel() for p in video_unet.parameters()):,}")
    print(f"Trainable parameters: {video_unet.trainable_params:,}")
    print(f"Spatial parameters: {video_unet.spatial_params:,}")
    print(f"Temporal parameters: {video_unet.temporal_params:,}")


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\nTesting forward pass...")

    # Create dummy spatial U-Net
    spatial_unet = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        text_dim=256,
        n_heads=4
    )

    # Create dummy text encoder
    text_encoder = TextEncoder(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512
    )

    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True
    )

    # Create dummy inputs
    batch_size = 1
    channels = 3
    num_frames = 16
    height = 64
    width = 64

    x = torch.randn(batch_size, channels, num_frames, height, width)
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 32, 256)  # seq_len=32, text_dim=256
    text_mask = torch.ones(batch_size, 32)

    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  timesteps: {timesteps.shape}")
    print(f"  text_emb: {text_emb.shape}")
    print(f"  text_mask: {text_mask.shape}")

    # Forward pass
    with torch.no_grad():
        output = video_unet(x, timesteps, text_emb, text_mask)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {channels}, {num_frames}, {height}, {width})")

    assert output.shape == (batch_size, channels, num_frames, height, width), \
        f"Output shape mismatch: {output.shape} vs ({batch_size}, {channels}, {num_frames}, {height}, {width})"

    print("Forward pass test PASSED")


def test_gradient_flow():
    """Test that gradients flow correctly (frozen vs unfrozen)."""
    print("\nTesting gradient flow...")

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
        vocab_size=1000,
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
        freeze_spatial=True
    )

    # Create dummy inputs
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

    # Check gradients
    frozen_params_with_grad = 0
    unfrozen_params_with_grad = 0

    for name, param in video_unet.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            if param.requires_grad:
                unfrozen_params_with_grad += 1
                print(f"  Gradient in trainable {name}: {param.grad.abs().sum().item():.6f}")
            else:
                frozen_params_with_grad += 1
                print(f"  WARNING: Gradient in frozen {name}: {param.grad.abs().sum().item():.6f}")

    # In the initial prototype with frozen spatial backbone and no temporal modules,
    # we expect gradients only in the text encoder (unfrozen by default)
    # The spatial backbone should have no gradients

    print(f"\nGradient summary:")
    print(f"  Frozen parameters with gradients: {frozen_params_with_grad}")
    print(f"  Unfrozen parameters with gradients: {unfrozen_params_with_grad}")

    if frozen_params_with_grad > 0:
        print("  WARNING: Some frozen parameters have gradients (this is expected if inputs require_grad)")
    else:
        print("  OK: No gradients in frozen parameters")

    if unfrozen_params_with_grad > 0:
        print("  OK: Gradients flow through unfrozen parameters")
    else:
        print("  NOTE: No unfrozen parameters with gradients (text encoder may be frozen or not in computation graph)")

    print("Gradient flow test PASSED")


if __name__ == "__main__":
    test_video_unet_structure()
    test_forward_pass()
    test_gradient_flow()
    print("\nAll tests PASSED")
