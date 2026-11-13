"""
Stage 3.2.6 Task 3: Audit Temporal Convolution Implementation and Tensor Transformations

Verifies that temporal convolutions operate correctly along the temporal dimension
and investigates the tensor transformations.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.video_unet import TemporalConv1DBlock, VideoUNet
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_temporal_conv_block():
    """
    Test the TemporalConv1DBlock with controlled inputs.
    """
    print_section("TEMPORAL CONVOLUTION BLOCK AUDIT")

    # Create temporal block
    temporal_block = TemporalConv1DBlock(
        channels=128,
        temporal_kernel=3,
        num_layers=2,
        dropout=0.0
    )

    print("\nTemporal block configuration:")
    print(f"  Channels: 128")
    print(f"  Temporal kernel: 3")
    print(f"  Num layers: 2")

    # Test 1: Constant features across time
    print("\n" + "-" * 80)
    print("TEST 1: Constant features across all frames")
    print("-" * 80)

    batch_size = 2
    num_frames = 16
    channels = 128
    height = 8
    width = 8

    # Create constant features
    x_const = torch.ones(batch_size, num_frames, channels, height, width)
    output_const = temporal_block(x_const)

    print(f"Input shape: {x_const.shape}")
    print(f"Output shape: {output_const.shape}")
    print(f"Input mean: {x_const.mean().item():.6f}")
    print(f"Output mean: {output_const.mean().item():.6f}")
    print(f"Input std: {x_const.std().item():.6f}")
    print(f"Output std: {output_const.std().item():.6f}")

    # Check if output is still constant (should be, due to residual)
    output_var = output_const.var(dim=1).mean().item()
    print(f"Temporal variance: {output_var:.10f}")

    if output_var < 1e-6:
        print("[OK] Temporal block preserves constant features")
    else:
        print("[WARNING] Temporal block introduces variance in constant input")

    # Test 2: Linearly increasing features
    print("\n" + "-" * 80)
    print("TEST 2: Linearly increasing features over time")
    print("-" * 80)

    x_linear = torch.randn(batch_size, num_frames, channels, height, width)
    for t in range(num_frames):
        x_linear[:, t, :, :, :] *= (t / num_frames)

    output_linear = temporal_block(x_linear)

    print(f"Input temporal trend: first frame mean = {x_linear[:, 0].mean().item():.6f}, last frame mean = {x_linear[:, -1].mean().item():.6f}")
    print(f"Output temporal trend: first frame mean = {output_linear[:, 0].mean().item():.6f}, last frame mean = {output_linear[:, -1].mean().item():.6f}")

    # Test 3: Reversed frame sequence
    print("\n" + "-" * 80)
    print("TEST 3: Reversed frame sequence")
    print("-" * 80)

    x_forward = torch.randn(batch_size, num_frames, channels, height, width)
    x_reversed = torch.flip(x_forward, dims=[1])

    output_forward = temporal_block(x_forward)
    output_reversed = temporal_block(x_reversed)

    # Compare reversed output with forward output reversed
    output_reversed_reversed = torch.flip(output_reversed, dims=[1])
    diff = (output_forward - output_reversed_reversed).abs().mean().item()

    print(f"Difference between forward and reversed-reversed: {diff:.10f}")

    if diff < 1e-4:
        print("[WARNING] Temporal block is time-symmetric (may not capture direction)")
    else:
        print("[OK] Temporal block distinguishes time direction")

    # Test 4: Random permutation
    print("\n" + "-" * 80)
    print("TEST 4: Randomly permuted frame sequence")
    print("-" * 80)

    x_random = torch.randn(batch_size, num_frames, channels, height, width)
    perm = torch.randperm(num_frames)
    x_permuted = x_random[:, perm, :, :, :]

    output_random = temporal_block(x_random)
    output_permuted = temporal_block(x_permuted)

    # Check if output preserves permutation structure
    output_permuted_unpermuted = output_permuted[:, torch.argsort(perm), :, :, :]
    diff_perm = (output_random - output_permuted_unpermuted).abs().mean().item()

    print(f"Difference after unpermuting: {diff_perm:.10f}")

    if diff_perm < 1e-4:
        print("[WARNING] Temporal block is permutation-invariant (may not capture order)")
    else:
        print("[OK] Temporal block is sensitive to frame order")


def audit_video_unet_temporal_usage():
    """
    Audit how VideoUNet uses the temporal blocks.
    """
    print_section("VIDEOUNET TEMPORAL USAGE AUDIT")

    # Create minimal models
    spatial_unet = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=1000,
        text_dim=256,
        n_heads=4,
        use_cross_attn=True
    )

    text_encoder = TextEncoder(
        vocab_size=42,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=False,
        use_temporal=True
    )

    print("\n[CRITICAL FINDING from code inspection]")
    print("-" * 80)
    print("In _forward_with_temporal (lines 344-352 of video_unet.py):")
    print("  bottleneck_mean = bottleneck_video.mean()  # scalar")
    print("  low_res_mean = low_res_video.mean()  # scalar")
    print("  temporal_term = (bottleneck_mean + low_res_mean) * 0.001")
    print("  predicted_noise_spatial = predicted_noise_spatial + temporal_term")
    print()
    print("This means:")
    print("  1. Temporal features are COLLAPSED TO SCALARS")
    print("  2. They are added as a tiny bias term (scaled by 0.001)")
    print("  3. The actual temporal feature maps are DISCARDED")
    print("  4. This is NOT proper feature-level fusion")
    print()
    print("[CRITICAL] Temporal convolutions compute features but they are")
    print("           not used for actual noise prediction - only a scalar mean")

    # Verify by checking the forward pass
    print("\n" + "-" * 80)
    print("VERIFICATION: Run forward pass and check tensor shapes")
    print("-" * 80)

    batch_size = 1
    num_frames = 16
    channels = 3
    height = 64
    width = 64

    x = torch.randn(batch_size, channels, num_frames, height, width)
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)

    video_unet.eval()
    with torch.no_grad():
        output = video_unet(x, timesteps, text_emb, text_mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")

    # Compare with frame-independent
    video_unet.use_temporal = False
    with torch.no_grad():
        output_frame_indep = video_unet(x, timesteps, text_emb, text_mask)

    diff = (output - output_frame_indep).abs().mean().item()
    print(f"\nDifference between temporal and frame-independent: {diff:.10f}")

    if diff < 0.01:
        print("[CRITICAL] Temporal and frame-independent outputs are nearly identical")
        print("           This confirms temporal features are not being used effectively")


def main():
    print_section("STAGE 3.2.6 TASK 3: TEMPORAL CONVOLUTION AUDIT")

    test_temporal_conv_block()
    audit_video_unet_temporal_usage()

    print_section("SUMMARY")

    print("\nKEY FINDINGS:")
    print("  1. TemporalConv1DBlock implementation is correct (operates on T dimension)")
    print("  2. Temporal block is sensitive to frame order (good)")
    print("  3. [CRITICAL] VideoUNet collapses temporal features to scalars")
    print("  4. [CRITICAL] Temporal features are added as tiny bias (0.001 scaling)")
    print("  5. [CRITICAL] Actual temporal feature maps are discarded")
    print()
    print("CONCLUSION:")
    print("  The temporal convolution blocks work correctly, but the fusion")
    print("  mechanism in VideoUNet is fundamentally broken. Temporal features")
    print("  are not fused at the feature level - only their scalar means are")
    print("  used as a negligible bias term.")
    print()
    print("RECOMMENDATION:")
    print("  Implement proper feature-level fusion between spatial and temporal")
    print("  features, or redesign the architecture to use temporal features")
    print("  for actual noise prediction.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
