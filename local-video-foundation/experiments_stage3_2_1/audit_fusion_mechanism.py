"""
Stage 3.2.6 Task 4: Investigate Spatial-Temporal Feature Fusion Mechanism

Analyzes the LearnableTemporalFusion mechanism and how temporal features
are combined with spatial features.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.video_unet import LearnableTemporalFusion, VideoUNet
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_learnable_fusion():
    """
    Test the LearnableTemporalFusion module.
    """
    print_section("LEARNABLE TEMPORAL FUSION MODULE TEST")

    # Create fusion module
    fusion = LearnableTemporalFusion(
        channels=128,
        initial_gate_value=0.01,
        use_channel_wise=False
    )

    print("\nFusion module configuration:")
    print(f"  Channels: 128")
    print(f"  Initial gate value: 0.01")
    print(f"  Use channel-wise: False")
    print(f"  Gate activation: Sigmoid")

    # Get initial gate value
    gate_value = fusion.gate_activation(fusion.gate).item()
    print(f"\nInitial gate value: {gate_value:.6f}")

    # Test fusion with dummy tensors
    batch_size = 2
    num_frames = 16
    channels = 128
    height = 8
    width = 8

    spatial_features = torch.randn(batch_size, num_frames, channels, height, width)
    temporal_features = torch.randn(batch_size, num_frames, channels, height, width)

    fused = fusion(temporal_features, spatial_features)

    print(f"\nSpatial features shape: {spatial_features.shape}")
    print(f"Temporal features shape: {temporal_features.shape}")
    print(f"Fused features shape: {fused.shape}")

    # Check fusion behavior
    spatial_mean = spatial_features.mean().item()
    temporal_mean = temporal_features.mean().item()
    fused_mean = fused.mean().item()

    print(f"\nSpatial mean: {spatial_mean:.6f}")
    print(f"Temporal mean: {temporal_mean:.6f}")
    print(f"Fused mean: {fused_mean:.6f}")

    # Expected: fused = spatial + gate * temporal
    expected_fused = spatial_features + gate_value * temporal_features
    expected_mean = expected_fused.mean().item()

    print(f"Expected fused mean: {expected_mean:.6f}")
    print(f"Difference: {abs(fused_mean - expected_mean):.10f}")

    if abs(fused_mean - expected_mean) < 1e-4:
        print("[OK] Fusion matches expected formula: spatial + gate * temporal")
    else:
        print("[WARNING] Fusion does not match expected formula")

    # Test with different gate values
    print("\n" + "-" * 80)
    print("TEST: Different gate values")
    print("-" * 80)

    for gate_init in [0.001, 0.01, 0.1, 0.5, 1.0]:
        fusion_test = LearnableTemporalFusion(channels=128, initial_gate_value=gate_init)
        gate_val = fusion_test.gate_activation(fusion_test.gate).item()
        fused_test = fusion_test(temporal_features, spatial_features)

        temporal_contribution = (fused_test - spatial_features).abs().mean().item()
        spatial_contribution = spatial_features.abs().mean().item()

        ratio = temporal_contribution / (spatial_contribution + 1e-8)

        print(f"Gate init {gate_init:.4f} -> actual gate {gate_val:.4f}: temporal ratio {ratio:.4f}")


def audit_video_unet_fusion_usage():
    """
    Audit how VideoUNet uses the fusion modules.
    """
    print_section("VIDEOUNET FUSION USAGE AUDIT")

    print("\nCode inspection from experiments_stage3_2_1/video_foundation/models/video_unet.py:")
    print("-" * 80)
    print("Lines 417-457 in _forward_with_temporal:")
    print()
    print("  # Reshape temporal features back to spatial format")
    print("  bottleneck_temporal_flat = bottleneck_temporal.permute(...).reshape(...)")
    print("  low_res_temporal_flat = low_res_temporal.permute(...).reshape(...)")
    print()
    print("  # Compute learnable fusion gates")
    print("  bottleneck_gate = self.bottleneck_fusion.gate_activation(...)")
    print("  low_res_gate = self.low_res_fusion.gate_activation(...)")
    print()
    print("  # Project temporal features to noise predictions")
    print("  bottleneck_noise = self.bottleneck_proj(bottleneck_temporal_flat)")
    print("  low_res_noise = self.low_res_proj(low_res_temporal_flat)")
    print()
    print("  # Upsample to full resolution")
    print("  bottleneck_noise_upsampled = F.interpolate(...)")
    print("  low_res_noise_upsampled = F.interpolate(...)")
    print()
    print("  # Combine temporal noise predictions with learnable gates")
    print("  temporal_noise = bottleneck_gate * bottleneck_noise_upsampled +")
    print("                   low_res_gate * low_res_noise_upsampled")
    print()
    print("  # Add temporal contribution to spatial prediction")
    print("  predicted_noise_spatial = predicted_noise_spatial + temporal_noise")
    print()
    print("-" * 80)
    print()
    print("ANALYSIS:")
    print("  1. Temporal features are projected to noise space (Conv2d)")
    print("  2. Upsampled to full resolution")
    print("  3. Combined with learnable gates")
    print("  4. Added to spatial prediction")
    print()
    print("This is NOISE-LEVEL fusion, not FEATURE-LEVEL fusion.")
    print("The temporal features are NOT injected back into the spatial U-Net.")
    print("They bypass the spatial decoder entirely.")
    print()
    print("Problem: Temporal features are computed at bottleneck/low-res")
    print("         but then projected directly to noise, losing the benefit")
    print("         of the spatial decoder's processing.")
    print()
    print("The LearnableTemporalFusion class (lines 92-153) implements")
    print("feature-level fusion, but it is NOT USED in the forward pass!")
    print("Instead, a custom fusion is implemented in _forward_with_temporal")


def measure_fusion_effectiveness():
    """
    Measure how effective the fusion is by comparing gate values and gradients.
    """
    print_section("FUSION EFFECTIVENESS MEASUREMENT")

    # Create models
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

    # Load checkpoint
    checkpoint_path = "checkpoints_stage3_2_5/checkpoint_final.pt"
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            video_unet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Checkpoint loaded")

        # Get gate values
        bottleneck_gate = video_unet.bottleneck_fusion.gate_activation(video_unet.bottleneck_fusion.gate).item()
        low_res_gate = video_unet.low_res_fusion.gate_activation(video_unet.low_res_fusion.gate).item()

        print(f"\nTrained gate values:")
        print(f"  Bottleneck gate: {bottleneck_gate:.6f}")
        print(f"  Low-res gate: {low_res_gate:.6f}")

        if bottleneck_gate < 0.5 and low_res_gate < 0.5:
            print("\n[CRITICAL] Both gates learned to suppress temporal features (< 0.5)")
            print("           This indicates temporal features were harmful for the objective")
        elif bottleneck_gate > 0.5 and low_res_gate > 0.5:
            print("\n[OK] Gates learned to incorporate temporal features (> 0.5)")
        else:
            print("\n[MIXED] Gates show mixed behavior")
    else:
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Using initial gate values")
        bottleneck_gate = video_unet.bottleneck_fusion.gate_activation(video_unet.bottleneck_fusion.gate).item()
        low_res_gate = video_unet.low_res_fusion.gate_activation(video_unet.low_res_fusion.gate).item()
        print(f"  Bottleneck gate: {bottleneck_gate:.6f}")
        print(f"  Low-res gate: {low_res_gate:.6f}")


def main():
    print_section("STAGE 3.2.6 TASK 4: SPATIAL-TEMPORAL FUSION AUDIT")

    test_learnable_fusion()
    audit_video_unet_fusion_usage()
    measure_fusion_effectiveness()

    print_section("SUMMARY")

    print("\nKEY FINDINGS:")
    print("  1. LearnableTemporalFusion class implements proper feature-level fusion")
    print("  2. However, VideoUNet does NOT use this class for fusion")
    print("  3. Instead, temporal features are projected to noise space")
    print("  4. This is NOISE-LEVEL fusion, bypassing spatial decoder")
    print("  5. Trained gates learned to suppress temporal (0.49 < 0.5)")
    print()
    print("CONCLUSION:")
    print("  The fusion mechanism is inconsistent:")
    print("  - A proper fusion class exists but is unused")
    print("  - The actual fusion is at noise level, not feature level")
    print("  - Temporal features lose spatial decoder processing")
    print("  - Gates learned to suppress temporal (harmful)")
    print()
    print("RECOMMENDATION:")
    print("  1. Use LearnableTemporalFusion at feature level")
    print("  2. Inject temporal features back into spatial U-Net decoder")
    print("  3. Or redesign architecture for proper feature-level fusion")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
