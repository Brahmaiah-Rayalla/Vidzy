"""
Verify that freeze_spatial actually works in VideoUNet.

Test whether the spatial backbone is correctly frozen when freeze_spatial=True.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def verify_freeze_spatial():
    print_section("VERIFY FREEZE_SPATIAL FUNCTIONALITY")

    # Create models
    print("\n1. Creating models...")
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

    print_section("2. TEST WITHOUT FREEZING")

    # Create video UNet without freezing
    video_unet_unfrozen = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=False,
        use_temporal=True
    )

    # Check requires_grad status
    spatial_params_unfrozen = list(video_unet_unfrozen.spatial_unet.parameters())
    frozen_count_unfrozen = sum(1 for p in spatial_params_unfrozen if not p.requires_grad)
    trainable_count_unfrozen = sum(1 for p in spatial_params_unfrozen if p.requires_grad)

    print(f"\nSpatial parameters (unfrozen):")
    print(f"  Total: {len(spatial_params_unfrozen)}")
    print(f"  Trainable (requires_grad=True): {trainable_count_unfrozen}")
    print(f"  Frozen (requires_grad=False): {frozen_count_unfrozen}")

    print_section("3. TEST WITH FREEZING")

    # Create video UNet with freezing
    spatial_unet2 = ConditionalUNet(
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

    text_encoder2 = TextEncoder(
        vocab_size=42,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    video_unet_frozen = VideoUNet(
        spatial_unet=spatial_unet2,
        text_encoder=text_encoder2,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    # Check requires_grad status
    spatial_params_frozen = list(video_unet_frozen.spatial_unet.parameters())
    frozen_count_frozen = sum(1 for p in spatial_params_frozen if not p.requires_grad)
    trainable_count_frozen = sum(1 for p in spatial_params_frozen if p.requires_grad)

    print(f"\nSpatial parameters (frozen):")
    print(f"  Total: {len(spatial_params_frozen)}")
    print(f"  Trainable (requires_grad=True): {trainable_count_frozen}")
    print(f"  Frozen (requires_grad=False): {frozen_count_frozen}")

    print_section("4. CHECK TOTAL TRAINABLE PARAMETERS")

    # Check total trainable parameters in the entire model
    total_trainable_unfrozen = sum(p.numel() for p in video_unet_unfrozen.parameters() if p.requires_grad)
    total_trainable_frozen = sum(p.numel() for p in video_unet_frozen.parameters() if p.requires_grad)

    print(f"\nTotal trainable parameters:")
    print(f"  Unfrozen: {total_trainable_unfrozen:,}")
    print(f"  Frozen: {total_trainable_frozen:,}")
    print(f"  Difference: {total_trainable_unfrozen - total_trainable_frozen:,}")

    print_section("5. TEST OPTIMIZER PARAMETER GROUPS")

    # Test optimizer parameter selection
    import torch.optim as optim

    print("\nUnfrozen model optimizer:")
    temporal_params_unfrozen = list(video_unet_unfrozen.temporal_modules.parameters())
    fusion_params_unfrozen = list(video_unet_unfrozen.bottleneck_fusion.parameters()) + list(video_unet_unfrozen.low_res_fusion.parameters())

    print(f"  Temporal params: {len(temporal_params_unfrozen)}")
    print(f"  Fusion params: {len(fusion_params_unfrozen)}")

    optimizer_unfrozen = optim.Adam([
        {'params': temporal_params_unfrozen, 'lr': 1e-4},
        {'params': fusion_params_unfrozen, 'lr': 1e-4}
    ])

    print(f"  Optimizer param groups: {len(optimizer_unfrozen.param_groups)}")
    for i, group in enumerate(optimizer_unfrozen.param_groups):
        print(f"    Group {i}: {len(group['params'])} parameters")

    print("\nFrozen model optimizer:")
    temporal_params_frozen = list(video_unet_frozen.temporal_modules.parameters())
    fusion_params_frozen = list(video_unet_frozen.bottleneck_fusion.parameters()) + list(video_unet_frozen.low_res_fusion.parameters())

    print(f"  Temporal params: {len(temporal_params_frozen)}")
    print(f"  Fusion params: {len(fusion_params_frozen)}")

    optimizer_frozen = optim.Adam([
        {'params': temporal_params_frozen, 'lr': 1e-4},
        {'params': fusion_params_frozen, 'lr': 1e-4}
    ])

    print(f"  Optimizer param groups: {len(optimizer_frozen.param_groups)}")
    for i, group in enumerate(optimizer_frozen.param_groups):
        print(f"    Group {i}: {len(group['params'])} parameters")

    print_section("6. CONCLUSION")

    if trainable_count_frozen == 0 and trainable_count_unfrozen > 0:
        print("\n[OK] freeze_spatial works correctly")
        print("  - Without freeze_spatial: spatial parameters are trainable")
        print("  - With freeze_spatial: spatial parameters are frozen")
    else:
        print("\n[ERROR] freeze_spatial does NOT work correctly")
        print(f"  - Expected frozen trainable count: 0")
        print(f"  - Actual frozen trainable count: {trainable_count_frozen}")

    return {
        'unfrozen_trainable_spatial': trainable_count_unfrozen,
        'frozen_trainable_spatial': trainable_count_frozen,
        'freeze_works': trainable_count_frozen == 0
    }


if __name__ == "__main__":
    results = verify_freeze_spatial()
