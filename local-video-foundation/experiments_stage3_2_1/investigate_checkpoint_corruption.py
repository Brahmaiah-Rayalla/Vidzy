"""
Investigate whether video training corrupted spatial backbone weights.

Compare spatial backbone weights before and after video training.
"""

import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def investigate_checkpoint_corruption(
    stage2_checkpoint_path,
    video_checkpoint_path
):
    print_section("CHECKPOINT CORRUPTION INVESTIGATION")

    # Load Stage 2 checkpoint
    print(f"\n1. Loading Stage 2 checkpoint: {stage2_checkpoint_path}")
    stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location='cpu')

    if 'model_state_dict' in stage2_checkpoint:
        stage2_state_dict = stage2_checkpoint['model_state_dict']
    else:
        stage2_state_dict = stage2_checkpoint

    print(f"Stage 2 parameters: {len(stage2_state_dict)}")

    # Load video checkpoint
    print(f"\n2. Loading video checkpoint: {video_checkpoint_path}")
    video_checkpoint = torch.load(video_checkpoint_path, map_location='cpu')

    if 'model_state_dict' in video_checkpoint:
        video_state_dict = video_checkpoint['model_state_dict']
    else:
        video_state_dict = video_checkpoint

    print(f"Video checkpoint parameters: {len(video_state_dict)}")

    print_section("3. SPATIAL BACKBone WEIGHT COMPARISON")

    # Extract spatial backbone weights from video checkpoint
    # Video UNet wraps spatial_unet, so weights are prefixed
    spatial_prefix = 'spatial_unet.'

    video_spatial_weights = {}
    for key, value in video_state_dict.items():
        if key.startswith(spatial_prefix):
            # Remove prefix
            unprefixed_key = key[len(spatial_prefix):]
            video_spatial_weights[unprefixed_key] = value

    print(f"Video spatial backbone parameters: {len(video_spatial_weights)}")

    # Compare weights
    print("\nComparing spatial backbone weights...")

    total_params = len(stage2_state_dict)
    matching_params = 0
    mismatching_params = 0
    missing_in_video = 0
    extra_in_video = 0

    max_diff = 0.0
    max_diff_key = None

    for key in stage2_state_dict.keys():
        if key in video_spatial_weights:
            stage2_param = stage2_state_dict[key]
            video_param = video_spatial_weights[key]

            if torch.allclose(stage2_param, video_param, rtol=1e-5, atol=1e-8):
                matching_params += 1
            else:
                mismatching_params += 1
                diff = (stage2_param - video_param).abs().max().item()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_key = key

                if mismatching_params <= 5:
                    print(f"  Mismatching: {key}")
                    print(f"    Stage 2 range: [{stage2_param.min().item():.6f}, {stage2_param.max().item():.6f}]")
                    print(f"    Video range:   [{video_param.min().item():.6f}, {video_param.max().item():.6f}]")
                    print(f"    Max diff: {diff:.8f}")
        else:
            missing_in_video += 1

    for key in video_spatial_weights.keys():
        if key not in stage2_state_dict:
            extra_in_video += 1

    print(f"\nParameter comparison summary:")
    print(f"  Total Stage 2 parameters: {total_params}")
    print(f"  Matching: {matching_params}")
    print(f"  Mismatching: {mismatching_params}")
    print(f"  Missing in video: {missing_in_video}")
    print(f"  Extra in video: {extra_in_video}")

    if max_diff > 0:
        print(f"\nMaximum difference: {max_diff:.8f} (key: {max_diff_key})")

    print_section("4. TRAINING CONFIGURATION CHECK")

    # Check if spatial was frozen during training
    print("\nChecking training configuration...")

    # Look for freeze_spatial flag in checkpoint config
    if 'config' in video_checkpoint:
        config = video_checkpoint['config']
        print(f"Checkpoint config: {config}")
        if 'freeze_spatial' in config:
            print(f"Freeze spatial setting: {config['freeze_spatial']}")
    else:
        print("No config found in checkpoint")

    # Check parameter requires_grad status
    # We can't check this from saved checkpoint, but we can infer from weight changes
    if mismatching_params > 0:
        print(f"\n[WARNING] {mismatching_params} spatial parameters changed during training")
        print("This suggests spatial backbone was NOT frozen during training")
    else:
        print(f"\n[OK] All spatial parameters unchanged")
        print("Spatial backbone was likely frozen during training")

    print_section("5. GRADIENT NORM ANALYSIS")

    # Check if temporal parameters have reasonable gradients
    print("\nAnalyzing temporal parameters...")

    temporal_keys = [k for k in video_state_dict.keys() if 'temporal' in k.lower()]
    print(f"Temporal parameters found: {len(temporal_keys)}")

    if len(temporal_keys) > 0:
        print(f"Sample temporal keys: {temporal_keys[:5]}")

    results = {
        'total_stage2_params': total_params,
        'matching_params': matching_params,
        'mismatching_params': mismatching_params,
        'missing_in_video': missing_in_video,
        'extra_in_video': extra_in_video,
        'max_diff': float(max_diff) if max_diff > 0 else 0.0,
        'max_diff_key': max_diff_key,
        'spatial_corrupted': mismatching_params > 0,
    }

    with open("checkpoint_corruption_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to checkpoint_corruption_analysis.json")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Investigate checkpoint corruption")
    parser.add_argument("--stage2_checkpoint", type=str, default="../checkpoints_conditional/checkpoint_final.pt", help="Stage 2 checkpoint")
    parser.add_argument("--video_checkpoint", type=str, default="checkpoints_controlled/checkpoint_final.pt", help="Video checkpoint")
    args = parser.parse_args()

    results = investigate_checkpoint_corruption(args.stage2_checkpoint, args.video_checkpoint)


if __name__ == "__main__":
    main()
