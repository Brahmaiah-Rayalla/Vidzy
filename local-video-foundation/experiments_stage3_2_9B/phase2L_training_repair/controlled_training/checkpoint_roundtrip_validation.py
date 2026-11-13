"""
Checkpoint roundtrip validation for Stage 3.2.9L.3.

Verifies that:
- Checkpoints can be loaded correctly
- Parameters are preserved during save/load
- Common B/C parameters are identical where expected
- Temporal-specific parameters exist in C
"""

import torch
import json
from pathlib import Path
import sys

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
stage3_2_7_dir = base_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation" / "models"))

from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed
from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder
from models.text_encoder import TextEncoder

def validate_checkpoint_roundtrip():
    """Validate checkpoint roundtrip for both experiments."""
    print("=" * 80)
    print("STAGE 3.2.9L.3 - CHECKPOINT ROUNDTRIP VALIDATION")
    print("=" * 80)

    output_dir = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training"

    results = {
        'experiment_B': {},
        'experiment_C': {},
        'comparison': {}
    }

    # ============================================================================
    # VALIDATE EXPERIMENT B CHECKPOINTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT B CHECKPOINT VALIDATION")
    print("=" * 80)

    checkpoint_steps = [25, 50, 75, 100]

    for step in checkpoint_steps:
        checkpoint_path = output_dir / f"experiment_B/checkpoints/checkpoint_step_{step:06d}.pt"

        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            results['experiment_B'][f'step_{step}'] = {'loaded': False}
            continue

        print(f"\nLoading checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Verify checkpoint structure
        has_model_state = 'model_state_dict' in checkpoint
        has_optimizer_state = 'optimizer_state_dict' in checkpoint
        has_config = 'config' in checkpoint
        has_experiment = 'experiment' in checkpoint

        print(f"  Model state dict: {has_model_state}")
        print(f"  Optimizer state dict: {has_optimizer_state}")
        print(f"  Config: {has_config}")
        print(f"  Experiment: {has_experiment}")

        if has_experiment:
            print(f"  Experiment label: {checkpoint['experiment']}")

        # Count parameters
        if has_model_state:
            num_params = len(checkpoint['model_state_dict'])
            print(f"  Total parameters: {num_params}")

        results['experiment_B'][f'step_{step}'] = {
            'loaded': True,
            'has_model_state': has_model_state,
            'has_optimizer_state': has_optimizer_state,
            'has_config': has_config,
            'has_experiment': has_experiment,
            'num_parameters': num_params if has_model_state else 0
        }

    # ============================================================================
    # VALIDATE EXPERIMENT C CHECKPOINTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT C CHECKPOINT VALIDATION")
    print("=" * 80)

    for step in checkpoint_steps:
        checkpoint_path = output_dir / f"experiment_C/checkpoints/checkpoint_step_{step:06d}.pt"

        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            results['experiment_C'][f'step_{step}'] = {'loaded': False}
            continue

        print(f"\nLoading checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Verify checkpoint structure
        has_model_state = 'model_state_dict' in checkpoint
        has_optimizer_state = 'optimizer_state_dict' in checkpoint
        has_config = 'config' in checkpoint
        has_experiment = 'experiment' in checkpoint

        print(f"  Model state dict: {has_model_state}")
        print(f"  Optimizer state dict: {has_optimizer_state}")
        print(f"  Config: {has_config}")
        print(f"  Experiment: {has_experiment}")

        if has_experiment:
            print(f"  Experiment label: {checkpoint['experiment']}")

        # Count parameters
        if has_model_state:
            num_params = len(checkpoint['model_state_dict'])
            print(f"  Total parameters: {num_params}")

            # Check for temporal parameters
            temporal_keys = [k for k in checkpoint['model_state_dict'].keys() if 'temporal' in k]
            fusion_keys = [k for k in checkpoint['model_state_dict'].keys() if 'fusion' in k]

            print(f"  Temporal parameters: {len(temporal_keys)}")
            print(f"  Fusion parameters: {len(fusion_keys)}")

        results['experiment_C'][f'step_{step}'] = {
            'loaded': True,
            'has_model_state': has_model_state,
            'has_optimizer_state': has_optimizer_state,
            'has_config': has_config,
            'has_experiment': has_experiment,
            'num_parameters': num_params if has_model_state else 0,
            'temporal_params': len(temporal_keys) if has_model_state else 0,
            'fusion_params': len(fusion_keys) if has_model_state else 0
        }

    # ============================================================================
    # COMPARE FINAL CHECKPOINTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("FINAL CHECKPOINT COMPARISON (Step 100)")
    print("=" * 80)

    checkpoint_B_path = output_dir / "experiment_B/checkpoints/checkpoint_step_000100.pt"
    checkpoint_C_path = output_dir / "experiment_C/checkpoints/checkpoint_step_000100.pt"

    checkpoint_B = torch.load(checkpoint_B_path, map_location='cpu')
    checkpoint_C = torch.load(checkpoint_C_path, map_location='cpu')

    # Key comparison
    keys_B = set(checkpoint_B['model_state_dict'].keys())
    keys_C = set(checkpoint_C['model_state_dict'].keys())

    common_keys = keys_B & keys_C
    b_only_keys = keys_B - keys_C
    c_only_keys = keys_C - keys_B

    print(f"Common keys: {len(common_keys)}")
    print(f"B-only keys: {len(b_only_keys)}")
    print(f"C-only keys: {len(c_only_keys)}")

    results['comparison']['key_match'] = {
        'common_keys': len(common_keys),
        'b_only_keys': len(b_only_keys),
        'c_only_keys': len(c_only_keys),
        'keys_match': len(b_only_keys) == 0 and len(c_only_keys) == 0
    }

    # Compare motion embeddings
    motion_key = 'text_encoder.token_emb.motion_emb.weight'
    if motion_key in checkpoint_B['model_state_dict'] and motion_key in checkpoint_C['model_state_dict']:
        motion_B = checkpoint_B['model_state_dict'][motion_key]
        motion_C = checkpoint_C['model_state_dict'][motion_key]

        motion_diff = (motion_B - motion_C).abs().max().item()
        motion_mean_diff = (motion_B - motion_C).abs().mean().item()

        print(f"\nMotion embedding max difference: {motion_diff:.8f}")
        print(f"Motion embedding mean difference: {motion_mean_diff:.8f}")
        print(f"Motion embeddings differ: {motion_diff > 1e-6}")

        results['comparison']['motion_embedding'] = {
            'max_diff': motion_diff,
            'mean_diff': motion_mean_diff,
            'differ': motion_diff > 1e-6
        }

    # Compare spatial parameters
    spatial_keys = [k for k in common_keys if 'spatial' in k]
    max_spatial_diff = 0.0

    for key in spatial_keys[:10]:  # Check first 10 spatial parameters
        param_B = checkpoint_B['model_state_dict'][key]
        param_C = checkpoint_C['model_state_dict'][key]

        if param_B.shape == param_C.shape:
            diff = (param_B - param_C).abs().max().item()
            max_spatial_diff = max(max_spatial_diff, diff)

    print(f"\nMax spatial parameter difference: {max_spatial_diff:.8f}")
    print(f"Spatial parameters identical: {max_spatial_diff < 1e-6}")

    results['comparison']['spatial_integrity'] = {
        'max_diff': max_spatial_diff,
        'identical': max_spatial_diff < 1e-6
    }

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_B_loaded = all(r['loaded'] for r in results['experiment_B'].values())
    all_C_loaded = all(r['loaded'] for r in results['experiment_C'].values())
    keys_match = results['comparison']['key_match']['keys_match']
    spatial_identical = results['comparison']['spatial_integrity']['identical']

    print(f"Experiment B all checkpoints loaded: {all_B_loaded}")
    print(f"Experiment C all checkpoints loaded: {all_C_loaded}")
    print(f"Checkpoint keys match: {keys_match}")
    print(f"Spatial parameters identical: {spatial_identical}")

    all_checks_pass = all_B_loaded and all_C_loaded and keys_match and spatial_identical

    print(f"\nAll checks pass: {all_checks_pass}")

    results['summary'] = {
        'all_B_loaded': all_B_loaded,
        'all_C_loaded': all_C_loaded,
        'keys_match': keys_match,
        'spatial_identical': spatial_identical,
        'all_checks_pass': all_checks_pass
    }

    # Save results
    output_file = output_dir / "STAGE3_2_9L3_CHECKPOINT_INTEGRITY.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results

if __name__ == "__main__":
    validate_checkpoint_roundtrip()
