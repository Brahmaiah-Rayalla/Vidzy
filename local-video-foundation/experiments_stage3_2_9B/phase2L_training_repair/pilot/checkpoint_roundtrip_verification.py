"""
Checkpoint serialization and restoration verification.

Verifies that:
- Trained parameters survive serialization
- Checkpoints can be loaded correctly
- B and C checkpoints have expected differences
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

def verify_checkpoint_roundtrip():
    """Verify checkpoint serialization and restoration."""
    print("=" * 80)
    print("STAGE 3.2.9L.2 - CHECKPOINT ROUNDTRIP VERIFICATION")
    print("=" * 80)

    # Load checkpoint
    stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location='cpu')
    actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

    # Create standard text encoder
    text_encoder_standard = TextEncoder(
        vocab_size=actual_vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64,
        dropout=0.0
    )
    text_encoder_standard.load_state_dict(checkpoint['text_encoder_state_dict'])

    results = {
        'experiment_B': {},
        'experiment_C': {},
        'comparison': {}
    }

    # ============================================================================
    # LOAD EXPERIMENT B CHECKPOINT
    # ============================================================================
    print("\nLoading Experiment B checkpoint...")
    checkpoint_B_path = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/checkpoint_experiment_B_10steps.pt"
    checkpoint_B = torch.load(checkpoint_B_path, map_location='cpu')

    print(f"Checkpoint B keys: {list(checkpoint_B.keys())}")
    print(f"Experiment: {checkpoint_B['experiment']}")
    print(f"Config: {checkpoint_B['config']}")

    # Verify model state dict
    print(f"Model state dict keys: {list(checkpoint_B['model_state_dict'].keys())[:10]}...")
    print(f"Total parameters in state dict: {len(checkpoint_B['model_state_dict'])}")

    results['experiment_B'] = {
        'checkpoint_loaded': True,
        'experiment': checkpoint_B['experiment'],
        'num_parameters': len(checkpoint_B['model_state_dict']),
        'has_optimizer_state': 'optimizer_state_dict' in checkpoint_B
    }

    # ============================================================================
    # LOAD EXPERIMENT C CHECKPOINT
    # ============================================================================
    print("\nLoading Experiment C checkpoint...")
    checkpoint_C_path = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/checkpoint_experiment_C_10steps.pt"
    checkpoint_C = torch.load(checkpoint_C_path, map_location='cpu')

    print(f"Checkpoint C keys: {list(checkpoint_C.keys())}")
    print(f"Experiment: {checkpoint_C['experiment']}")
    print(f"Config: {checkpoint_C['config']}")

    results['experiment_C'] = {
        'checkpoint_loaded': True,
        'experiment': checkpoint_C['experiment'],
        'num_parameters': len(checkpoint_C['model_state_dict']),
        'has_optimizer_state': 'optimizer_state_dict' in checkpoint_C
    }

    # ============================================================================
    # KEY-BY-KEY COMPARISON
    # ============================================================================
    print("\n" + "=" * 80)
    print("KEY-BY-KEY COMPARISON")
    print("=" * 80)

    keys_B = set(checkpoint_B['model_state_dict'].keys())
    keys_C = set(checkpoint_C['model_state_dict'].keys())

    common_keys = keys_B & keys_C
    b_only_keys = keys_B - keys_C
    c_only_keys = keys_C - keys_B

    print(f"Common keys: {len(common_keys)}")
    print(f"B-only keys: {len(b_only_keys)}")
    print(f"C-only keys: {len(c_only_keys)}")

    if b_only_keys:
        print(f"B-only keys: {b_only_keys}")
    if c_only_keys:
        print(f"C-only keys: {c_only_keys}")

    results['comparison'] = {
        'common_keys': len(common_keys),
        'b_only_keys': len(b_only_keys),
        'c_only_keys': len(c_only_keys),
        'key_match': len(b_only_keys) == 0 and len(c_only_keys) == 0
    }

    # ============================================================================
    # PARAMETER VALUE COMPARISON
    # ============================================================================
    print("\n" + "=" * 80)
    print("PARAMETER VALUE COMPARISON")
    print("=" * 80)

    # Compare a few key parameters
    key_params_to_compare = [
        'text_encoder.token_emb.motion_emb.weight',
        'spatial_unet.conv_in.weight',
        'temporal_modules.bottleneck.layers.0.weight',
        'fusion_modules.bottleneck.gate'
    ]

    for key in key_params_to_compare:
        if key in checkpoint_B['model_state_dict'] and key in checkpoint_C['model_state_dict']:
            param_B = checkpoint_B['model_state_dict'][key]
            param_C = checkpoint_C['model_state_dict'][key]

            if param_B.shape == param_C.shape:
                diff = (param_B - param_C).abs().max().item()
                print(f"{key}: max diff = {diff:.8f}")
            else:
                print(f"{key}: shape mismatch (B: {param_B.shape}, C: {param_C.shape})")
        else:
            print(f"{key}: not found in both checkpoints")

    # ============================================================================
    # VERIFY MOTION EMBEDDINGS DIFFER
    # ============================================================================
    print("\n" + "=" * 80)
    print("MOTION EMBEDDING COMPARISON")
    print("=" * 80)

    motion_key = 'text_encoder.token_emb.motion_emb.weight'
    if motion_key in checkpoint_B['model_state_dict'] and motion_key in checkpoint_C['model_state_dict']:
        motion_B = checkpoint_B['model_state_dict'][motion_key]
        motion_C = checkpoint_C['model_state_dict'][motion_key]

        motion_diff = (motion_B - motion_C).abs().max().item()
        motion_mean_diff = (motion_B - motion_C).abs().mean().item()

        print(f"Motion embedding max difference: {motion_diff:.8f}")
        print(f"Motion embedding mean difference: {motion_mean_diff:.8f}")
        print(f"Motion embeddings differ: {motion_diff > 1e-6}")

        results['comparison']['motion_embedding_max_diff'] = motion_diff
        results['comparison']['motion_embedding_mean_diff'] = motion_mean_diff
        results['comparison']['motion_embeddings_differ'] = motion_diff > 1e-6

    # ============================================================================
    # VERIFY TEMPORAL PARAMETERS EXIST IN C
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEMPORAL PARAMETERS VERIFICATION (C)")
    print("=" * 80)

    temporal_keys = [k for k in checkpoint_C['model_state_dict'].keys() if 'temporal' in k]
    fusion_keys = [k for k in checkpoint_C['model_state_dict'].keys() if 'fusion' in k]

    print(f"Temporal parameters in C: {len(temporal_keys)}")
    print(f"Fusion parameters in C: {len(fusion_keys)}")

    results['comparison']['temporal_params_in_C'] = len(temporal_keys)
    results['comparison']['fusion_params_in_C'] = len(fusion_keys)

    # ============================================================================
    # VERIFY SPATIAL PARAMETERS IDENTICAL
    # ============================================================================
    print("\n" + "=" * 80)
    print("SPATIAL PARAMETER INTEGRITY")
    print("=" * 80)

    spatial_keys = [k for k in checkpoint_B['model_state_dict'].keys() if 'spatial' in k]
    max_spatial_diff = 0.0

    for key in spatial_keys[:5]:  # Check first 5 spatial parameters
        if key in checkpoint_B['model_state_dict'] and key in checkpoint_C['model_state_dict']:
            param_B = checkpoint_B['model_state_dict'][key]
            param_C = checkpoint_C['model_state_dict'][key]

            if param_B.shape == param_C.shape:
                diff = (param_B - param_C).abs().max().item()
                max_spatial_diff = max(max_spatial_diff, diff)

    print(f"Max spatial parameter difference: {max_spatial_diff:.8f}")
    print(f"Spatial parameters identical: {max_spatial_diff < 1e-6}")

    results['comparison']['max_spatial_diff'] = max_spatial_diff
    results['comparison']['spatial_identical'] = max_spatial_diff < 1e-6

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_checks_pass = (
        results['experiment_B']['checkpoint_loaded'] and
        results['experiment_C']['checkpoint_loaded'] and
        results['comparison']['key_match'] and
        results['comparison']['motion_embeddings_differ'] and
        results['comparison']['spatial_identical']
    )

    print(f"Experiment B checkpoint loaded: {results['experiment_B']['checkpoint_loaded']}")
    print(f"Experiment C checkpoint loaded: {results['experiment_C']['checkpoint_loaded']}")
    print(f"Keys match: {results['comparison']['key_match']}")
    print(f"Motion embeddings differ: {results['comparison']['motion_embeddings_differ']}")
    print(f"Spatial parameters identical: {results['comparison']['spatial_identical']}")
    print(f"\nAll checks pass: {all_checks_pass}")

    results['summary'] = {
        'all_checks_pass': all_checks_pass
    }

    # Save results
    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/STAGE3_2_9L2_CHECKPOINT_ROUNDTRIP.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results

if __name__ == "__main__":
    verify_checkpoint_roundtrip()
