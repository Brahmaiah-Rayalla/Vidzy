"""
Stage 3.2.9L - Complete B/C Checkpoint Comparison

Performs a comprehensive comparison of Experiment B and C checkpoints,
including all keys, tensor values, shapes, dtypes, and optimizer state.

Author: Devin
Date: 2026-11-04
"""

import torch
import json
from pathlib import Path
import sys
from collections import defaultdict

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "video_foundation"))

def compute_tensor_statistics(tensor):
    """Compute comprehensive statistics for a tensor."""
    if not isinstance(tensor, torch.Tensor):
        return {
            'is_tensor': False,
            'type': str(type(tensor))
        }

    stats = {
        'is_tensor': True,
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'norm': tensor.norm().item(),
        'num_parameters': tensor.numel(),
        'requires_grad': tensor.requires_grad
    }
    return stats

def compare_nested_dicts(dict_a, dict_b):
    """Compare two nested dictionaries (like model state dicts)."""
    keys_a = set(dict_a.keys())
    keys_b = set(dict_b.keys())

    result = {
        'keys_a_only': list(keys_a - keys_b),
        'keys_b_only': list(keys_b - keys_a),
        'common_keys': list(keys_a & keys_b),
        'total_keys_a': len(keys_a),
        'total_keys_b': len(keys_b),
        'tensor_comparisons': {},
        'summary': {}
    }

    exact_matches = 0
    total_tensors = 0
    max_diff = 0.0

    for key in result['common_keys']:
        val_a = dict_a[key]
        val_b = dict_b[key]

        if isinstance(val_a, torch.Tensor) and isinstance(val_b, torch.Tensor):
            comp = compare_tensors(val_a, val_b)
            result['tensor_comparisons'][key] = comp
            total_tensors += 1

            if comp['exact_match']:
                exact_matches += 1
            else:
                if comp['max_abs_diff'] > max_diff:
                    max_diff = comp['max_abs_diff']

    result['summary'] = {
        'total_tensors': total_tensors,
        'exact_matches': exact_matches,
        'different_tensors': total_tensors - exact_matches,
        'match_percentage': (exact_matches / total_tensors * 100) if total_tensors > 0 else 0,
        'max_diff': max_diff
    }

    return result

def compare_tensors(tensor_a, tensor_b):
    """Compare two tensors with multiple metrics."""
    if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
        return {
            'both_tensors': False,
            'type_a': str(type(tensor_a)),
            'type_b': str(type(tensor_b))
        }

    # Check shapes
    shape_match = tensor_a.shape == tensor_b.shape

    # Check dtypes
    dtype_match = tensor_a.dtype == tensor_b.dtype

    # Check exact equality
    exact_match = torch.equal(tensor_a, tensor_b)

    # Compute differences
    diff = tensor_a - tensor_b
    max_abs_diff = diff.abs().max().item()
    mean_abs_diff = diff.abs().mean().item()

    # Relative L2 difference
    norm_a = tensor_a.norm().item()
    norm_b = tensor_b.norm().item()
    if norm_a > 0 and norm_b > 0:
        relative_l2_diff = (diff.norm().item()) / ((norm_a + norm_b) / 2)
    else:
        relative_l2_diff = 0.0 if norm_a == norm_b else 1.0

    return {
        'both_tensors': True,
        'shape_match': shape_match,
        'dtype_match': dtype_match,
        'exact_match': exact_match,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'relative_l2_diff': relative_l2_diff,
        'norm_a': norm_a,
        'norm_b': norm_b
    }

def complete_checkpoint_comparison():
    """Perform complete checkpoint comparison."""
    checkpoint_B_path = base_dir / "experiments_stage3_2_9B/phase2C/checkpoints/checkpoint_phase2c_B.pt"
    checkpoint_C_path = base_dir / "experiments_stage3_2_9B/phase2C/checkpoints/checkpoint_phase2c_C.pt"

    print("=" * 80)
    print("STAGE 3.2.9L - COMPLETE B/C CHECKPOINT COMPARISON")
    print("=" * 80)

    print(f"\nLoading checkpoint B: {checkpoint_B_path}")
    checkpoint_B = torch.load(checkpoint_B_path, map_location='cpu')

    print(f"Loading checkpoint C: {checkpoint_C_path}")
    checkpoint_C = torch.load(checkpoint_C_path, map_location='cpu')

    comparison = {
        'checkpoint_B_path': str(checkpoint_B_path),
        'checkpoint_C_path': str(checkpoint_C_path),
        'key_set_analysis': {},
        'tensor_comparisons': {},
        'temporal_analysis': {},
        'motion_embedding_analysis': {},
        'text_encoder_analysis': {},
        'optimizer_analysis': {},
        'summary': {}
    }

    # Key set analysis
    print("\n" + "=" * 80)
    print("KEY SET ANALYSIS")
    print("=" * 80)

    keys_B = set(checkpoint_B.keys())
    keys_C = set(checkpoint_C.keys())

    comparison['key_set_analysis'] = {
        'keys_B_only': list(keys_B - keys_C),
        'keys_C_only': list(keys_C - keys_B),
        'common_keys': list(keys_B & keys_C),
        'total_keys_B': len(keys_B),
        'total_keys_C': len(keys_C),
        'common_key_count': len(keys_B & keys_C)
    }

    print(f"Keys in B only: {len(comparison['key_set_analysis']['keys_B_only'])}")
    print(f"Keys in C only: {len(comparison['key_set_analysis']['keys_C_only'])}")
    print(f"Common keys: {len(comparison['key_set_analysis']['common_keys'])}")

    if comparison['key_set_analysis']['keys_B_only']:
        print("\nKeys in B only:")
        for key in comparison['key_set_analysis']['keys_B_only']:
            print(f"  - {key}")

    if comparison['key_set_analysis']['keys_C_only']:
        print("\nKeys in C only:")
        for key in comparison['key_set_analysis']['keys_C_only']:
            print(f"  - {key}")

    # Tensor comparisons for all common keys
    print("\n" + "=" * 80)
    print("TENSOR COMPARISON")
    print("=" * 80)

    temporal_keys = []
    motion_embedding_keys = []
    text_encoder_keys = []
    optimizer_keys = []

    for key in sorted(comparison['key_set_analysis']['common_keys']):
        value_B = checkpoint_B[key]
        value_C = checkpoint_C[key]

        # Categorize keys
        if 'temporal' in key.lower():
            temporal_keys.append(key)
        if 'motion' in key.lower() or 'embedding' in key.lower():
            motion_embedding_keys.append(key)
        if 'text' in key.lower() or 'encoder' in key.lower():
            text_encoder_keys.append(key)
        if 'optimizer' in key.lower():
            optimizer_keys.append(key)

        # If this is the video_unet dict, also categorize its internal keys
        if key == 'video_unet' and isinstance(value_B, dict) and isinstance(value_C, dict):
            for subkey in value_B.keys():
                if 'temporal' in subkey.lower():
                    temporal_keys.append(f"video_unet.{subkey}")
                if 'motion' in subkey.lower() or 'embedding' in subkey.lower():
                    motion_embedding_keys.append(f"video_unet.{subkey}")
                if 'text' in subkey.lower() or 'encoder' in subkey.lower():
                    text_encoder_keys.append(f"video_unet.{subkey}")

        # Compare if both are tensors
        if isinstance(value_B, torch.Tensor) and isinstance(value_C, torch.Tensor):
            comp = compare_tensors(value_B, value_C)
            stats_B = compute_tensor_statistics(value_B)
            stats_C = compute_tensor_statistics(value_C)

            comparison['tensor_comparisons'][key] = {
                'comparison': comp,
                'stats_B': stats_B,
                'stats_C': stats_C
            }

            if not comp['exact_match']:
                print(f"\n{key}: DIFFERENT")
                print(f"  Max abs diff: {comp['max_abs_diff']:.8f}")
                print(f"  Mean abs diff: {comp['mean_abs_diff']:.8f}")
                print(f"  Relative L2 diff: {comp['relative_l2_diff']:.8f}")
        elif isinstance(value_B, dict) and isinstance(value_C, dict):
            # Compare nested dicts (like video_unet state dict)
            comp = compare_nested_dicts(value_B, value_C)
            comparison['tensor_comparisons'][key] = {
                'both_tensors': False,
                'type': 'dict',
                'comparison': comp
            }
        else:
            # Non-tensor comparison
            try:
                equal = value_B == value_C
            except:
                equal = False

            comparison['tensor_comparisons'][key] = {
                'both_tensors': False,
                'type_B': str(type(value_B)),
                'type_C': str(type(value_C)),
                'equal': equal
            }

    # Temporal analysis
    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS")
    print("=" * 80)

    comparison['temporal_analysis'] = {
        'temporal_keys': temporal_keys,
        'temporal_key_count': len(temporal_keys)
    }

    print(f"Temporal keys found: {len(temporal_keys)}")
    for key in temporal_keys:
        if key in comparison['tensor_comparisons']:
            comp = comparison['tensor_comparisons'][key]
            if comp.get('both_tensors', False):
                print(f"  {key}: exact_match={comp['comparison']['exact_match']}, max_diff={comp['comparison']['max_abs_diff']:.8f}")

    # Motion embedding analysis
    print("\n" + "=" * 80)
    print("MOTION EMBEDDING ANALYSIS")
    print("=" * 80)

    comparison['motion_embedding_analysis'] = {
        'motion_embedding_keys': motion_embedding_keys,
        'motion_embedding_key_count': len(motion_embedding_keys)
    }

    print(f"Motion/embedding keys found: {len(motion_embedding_keys)}")
    for key in motion_embedding_keys:
        if key in comparison['tensor_comparisons']:
            comp = comparison['tensor_comparisons'][key]
            if comp.get('both_tensors', False):
                print(f"  {key}: exact_match={comp['comparison']['exact_match']}, max_diff={comp['comparison']['max_abs_diff']:.8f}")

    # Text encoder analysis
    print("\n" + "=" * 80)
    print("TEXT ENCODER ANALYSIS")
    print("=" * 80)

    comparison['text_encoder_analysis'] = {
        'text_encoder_keys': text_encoder_keys,
        'text_encoder_key_count': len(text_encoder_keys)
    }

    print(f"Text encoder keys found: {len(text_encoder_keys)}")
    for key in text_encoder_keys:
        if key in comparison['tensor_comparisons']:
            comp = comparison['tensor_comparisons'][key]
            if comp.get('both_tensors', False):
                print(f"  {key}: exact_match={comp['comparison']['exact_match']}, max_diff={comp['comparison']['max_abs_diff']:.8f}")

    # Optimizer analysis
    print("\n" + "=" * 80)
    print("OPTIMIZER ANALYSIS")
    print("=" * 80)

    comparison['optimizer_analysis'] = {
        'optimizer_keys': optimizer_keys,
        'optimizer_key_count': len(optimizer_keys)
    }

    print(f"Optimizer keys found: {len(optimizer_keys)}")
    for key in optimizer_keys:
        if key in comparison['tensor_comparisons']:
            comp = comparison['tensor_comparisons'][key]
            if comp.get('both_tensors', False):
                print(f"  {key}: exact_match={comp['comparison']['exact_match']}, max_diff={comp['comparison']['max_abs_diff']:.8f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Use video_unet nested dict comparison for the main summary
    if 'video_unet' in comparison['tensor_comparisons'] and comparison['tensor_comparisons']['video_unet'].get('type') == 'dict':
        video_unet_summary = comparison['tensor_comparisons']['video_unet']['comparison']['summary']
        total_tensor_comparisons = video_unet_summary['total_tensors']
        exact_matches = video_unet_summary['exact_matches']
        max_overall_diff = video_unet_summary['max_diff']
    else:
        # Fallback to direct tensor comparisons
        tensor_comparisons = [v for v in comparison['tensor_comparisons'].values() if v.get('both_tensors', False)]
        total_tensor_comparisons = len(tensor_comparisons)
        exact_matches = sum(1 for v in tensor_comparisons if v['comparison']['exact_match'])
        max_diffs = [v['comparison']['max_abs_diff'] for v in tensor_comparisons if not v['comparison']['exact_match']]
        max_overall_diff = max(max_diffs) if max_diffs else 0.0

    # Count different tensors in each category from video_unet
    temporal_tensors_different = 0
    motion_embedding_tensors_different = 0
    text_encoder_tensors_different = 0

    if 'video_unet' in comparison['tensor_comparisons'] and comparison['tensor_comparisons']['video_unet'].get('type') == 'dict':
        for subkey, subcomp in comparison['tensor_comparisons']['video_unet']['comparison']['tensor_comparisons'].items():
            if 'temporal' in subkey.lower() and not subcomp['exact_match']:
                temporal_tensors_different += 1
            if ('motion' in subkey.lower() or 'embedding' in subkey.lower()) and not subcomp['exact_match']:
                motion_embedding_tensors_different += 1
            if ('text' in subkey.lower() or 'encoder' in subkey.lower()) and not subcomp['exact_match']:
                text_encoder_tensors_different += 1

    comparison['summary'] = {
        'total_tensor_comparisons': total_tensor_comparisons,
        'exact_matches': exact_matches,
        'different_tensors': total_tensor_comparisons - exact_matches,
        'match_percentage': (exact_matches / total_tensor_comparisons * 100) if total_tensor_comparisons > 0 else 0,
        'max_overall_diff': max_overall_diff,
        'temporal_tensors_different': temporal_tensors_different,
        'motion_embedding_tensors_different': motion_embedding_tensors_different,
        'text_encoder_tensors_different': text_encoder_tensors_different
    }

    print(f"Total tensor comparisons: {total_tensor_comparisons}")
    print(f"Exact matches: {exact_matches} ({comparison['summary']['match_percentage']:.2f}%)")
    print(f"Different tensors: {comparison['summary']['different_tensors']}")
    print(f"Max overall difference: {max_overall_diff:.8f}")
    print(f"Temporal tensors different: {temporal_tensors_different}")
    print(f"Motion embedding tensors different: {motion_embedding_tensors_different}")
    print(f"Text encoder tensors different: {text_encoder_tensors_different}")

    # Save results
    output_dir = base_dir / "experiments_stage3_2_9B/phase2L_training_repair"
    output_file = output_dir / "STAGE3_2_9L_CHECKPOINT_COMPARISON.json"

    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_file}")

    return comparison

if __name__ == "__main__":
    complete_checkpoint_comparison()
