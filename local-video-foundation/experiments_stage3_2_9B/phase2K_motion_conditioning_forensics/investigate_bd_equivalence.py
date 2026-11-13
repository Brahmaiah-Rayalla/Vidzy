"""
Stage 3.2.9K - B/D Equivalence Investigation

Investigates why Experiment B and Experiment D produce identical outputs.

Author: Devin
Date: 2026-11-03
"""

import torch
import json
import hashlib
from pathlib import Path
import sys

# Add paths
base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "experiments_stage3_2_7" / "video_foundation"))
sys.path.insert(0, str(base_dir / "experiments_stage3_2_9B" / "phase2J_corrected_motion_evaluation"))

def compute_tensor_hash(tensor):
    """Compute hash of a tensor."""
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

def load_checkpoint(checkpoint_path):
    """Load checkpoint and return state dict."""
    return torch.load(checkpoint_path, map_location='cpu')

def compare_checkpoints():
    """Compare Experiment B and Experiment C checkpoints."""
    checkpoint_B_path = base_dir / "experiments_stage3_2_9B/phase2C/checkpoints/checkpoint_phase2c_B.pt"
    checkpoint_C_path = base_dir / "experiments_stage3_2_9B/phase2C/checkpoints/checkpoint_phase2c_C.pt"

    print("Loading checkpoints...")
    checkpoint_B = load_checkpoint(checkpoint_B_path)
    checkpoint_C = load_checkpoint(checkpoint_C_path)

    comparison = {
        'checkpoint_B_path': str(checkpoint_B_path),
        'checkpoint_C_path': str(checkpoint_C_path),
        'parameter_comparison': {},
        'key_differences': [],
        'summary': {}
    }

    # Get all keys
    keys_B = set(checkpoint_B.keys())
    keys_C = set(checkpoint_C.keys())

    # Check key sets
    comparison['summary']['keys_in_B_not_C'] = list(keys_B - keys_C)
    comparison['summary']['keys_in_C_not_B'] = list(keys_C - keys_B)
    comparison['summary']['common_keys'] = list(keys_B & keys_C)

    print(f"\nKeys in B: {len(keys_B)}")
    print(f"Keys in C: {len(keys_C)}")
    print(f"Common keys: {len(comparison['summary']['common_keys'])}")

    # Compare common parameters
    max_diff = 0.0
    max_diff_key = None
    tensor_diff_max = 0.0
    tensor_diff_key = None

    for key in sorted(comparison['summary']['common_keys']):
        param_B = checkpoint_B[key]
        param_C = checkpoint_C[key]

        # Check if both are tensors
        if isinstance(param_B, torch.Tensor) and isinstance(param_C, torch.Tensor):
            if torch.equal(param_B, param_C):
                diff = 0.0
                equal = True
            else:
                diff = (param_B - param_C).abs().max().item()
                equal = False

            comparison['parameter_comparison'][key] = {
                'shape_B': list(param_B.shape),
                'shape_C': list(param_C.shape),
                'max_diff': diff,
                'equal': equal
            }

            if diff > max_diff:
                max_diff = diff
                max_diff_key = key

            # Track tensor differences separately
            if diff > tensor_diff_max:
                tensor_diff_max = diff
                tensor_diff_key = key
        else:
            # Handle non-tensor parameters (dicts, strings, etc.)
            # For non-tensors, use direct comparison
            try:
                if param_B == param_C:
                    equal = True
                    diff = 0.0
                else:
                    equal = False
                    diff = 1.0  # Use 1.0 to indicate difference for non-tensors
            except:
                # If comparison fails, mark as different
                equal = False
                diff = 1.0

            comparison['parameter_comparison'][key] = {
                'type_B': str(type(param_B)),
                'type_C': str(type(param_C)),
                'equal': equal,
                'value_B': str(param_B) if not isinstance(param_B, dict) else f"<dict with {len(param_B)} keys>",
                'value_C': str(param_C) if not isinstance(param_C, dict) else f"<dict with {len(param_C)} keys>"
            }

            if not equal and diff > max_diff:
                max_diff = diff
                max_diff_key = key

    comparison['summary']['max_difference'] = max_diff
    comparison['summary']['max_difference_key'] = max_diff_key
    comparison['summary']['max_tensor_difference'] = tensor_diff_max
    comparison['summary']['max_tensor_difference_key'] = tensor_diff_key

    print(f"\nMax parameter difference: {max_diff}")
    print(f"Max difference key: {max_diff_key}")
    print(f"Max tensor difference: {tensor_diff_max}")
    print(f"Max tensor difference key: {tensor_diff_key}")

    # Check temporal-specific parameters
    temporal_keys = [k for k in checkpoint_C.keys() if 'temporal' in k.lower()]
    comparison['summary']['temporal_keys_in_C'] = temporal_keys
    print(f"\nTemporal keys in C: {len(temporal_keys)}")

    # Check motion embeddings
    motion_embedding_keys = [k for k in checkpoint_C.keys() if 'motion' in k.lower() or 'embedding' in k.lower()]
    comparison['summary']['motion_embedding_keys'] = motion_embedding_keys
    print(f"Motion/embedding keys: {len(motion_embedding_keys)}")

    return comparison

def compare_inference_adapters():
    """Compare how B and D adapters load checkpoints."""
    from inference_adapters import create_adapter

    checkpoint_B_path = str(base_dir / "experiments_stage3_2_9B/phase2C/checkpoints/checkpoint_phase2c_B.pt")
    checkpoint_C_path = str(base_dir / "experiments_stage3_2_9B/phase2C/checkpoints/checkpoint_phase2c_C.pt")

    print("\n" + "=" * 80)
    print("COMPARING INFERENCE ADAPTERS")
    print("=" * 80)

    # Load B adapter
    print("\nLoading B adapter...")
    adapter_B = create_adapter(checkpoint_B_path, 'B')
    model_B = adapter_B.load()

    # Load C adapter (D uses C checkpoint)
    print("\nLoading C adapter...")
    adapter_C = create_adapter(checkpoint_C_path, 'C')
    model_C = adapter_C.load()

    # Load D adapter (C checkpoint with temporal disabled)
    print("\nLoading D adapter...")
    adapter_D = create_adapter(checkpoint_C_path, 'D')
    model_D = adapter_D.load()

    comparison = {
        'B_config': model_B.config if hasattr(model_B, 'config') else {},
        'C_config': model_C.config if hasattr(model_C, 'config') else {},
        'D_config': model_D.config if hasattr(model_D, 'config') else {},
        'parameter_hashes': {},
        'findings': []
    }

    # Compare B and D model parameters
    print("\nComparing B and D model parameters...")
    for name_B, param_B in model_B.named_parameters():
        # Find corresponding parameter in D
        param_D = dict(model_D.named_parameters()).get(name_B)

        if param_D is not None:
            if torch.equal(param_B, param_D):
                comparison['parameter_hashes'][f"B_vs_D_{name_B}"] = {
                    'equal': True,
                    'hash_B': compute_tensor_hash(param_B),
                    'hash_D': compute_tensor_hash(param_D)
                }
            else:
                diff = (param_B - param_D).abs().max().item()
                comparison['parameter_hashes'][f"B_vs_D_{name_B}"] = {
                    'equal': False,
                    'max_diff': diff,
                    'hash_B': compute_tensor_hash(param_B),
                    'hash_D': compute_tensor_hash(param_D)
                }
                comparison['findings'].append(f"B and D differ in {name_B} (max diff: {diff})")

    # Compare C and D model parameters
    print("\nComparing C and D model parameters...")
    for name_C, param_C in model_C.named_parameters():
        param_D = dict(model_D.named_parameters()).get(name_C)

        if param_D is not None:
            if torch.equal(param_C, param_D):
                comparison['parameter_hashes'][f"C_vs_D_{name_C}"] = {
                    'equal': True,
                    'hash_C': compute_tensor_hash(param_C),
                    'hash_D': compute_tensor_hash(param_D)
                }
            else:
                diff = (param_C - param_D).abs().max().item()
                comparison['parameter_hashes'][f"C_vs_D_{name_C}"] = {
                    'equal': False,
                    'max_diff': diff,
                    'hash_C': compute_tensor_hash(param_C),
                    'hash_D': compute_tensor_hash(param_D)
                }
                comparison['findings'].append(f"C and D differ in {name_C} (max diff: {diff})")

    comparison['summary'] = {
        'total_B_D_comparisons': len([k for k in comparison['parameter_hashes'].keys() if 'B_vs_D' in k]),
        'total_C_D_comparisons': len([k for k in comparison['parameter_hashes'].keys() if 'C_vs_D' in k]),
        'B_D_equal_count': sum(1 for k, v in comparison['parameter_hashes'].items() if 'B_vs_D' in k and v['equal']),
        'C_D_equal_count': sum(1 for k, v in comparison['parameter_hashes'].items() if 'C_vs_D' in k and v['equal']),
        'findings_count': len(comparison['findings'])
    }

    print(f"\nB vs D comparisons: {comparison['summary']['total_B_D_comparisons']}")
    print(f"B vs D equal: {comparison['summary']['B_D_equal_count']}")
    print(f"C vs D comparisons: {comparison['summary']['total_C_D_comparisons']}")
    print(f"C vs D equal: {comparison['summary']['C_D_equal_count']}")
    print(f"Total findings: {comparison['summary']['findings_count']}")

    if comparison['findings']:
        print("\nFindings:")
        for finding in comparison['findings']:
            print(f"  - {finding}")

    return comparison

def main():
    """Run B/D equivalence investigation."""
    print("=" * 80)
    print("STAGE 3.2.9K - B/D EQUIVALENCE INVESTIGATION")
    print("=" * 80)

    # Compare checkpoints
    print("\n" + "=" * 80)
    print("COMPARING CHECKPOINTS")
    print("=" * 80)
    checkpoint_comparison = compare_checkpoints()

    # Compare inference adapters
    adapter_comparison = compare_inference_adapters()

    # Combine results
    results = {
        'checkpoint_comparison': checkpoint_comparison,
        'adapter_comparison': adapter_comparison,
        'conclusion': {}
    }

    # Determine conclusion
    if checkpoint_comparison['summary']['max_tensor_difference'] == 0.0:
        results['conclusion']['checkpoints_identical'] = True
    else:
        results['conclusion']['checkpoints_identical'] = False

    if adapter_comparison['summary']['B_D_equal_count'] == adapter_comparison['summary']['total_B_D_comparisons']:
        results['conclusion']['B_D_models_identical'] = True
    else:
        results['conclusion']['B_D_models_identical'] = False

    if adapter_comparison['summary']['C_D_equal_count'] == adapter_comparison['summary']['total_C_D_comparisons']:
        results['conclusion']['C_D_models_identical'] = True
    else:
        results['conclusion']['C_D_models_identical'] = False

    # Save results
    output_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2K_motion_conditioning_forensics")

    with open(output_dir / "STAGE3_2_9K_B_D_EQUIVALENCE_AUDIT.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_dir}")

    return results

if __name__ == "__main__":
    main()
