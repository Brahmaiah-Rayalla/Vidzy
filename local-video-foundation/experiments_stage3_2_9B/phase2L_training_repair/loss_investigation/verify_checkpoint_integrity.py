"""
Checkpoint integrity verification.

This script verifies that checkpoints are valid and can be loaded correctly.
"""

import torch
import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

print("=" * 80)
print("CHECKPOINT INTEGRITY VERIFICATION")
print("=" * 80)

# Define checkpoints
checkpoints = {
    'B_step_25': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000025.pt",
    'B_step_50': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000050.pt",
    'B_step_75': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000075.pt",
    'B_step_100': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000100.pt",
    'C_step_25': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000025.pt",
    'C_step_50': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000050.pt",
    'C_step_75': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000075.pt",
    'C_step_100': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000100.pt"
}

verification_results = {}

for name, path in checkpoints.items():
    print(f"\nVerifying {name}: {path}")

    if not path.exists():
        print(f"  ERROR: File not found")
        verification_results[name] = {'status': 'FILE_NOT_FOUND'}
        continue

    try:
        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu')

        # Check required keys
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'config', 'experiment', 'step']
        missing_keys = [key for key in required_keys if key not in checkpoint]

        if missing_keys:
            print(f"  ERROR: Missing keys: {missing_keys}")
            verification_results[name] = {'status': 'MISSING_KEYS', 'missing_keys': missing_keys}
            continue

        # Count parameters
        num_params = len(checkpoint['model_state_dict'])
        print(f"  Parameters: {num_params}")
        print(f"  Experiment: {checkpoint['experiment']}")
        print(f"  Step: {checkpoint['step']}")

        # Check parameter types
        param_types = {}
        for key, value in checkpoint['model_state_dict'].items():
            if isinstance(value, torch.Tensor):
                param_types[key] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype)
                }

        verification_results[name] = {
            'status': 'OK',
            'num_params': num_params,
            'experiment': checkpoint['experiment'],
            'step': checkpoint['step'],
            'param_types': param_types
        }

        print(f"  Status: OK")

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        verification_results[name] = {'status': 'LOAD_ERROR', 'error': str(e)}

# Compare checkpoints
print("\n" + "=" * 80)
print("CHECKPOINT COMPARISON")
print("=" * 80)

# Compare B checkpoints
print("\nExperiment B checkpoint comparison:")
B_checkpoints = [f'B_step_{step}' for step in [25, 50, 75, 100]]
for i in range(len(B_checkpoints) - 1):
    name1 = B_checkpoints[i]
    name2 = B_checkpoints[i + 1]

    if verification_results[name1]['status'] == 'OK' and verification_results[name2]['status'] == 'OK':
        num_params1 = verification_results[name1]['num_params']
        num_params2 = verification_results[name2]['num_params']
        print(f"  {name1} -> {name2}: {num_params1} -> {num_params2} params ({'same' if num_params1 == num_params2 else 'different'})")

# Compare C checkpoints
print("\nExperiment C checkpoint comparison:")
C_checkpoints = [f'C_step_{step}' for step in [25, 50, 75, 100]]
for i in range(len(C_checkpoints) - 1):
    name1 = C_checkpoints[i]
    name2 = C_checkpoints[i + 1]

    if verification_results[name1]['status'] == 'OK' and verification_results[name2]['status'] == 'OK':
        num_params1 = verification_results[name1]['num_params']
        num_params2 = verification_results[name2]['num_params']
        print(f"  {name1} -> {name2}: {num_params1} -> {num_params2} params ({'same' if num_params1 == num_params2 else 'different'})")

# Compare B and C at step 100
print("\nExperiment B vs C at step 100:")
if verification_results['B_step_100']['status'] == 'OK' and verification_results['C_step_100']['status'] == 'OK':
    num_params_B = verification_results['B_step_100']['num_params']
    num_params_C = verification_results['C_step_100']['num_params']
    print(f"  B: {num_params_B} params")
    print(f"  C: {num_params_C} params")
    print(f"  Status: {'same' if num_params_B == num_params_C else 'different'}")

# Save results
output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/loss_investigation/CHECKPOINT_INTEGRITY_VERIFICATION.json"
with open(output_file, 'w') as f:
    json.dump(verification_results, f, indent=2)

print(f"\nResults saved to: {output_file}")

print("\n--- CONCLUSION ---")
all_ok = all(result['status'] == 'OK' for result in verification_results.values())
if all_ok:
    print("All checkpoints verified successfully.")
else:
    failed = [name for name, result in verification_results.items() if result['status'] != 'OK']
    print(f"Failed checkpoints: {failed}")
