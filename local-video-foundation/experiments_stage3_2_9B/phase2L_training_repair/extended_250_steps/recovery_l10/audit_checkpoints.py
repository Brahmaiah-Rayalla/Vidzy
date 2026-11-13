"""
Phase 2 - Resume integrity audit: Check checkpoint payloads for optimizer state
"""

import torch
import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

checkpoints = {
    'L7_C1_step_100': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C1/checkpoint_step_000100.pt",
    'L7_C25_step_100': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000100.pt",
    'L8_C1_step_150': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/C1/checkpoint_step_000150.pt",
    'L8_C1_step_200': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/C1/checkpoint_step_000200.pt"
}

audit = {}

for name, path in checkpoints.items():
    result = {
        'path': str(path),
        'exists': path.exists(),
        'loadable': False,
        'keys': None,
        'has_optimizer_state': False,
        'has_training_step': False,
        'step_number': None,
        'has_config': False
    }
    
    if path.exists():
        try:
            checkpoint = torch.load(path, map_location='cpu')
            result['loadable'] = True
            result['keys'] = list(checkpoint.keys())
            
            # Check for optimizer state
            if 'optimizer_state_dict' in checkpoint:
                result['has_optimizer_state'] = True
                opt_state = checkpoint['optimizer_state_dict']
                result['optimizer_keys'] = list(opt_state.keys()) if isinstance(opt_state, dict) else type(opt_state).__name__
            
            # Check for training step
            if 'step' in checkpoint:
                result['has_training_step'] = True
                result['step_number'] = checkpoint['step']
            elif 'iteration' in checkpoint:
                result['has_training_step'] = True
                result['step_number'] = checkpoint['iteration']
            
            # Check for config
            if 'config' in checkpoint:
                result['has_config'] = True
                result['config_keys'] = list(checkpoint['config'].keys())
                
        except Exception as e:
            result['load_error'] = str(e)
    
    audit[name] = result

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/L10_CHECKPOINT_AND_OPTIMIZER_AUDIT.json"
with open(output_file, 'w') as f:
    json.dump(audit, f, indent=2)

print("Checkpoint and optimizer audit complete")
print(f"Saved to: {output_file}")

print("\nAudit Summary:")
for name, result in audit.items():
    print(f"\n{name}:")
    print(f"  Exists: {result['exists']}")
    print(f"  Loadable: {result['loadable']}")
    print(f"  Keys: {result['keys']}")
    print(f"  Has Optimizer State: {result['has_optimizer_state']}")
    print(f"  Has Training Step: {result['has_training_step']}")
    print(f"  Step Number: {result['step_number']}")
    print(f"  Has Config: {result['has_config']}")
