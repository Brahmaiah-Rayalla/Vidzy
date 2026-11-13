"""
Verify checkpoint integrity for Stage 3.2.9L.9
"""

import torch
import hashlib
import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

checkpoints = {
    'L7_C1_step_100': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C1/checkpoint_step_000100.pt",
    'L7_C25_step_100': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000100.pt",
    'L8_C1_step_150': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/C1/checkpoint_step_000150.pt",
    'L8_C1_step_200': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/C1/checkpoint_step_000200.pt",
    'stage2_spatial': base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_spatial.pt"
}

integrity = {}
for name, path in checkpoints.items():
    result = {'path': str(path), 'exists': False, 'sha256': None, 'loadable': False, 'keys': None}
    
    if path.exists():
        result['exists'] = True
        
        # Calculate hash
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        result['sha256'] = sha256.hexdigest()
        
        # Try to load
        try:
            checkpoint = torch.load(path, map_location='cpu')
            result['loadable'] = True
            result['keys'] = list(checkpoint.keys())
            
            # Count parameters
            if 'model_state_dict' in checkpoint:
                total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
                result['total_parameters'] = total_params
        except Exception as e:
            result['load_error'] = str(e)
    
    integrity[name] = result

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/L9_CHECKPOINT_INTEGRITY.json"
with open(output_file, 'w') as f:
    json.dump(integrity, f, indent=2)

print("Checkpoint integrity verification complete")
print(f"Saved to: {output_file}")

# Print summary
print("\nSummary:")
for name, result in integrity.items():
    status = "OK" if result['exists'] and result['loadable'] else "MISSING/ERROR"
    print(f"  {name}: {status}")
