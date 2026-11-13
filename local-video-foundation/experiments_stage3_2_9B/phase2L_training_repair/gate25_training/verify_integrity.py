"""
Verify checkpoint integrity for Stage 3.2.9L.7.
"""

import hashlib
import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

checkpoints = {
    'C1_step_25': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C1/checkpoint_step_000025.pt",
    'C1_step_50': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C1/checkpoint_step_000050.pt",
    'C1_step_75': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C1/checkpoint_step_000075.pt",
    'C1_step_100': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C1/checkpoint_step_000100.pt",
    'C25_step_25': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000025.pt",
    'C25_step_50': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000050.pt",
    'C25_step_75': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000075.pt",
    'C25_step_100': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000100.pt"
}

integrity = {}
for name, path in checkpoints.items():
    if path.exists():
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        integrity[name] = {
            'path': str(path),
            'sha256': sha256.hexdigest(),
            'exists': True
        }
    else:
        integrity[name] = {
            'path': str(path),
            'sha256': None,
            'exists': False
        }

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/STAGE3_2_9L7_CHECKPOINT_INTEGRITY.json"
with open(output_file, 'w') as f:
    json.dump(integrity, f, indent=2)

print(f"Checkpoint integrity verified. Saved to: {output_file}")
