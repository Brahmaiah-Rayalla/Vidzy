"""
Generate artifact hashes for L.10
"""

import hashlib
import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

artifacts = {
    'stage2_spatial': base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_spatial.pt",
    'stage2_motion_expanded': base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt",
    'recovery_C1_250': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C1/checkpoint_step_000250.pt",
    'recovery_C25_250': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C25/checkpoint_step_000250.pt",
    'recovery_C25_150': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C25/checkpoint_step_000150.pt",
    'recovery_C25_200': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C25/checkpoint_step_000200.pt"
}

hashes = {}
for name, path in artifacts.items():
    if path.exists():
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        hashes[name] = sha256.hexdigest()
    else:
        hashes[name] = None

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/L10_ARTIFACT_HASHES.json"
with open(output_file, 'w') as f:
    json.dump(hashes, f, indent=2)

print(f"Artifact hashes saved to: {output_file}")
