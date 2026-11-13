"""
Verify checkpoint integrity for Stage 3.2.9L.6.
"""

import hashlib
import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

checkpoint_C = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000100.pt"

with open(checkpoint_C, 'rb') as f:
    hash_after = hashlib.sha256(f.read()).hexdigest()

# Expected hash from Stage 3.2.9L.5
expected_hash = "c2dc17c93fbd855552fc473d4074236a2c0545bd187697a024ce49f801e7ea59"

integrity = {
    'checkpoint': str(checkpoint_C),
    'hash_after': hash_after,
    'expected_hash': expected_hash,
    'hash_match': hash_after == expected_hash,
    'status': 'PASS' if hash_after == expected_hash else 'FAIL'
}

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/fusion_motion_validation/STAGE3_2_9L6_CHECKPOINT_INTEGRITY.json"
with open(output_file, 'w') as f:
    json.dump(integrity, f, indent=2)

print(f"Checkpoint integrity: {integrity['status']}")
print(f"Saved to: {output_file}")
