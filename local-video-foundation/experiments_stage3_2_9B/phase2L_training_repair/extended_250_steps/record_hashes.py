"""
Record SHA256 hashes for Stage 3.2.9L.8.
"""

import hashlib
import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

artifacts = {
    "stage2_spatial": base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_spatial.pt",
    "stage2_motion_expanded": base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt",
    "L7_C1_step_100": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C1/checkpoint_step_000100.pt",
    "L7_C25_step_100": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000100.pt",
    "tokenizer": base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json",
    "video_unet_fixed": base_dir / "experiments_stage3_2_7/video_foundation/models/video_unet_fixed.py"
}

hashes = {}
for name, path in artifacts.items():
    if path.exists():
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        hashes[name] = sha256.hexdigest()

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/ARTIFACT_HASHES.json"
with open(output_file, 'w') as f:
    json.dump(hashes, f, indent=2)

print(f"Artifact hashes saved to: {output_file}")
