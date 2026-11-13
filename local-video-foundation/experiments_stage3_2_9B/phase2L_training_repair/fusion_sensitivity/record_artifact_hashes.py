"""
Record SHA256 hashes of critical artifacts for Stage 3.2.9L.5.
"""

import hashlib
import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

artifacts = {
    "stage2_spatial_checkpoint": base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_spatial.pt",
    "stage2_motion_expanded_checkpoint": base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt",
    "experiment_B_step_25": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000025.pt",
    "experiment_B_step_50": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000050.pt",
    "experiment_B_step_75": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000075.pt",
    "experiment_B_step_100": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000100.pt",
    "experiment_C_step_25": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000025.pt",
    "experiment_C_step_50": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000050.pt",
    "experiment_C_step_75": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000075.pt",
    "experiment_C_step_100": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000100.pt",
    "text_encoder_split": base_dir / "experiments_stage3_2_7/video_foundation/models/text_encoder_split.py",
    "video_unet_fixed": base_dir / "experiments_stage3_2_7/video_foundation/models/video_unet_fixed.py",
    "training_script": base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/run_100_step_training.py",
    "tokenizer": base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"
}

hashes = {}

for name, path in artifacts.items():
    if path.exists():
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        hashes[name] = {
            "path": str(path),
            "sha256": sha256.hexdigest(),
            "exists": True
        }
    else:
        hashes[name] = {
            "path": str(path),
            "sha256": None,
            "exists": False
        }

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/fusion_sensitivity/ARTIFACT_HASHES.json"
with open(output_file, 'w') as f:
    json.dump(hashes, f, indent=2)

print("Artifact hashes recorded:")
for name, info in hashes.items():
    if info['exists']:
        print(f"  {name}: {info['sha256']}")
    else:
        print(f"  {name}: NOT FOUND")

print(f"\nSaved to: {output_file}")
