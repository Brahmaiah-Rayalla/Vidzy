"""
Record SHA256 hashes of all existing artifacts before pilot training.
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

def get_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def record_hashes():
    """Record hashes of all critical artifacts."""
    artifacts = {
        'timestamp': datetime.now().isoformat(),
        'artifacts': {}
    }

    # Stage 2.1 checkpoint
    stage2_checkpoint = base_dir / "checkpoints_conditional/checkpoint_final.pt"
    artifacts['artifacts']['stage2_checkpoint'] = {
        'path': str(stage2_checkpoint),
        'hash': get_file_hash(stage2_checkpoint)
    }

    # Stage 3.2.7 checkpoints
    stage3_spatial = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_spatial.pt"
    stage3_motion = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    tokenizer = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"

    artifacts['artifacts']['stage3_spatial_checkpoint'] = {
        'path': str(stage3_spatial),
        'hash': get_file_hash(stage3_spatial)
    }
    artifacts['artifacts']['stage3_motion_checkpoint'] = {
        'path': str(stage3_motion),
        'hash': get_file_hash(stage3_motion)
    }
    artifacts['artifacts']['tokenizer'] = {
        'path': str(tokenizer),
        'hash': get_file_hash(tokenizer)
    }

    # Stage 3.2.9B experiment checkpoints
    exp_b_checkpoint = base_dir / "experiments_stage3_2_9B/checkpoints/checkpoint_experiment_B.pt"
    exp_c_checkpoint = base_dir / "experiments_stage3_2_9B/checkpoints/checkpoint_experiment_C.pt"

    artifacts['artifacts']['experiment_B_checkpoint'] = {
        'path': str(exp_b_checkpoint),
        'hash': get_file_hash(exp_b_checkpoint)
    }
    artifacts['artifacts']['experiment_C_checkpoint'] = {
        'path': str(exp_c_checkpoint),
        'hash': get_file_hash(exp_c_checkpoint)
    }

    # Model files
    text_encoder_split = base_dir / "experiments_stage3_2_7/video_foundation/models/text_encoder_split.py"
    video_unet_fixed = base_dir / "experiments_stage3_2_7/video_foundation/models/video_unet_fixed.py"
    conditional_unet = base_dir / "experiments_stage3_2_7/video_foundation/models/conditional_unet.py"

    artifacts['artifacts']['text_encoder_split'] = {
        'path': str(text_encoder_split),
        'hash': get_file_hash(text_encoder_split)
    }
    artifacts['artifacts']['video_unet_fixed'] = {
        'path': str(video_unet_fixed),
        'hash': get_file_hash(video_unet_fixed)
    }
    artifacts['artifacts']['conditional_unet'] = {
        'path': str(conditional_unet),
        'hash': get_file_hash(conditional_unet)
    }

    # Training script
    training_script = base_dir / "experiments_stage3_2_9B/stage3_2_9b_experiment.py"
    artifacts['artifacts']['training_script'] = {
        'path': str(training_script),
        'hash': get_file_hash(training_script)
    }

    # Save hashes
    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/STAGE3_2_9L2_ARTIFACT_HASHES_PRE_PILOT.json"
    with open(output_file, 'w') as f:
        json.dump(artifacts, f, indent=2)

    print("=" * 80)
    print("ARTIFACT HASHES RECORDED")
    print("=" * 80)
    print(f"\nSaved to: {output_file}\n")

    for name, info in artifacts['artifacts'].items():
        status = "OK" if info['hash'] != "FILE_NOT_FOUND" else "MISSING"
        print(f"[{status}] {name}: {info['hash'][:16]}...")

    return artifacts

if __name__ == "__main__":
    record_hashes()
