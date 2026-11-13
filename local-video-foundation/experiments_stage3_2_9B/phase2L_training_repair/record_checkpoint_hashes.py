"""
Stage 3.2.9L - Checkpoint Protection

Records SHA256 hashes of all existing checkpoints before any modifications.

Author: Devin
Date: 2026-11-04
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime

def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def record_checkpoint_hashes():
    """Record hashes of all critical checkpoints."""
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

    # Critical checkpoints to protect
    checkpoints = {
        'stage2_1_checkpoint': base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt",
        'experiment_B_checkpoint': base_dir / "experiments_stage3_2_9B/phase2C/checkpoints/checkpoint_phase2c_B.pt",
        'experiment_C_checkpoint': base_dir / "experiments_stage3_2_9B/phase2C/checkpoints/checkpoint_phase2c_C.pt",
        'tokenizer': base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json",
    }

    # Add all Stage 3.2.9A-K reports
    report_dirs = [
        base_dir / "experiments_stage3_2_9B/phase2K_motion_conditioning_forensics",
        base_dir / "experiments_stage3_2_9B/phase2J_corrected_motion_evaluation",
        base_dir / "experiments_stage3_2_9B/phase2I_text_conditioning",
        base_dir / "experiments_stage3_2_9B/phase2H_temporal_instrumentation",
        base_dir / "experiments_stage3_2_9B/phase2G_motion_evaluation",
        base_dir / "experiments_stage3_2_9B/phase2F_video_generation",
        base_dir / "experiments_stage3_2_9B/phase2E_inference_compatibility",
        base_dir / "experiments_stage3_2_9B/phase2C",
    ]

    print("=" * 80)
    print("STAGE 3.2.9L - CHECKPOINT PROTECTION")
    print("=" * 80)

    hash_record = {
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Record SHA256 hashes before any modifications',
        'checkpoints': {},
        'reports': {}
    }

    # Record checkpoint hashes
    print("\nCheckpoint hashes:")
    for name, path in checkpoints.items():
        if path.exists():
            file_hash = compute_file_hash(path)
            file_size = path.stat().st_size
            hash_record['checkpoints'][name] = {
                'path': str(path),
                'sha256': file_hash,
                'size_bytes': file_size,
                'exists': True
            }
            print(f"  {name}: {file_hash[:16]}... ({file_size} bytes)")
        else:
            hash_record['checkpoints'][name] = {
                'path': str(path),
                'sha256': None,
                'size_bytes': 0,
                'exists': False
            }
            print(f"  {name}: NOT FOUND")

    # Record report file hashes
    print("\nReport file hashes:")
    for report_dir in report_dirs:
        if report_dir.exists():
            for report_file in report_dir.glob("*.md"):
                file_hash = compute_file_hash(report_file)
                file_size = report_file.stat().st_size
                rel_path = report_file.relative_to(base_dir)
                hash_record['reports'][str(rel_path)] = {
                    'sha256': file_hash,
                    'size_bytes': file_size
                }
                print(f"  {rel_path}: {file_hash[:16]}...")

            for report_file in report_dir.glob("*.json"):
                file_hash = compute_file_hash(report_file)
                file_size = report_file.stat().st_size
                rel_path = report_file.relative_to(base_dir)
                hash_record['reports'][str(rel_path)] = {
                    'sha256': file_hash,
                    'size_bytes': file_size
                }
                print(f"  {rel_path}: {file_hash[:16]}...")

    # Save hash record
    output_dir = base_dir / "experiments_stage3_2_9B/phase2L_training_repair"
    output_file = output_dir / "STAGE3_2_9L_CHECKPOINT_HASHES_BEFORE.json"

    with open(output_file, 'w') as f:
        json.dump(hash_record, f, indent=2)

    print("\n" + "=" * 80)
    print("PROTECTION COMPLETE")
    print("=" * 80)
    print(f"Hash record saved to {output_file}")

    return hash_record

if __name__ == "__main__":
    record_checkpoint_hashes()
