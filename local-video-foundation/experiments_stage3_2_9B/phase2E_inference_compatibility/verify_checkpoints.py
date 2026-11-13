"""
Stage 3.2.9D - Checkpoint Verification

Verify SHA256 hashes of all checkpoints for inference compatibility stage.

Author: Devin
Date: 2026-11-02
"""

import hashlib
import json
from pathlib import Path


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    base_dir = Path(__file__).parent.parent.parent

    checkpoints = {
        "Experiment_A": {
            "path": str(base_dir / "checkpoints_conditional" / "checkpoint_final.pt"),
            "expected": "a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c"
        },
        "Experiment_B": {
            "path": str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt"),
            "expected": "27eef3c7afec90a0cfaf32384f6535d271b9369ba7baaba51f17669fd3b0ff14"
        },
        "Experiment_C": {
            "path": str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt"),
            "expected": "cc71317860e152b818413c67a7e0304079a86e6e243c6e5e5d9559cec21499f5"
        }
    }

    print("=" * 80)
    print("  STAGE 3.2.9D - CHECKPOINT VERIFICATION")
    print("=" * 80)

    results = {}
    all_match = True

    for name, info in checkpoints.items():
        print(f"\n{name}:")
        print(f"  Path: {info['path']}")

        if not Path(info['path']).exists():
            print(f"  [X] FILE NOT FOUND")
            results[name] = {"exists": False, "hash": None, "expected": info['expected'], "match": False}
            all_match = False
            continue

        actual_hash = compute_sha256(info['path'])
        print(f"  Expected: {info['expected']}")
        print(f"  Actual:   {actual_hash}")

        match = actual_hash == info['expected']
        if match:
            print(f"  [OK] HASH MATCH")
        else:
            print(f"  [X] HASH MISMATCH")
            all_match = False

        results[name] = {"exists": True, "hash": actual_hash, "expected": info['expected'], "match": match}

    print("\n" + "=" * 80)
    if all_match:
        print("  [OK] ALL CHECKPOINT HASHES VERIFIED")
    else:
        print("  [X] CHECKPOINT HASH VERIFICATION FAILED")
    print("=" * 80)

    output_path = Path(__file__).parent / "checkpoint_verification.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved verification results to {output_path}")
    return all_match


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
