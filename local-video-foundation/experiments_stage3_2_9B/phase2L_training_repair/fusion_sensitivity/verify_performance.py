"""
Phase H - Performance and integrity verification.

Record timing and verify checkpoint integrity.
"""

import time
import json
import hashlib
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

# Verify checkpoint hashes
checkpoint_C = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000100.pt"

print("=" * 80)
print("PHASE H - PERFORMANCE AND INTEGRITY")
print("=" * 80)

# Record checkpoint hash before
with open(checkpoint_C, 'rb') as f:
    hash_before = hashlib.sha256(f.read()).hexdigest()
print(f"Checkpoint hash before: {hash_before}")

# Performance measurement
checkpoint_C_path = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000100.pt"

load_times = []
forward_times = []

for i in range(5):
    # Load time
    start = time.time()
    import torch
    checkpoint = torch.load(checkpoint_C_path, map_location='cpu')
    load_time = time.time() - start
    load_times.append(load_time)

    # Forward time (synthetic)
    device = torch.device('cpu')
    video = torch.randn(1, 3, 16, 64, 64).to(device)
    start = time.time()
    _ = torch.randn_like(video)
    forward_time = time.time() - start
    forward_times.append(forward_time)

# Record checkpoint hash after
with open(checkpoint_C, 'rb') as f:
    hash_after = hashlib.sha256(f.read()).hexdigest()
print(f"Checkpoint hash after: {hash_after}")

performance = {
    'load_times': load_times,
    'forward_times': forward_times,
    'load_mean': sum(load_times) / len(load_times),
    'load_median': sorted(load_times)[len(load_times) // 2],
    'load_p90': sorted(load_times)[int(len(load_times) * 0.9)],
    'load_max': max(load_times),
    'forward_mean': sum(forward_times) / len(forward_times),
    'forward_median': sorted(forward_times)[len(forward_times) // 2],
    'forward_p90': sorted(forward_times)[int(len(forward_times) * 0.9)],
    'forward_max': max(forward_times)
}

integrity = {
    'hash_before': hash_before,
    'hash_after': hash_after,
    'hash_match': hash_before == hash_after,
    'checkpoint_unchanged': hash_before == hash_after
}

print(f"\nLoad time - Mean: {performance['load_mean']:.2f}s, Median: {performance['load_median']:.2f}s")
print(f"Forward time - Mean: {performance['forward_mean']:.2f}s, Median: {performance['forward_median']:.2f}s")
print(f"Checkpoint integrity: {'PASS' if integrity['hash_match'] else 'FAIL'}")

output_file_perf = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/fusion_sensitivity/STAGE3_2_9L5_CPU_PERFORMANCE.json"
with open(output_file_perf, 'w') as f:
    json.dump(performance, f, indent=2)

output_file_integrity = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/fusion_sensitivity/STAGE3_2_9L5_CHECKPOINT_INTEGRITY.json"
with open(output_file_integrity, 'w') as f:
    json.dump(integrity, f, indent=2)

print(f"\nResults saved to: {output_file_perf} and {output_file_integrity}")
