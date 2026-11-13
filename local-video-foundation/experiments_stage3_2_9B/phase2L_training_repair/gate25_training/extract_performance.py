"""
Extract CPU performance from training results.
"""

import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
results_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/STAGE3_2_9L7_TRAINING_RESULTS.json"

with open(results_file, 'r') as f:
    data = json.load(f)

c1_times = [step['step_time'] for step in data['pilot_C1']['steps']]
c25_times = [step['step_time'] for step in data['pilot_C25']['steps']]

performance = {
    'c1': {
        'mean_step_time': sum(c1_times) / len(c1_times),
        'median_step_time': sorted(c1_times)[len(c1_times) // 2],
        'min_step_time': min(c1_times),
        'max_step_time': max(c1_times),
        'total_time': sum(c1_times)
    },
    'c25': {
        'mean_step_time': sum(c25_times) / len(c25_times),
        'median_step_time': sorted(c25_times)[len(c25_times) // 2],
        'min_step_time': min(c25_times),
        'max_step_time': max(c25_times),
        'total_time': sum(c25_times)
    }
}

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/STAGE3_2_9L7_CPU_PERFORMANCE.json"
with open(output_file, 'w') as f:
    json.dump(performance, f, indent=2)

print(f"CPU performance extracted. Saved to: {output_file}")
