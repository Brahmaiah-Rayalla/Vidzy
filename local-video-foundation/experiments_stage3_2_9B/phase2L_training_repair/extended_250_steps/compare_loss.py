"""
Validation loss comparison using L.7 results
"""

import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

# Load L.7 training results
with open(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/STAGE3_2_9L7_TRAINING_RESULTS.json", 'r') as f:
    l7_data = json.load(f)

# Extract loss data
c1_step_100_loss = l7_data['pilot_C1']['steps'][-1]['loss']
c25_step_100_loss = l7_data['pilot_C25']['steps'][-1]['loss']

loss_comparison = {
    'C1': {
        'step_100_loss': c1_step_100_loss,
        'step_150_loss': 'NOT_AVAILABLE',
        'step_200_loss': 'NOT_AVAILABLE',
        'step_250_loss': 'NOT_AVAILABLE'
    },
    'C25': {
        'step_100_loss': c25_step_100_loss,
        'step_150_loss': 'NOT_AVAILABLE',
        'step_200_loss': 'NOT_AVAILABLE',
        'step_250_loss': 'NOT_AVAILABLE'
    },
    'comparison_at_step_100': {
        'C1_loss': c1_step_100_loss,
        'C25_loss': c25_step_100_loss,
        'improvement_percent': ((c1_step_100_loss - c25_step_100_loss) / c1_step_100_loss) * 100
    },
    'note': 'L.8 training did not complete for C25. Validation loss comparison limited to step 100 from L.7.'
}

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/L9_VALIDATION_LOSS_COMPARISON.json"
with open(output_file, 'w') as f:
    json.dump(loss_comparison, f, indent=2)

print("Validation loss comparison complete")
print(f"Saved to: {output_file}")

print("\nLoss Comparison (Step 100):")
print(f"  C1: {c1_step_100_loss:.6f}")
print(f"  C25: {c25_step_100_loss:.6f}")
print(f"  Improvement: {loss_comparison['comparison_at_step_100']['improvement_percent']:.2f}%")
