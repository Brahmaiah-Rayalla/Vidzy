"""
C1 vs C25 comparison based on available data
"""

import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

# Load gate progression
with open(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/L9_GATE_PROGRESSION.json", 'r') as f:
    gate_data = json.load(f)

# Load loss comparison
with open(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/L9_VALIDATION_LOSS_COMPARISON.json", 'r') as f:
    loss_data = json.load(f)

comparison = {
    'summary': {
        'available_data': 'C1 steps 100, 150, 200; C25 step 100 only',
        'L8_status': 'INCOMPLETE - C25 training did not complete',
        'limitation': 'Cannot assess C25 advantage beyond step 100'
    },
    'loss_comparison': {
        'step_100': {
            'C1': loss_data['C1']['step_100_loss'],
            'C25': loss_data['C25']['step_100_loss'],
            'improvement_percent': loss_data['comparison_at_step_100']['improvement_percent'],
            'conclusion': 'C25 shows 17.62% lower loss at step 100'
        }
    },
    'gate_convergence': {
        'C1': {
            'step_100': gate_data['C1']['step_100']['activated'],
            'step_150': gate_data['C1']['step_150']['activated'],
            'step_200': gate_data['C1']['step_200']['activated'],
            'change_100_to_200': gate_data['C1']['change_100_to_200'],
            'conclusion': 'C1 gates show minimal convergence (1.84% bottleneck, 3.05% low-res over 100 steps)'
        },
        'C25': {
            'step_100': gate_data['C25']['step_100']['activated'],
            'step_150': 'NOT_AVAILABLE',
            'step_200': 'NOT_AVAILABLE',
            'conclusion': 'C25 gate convergence cannot be assessed (L.8 incomplete)'
        }
    },
    'key_findings': [
        'C25 (25% init) achieved 17.62% lower loss than C1 (1% init) at step 100',
        'C1 gates show minimal convergence over 100 additional steps (1-3% change)',
        'C25 gate convergence cannot be assessed due to incomplete L.8 training',
        'Motion evaluation not available (requires DDIM infrastructure)',
        'L.8 training: C1 reached step 200, C25 training did not complete'
    ],
    'limitations': [
        'L.8 training incomplete - C25 checkpoints for steps 150, 200, 250 not available',
        'Cannot assess whether C25 advantage persists beyond step 100',
        'Cannot assess C25 gate convergence over extended training',
        'Motion evaluation deferred (DDIM infrastructure not available)',
        'Validation loss comparison limited to step 100'
    ]
}

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/L9_C1_C25_COMPARISON.json"
with open(output_file, 'w') as f:
    json.dump(comparison, f, indent=2)

print("C1 vs C25 comparison complete")
print(f"Saved to: {output_file}")

print("\nKey Findings:")
for finding in comparison['key_findings']:
    print(f"  - {finding}")
