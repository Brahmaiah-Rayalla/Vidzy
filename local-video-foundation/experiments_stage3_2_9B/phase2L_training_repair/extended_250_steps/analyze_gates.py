"""
Analyze gate progression from available checkpoints
"""

import torch
import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

# Load L.7 C1 step 100
c1_100 = torch.load(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C1/checkpoint_step_000100.pt", map_location='cpu')
c1_100_gates = {'bottleneck': c1_100['model_state_dict']['fusion_modules.bottleneck.gate'].item(),
                 'low_res': c1_100['model_state_dict']['fusion_modules.low_res.gate'].item()}

# Load L.7 C25 step 100
c25_100 = torch.load(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000100.pt", map_location='cpu')
c25_100_gates = {'bottleneck': c25_100['model_state_dict']['fusion_modules.bottleneck.gate'].item(),
                  'low_res': c25_100['model_state_dict']['fusion_modules.low_res.gate'].item()}

# Load L.8 C1 step 150
c1_150 = torch.load(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/C1/checkpoint_step_000150.pt", map_location='cpu')
c1_150_gates = {'bottleneck': c1_150['model_state_dict']['fusion_modules.bottleneck.gate'].item(),
                 'low_res': c1_150['model_state_dict']['fusion_modules.low_res.gate'].item()}

# Load L.8 C1 step 200
c1_200 = torch.load(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/C1/checkpoint_step_000200.pt", map_location='cpu')
c1_200_gates = {'bottleneck': c1_200['model_state_dict']['fusion_modules.bottleneck.gate'].item(),
                 'low_res': c1_200['model_state_dict']['fusion_modules.low_res.gate'].item()}

# Calculate changes
def calc_change(initial, final):
    return ((final - initial) / initial) * 100 if initial != 0 else 0

gate_progression = {
    'C1': {
        'step_100': c1_100_gates,
        'step_150': c1_150_gates,
        'step_200': c1_200_gates,
        'change_100_to_150': {
            'bottleneck_percent': calc_change(c1_100_gates['bottleneck'], c1_150_gates['bottleneck']),
            'low_res_percent': calc_change(c1_100_gates['low_res'], c1_150_gates['low_res'])
        },
        'change_100_to_200': {
            'bottleneck_percent': calc_change(c1_100_gates['bottleneck'], c1_200_gates['bottleneck']),
            'low_res_percent': calc_change(c1_100_gates['low_res'], c1_200_gates['low_res'])
        }
    },
    'C25': {
        'step_100': c25_100_gates,
        'step_150': 'NOT_AVAILABLE',
        'step_200': 'NOT_AVAILABLE',
        'note': 'C25 checkpoints for steps 150 and 200 were not generated in L.8'
    }
}

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/L9_GATE_PROGRESSION.json"
with open(output_file, 'w') as f:
    json.dump(gate_progression, f, indent=2)

print("Gate progression analysis complete")
print(f"Saved to: {output_file}")

print("\nC1 Gate Progression:")
print(f"  Step 100: bottleneck={c1_100_gates['bottleneck']:.6f}, low_res={c1_100_gates['low_res']:.6f}")
print(f"  Step 150: bottleneck={c1_150_gates['bottleneck']:.6f}, low_res={c1_150_gates['low_res']:.6f}")
print(f"  Step 200: bottleneck={c1_200_gates['bottleneck']:.6f}, low_res={c1_200_gates['low_res']:.6f}")
print(f"  Change 100->150: bottleneck={gate_progression['C1']['change_100_to_150']['bottleneck_percent']:.2f}%, low_res={gate_progression['C1']['change_100_to_150']['low_res_percent']:.2f}%")
print(f"  Change 100->200: bottleneck={gate_progression['C1']['change_100_to_200']['bottleneck_percent']:.2f}%, low_res={gate_progression['C1']['change_100_to_200']['low_res_percent']:.2f}%")

print("\nC25 Gate Values (step 100 only):")
print(f"  Step 100: bottleneck={c25_100_gates['bottleneck']:.6f}, low_res={c25_100_gates['low_res']:.6f}")
