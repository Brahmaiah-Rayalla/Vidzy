"""
Analyze gate progression with activated sigmoid values
"""

import torch
import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

def sigmoid(x):
    return 1 / (1 + torch.exp(-torch.tensor(x)))

# Load L.7 C1 step 100
c1_100 = torch.load(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C1/checkpoint_step_000100.pt", map_location='cpu')
c1_100_raw = {'bottleneck': c1_100['model_state_dict']['fusion_modules.bottleneck.gate'].item(),
               'low_res': c1_100['model_state_dict']['fusion_modules.low_res.gate'].item()}
c1_100_activated = {'bottleneck': sigmoid(c1_100_raw['bottleneck']).item(),
                     'low_res': sigmoid(c1_100_raw['low_res']).item()}

# Load L.7 C25 step 100
c25_100 = torch.load(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000100.pt", map_location='cpu')
c25_100_raw = {'bottleneck': c25_100['model_state_dict']['fusion_modules.bottleneck.gate'].item(),
                'low_res': c25_100['model_state_dict']['fusion_modules.low_res.gate'].item()}
c25_100_activated = {'bottleneck': sigmoid(c25_100_raw['bottleneck']).item(),
                      'low_res': sigmoid(c25_100_raw['low_res']).item()}

# Load L.8 C1 step 150
c1_150 = torch.load(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/C1/checkpoint_step_000150.pt", map_location='cpu')
c1_150_raw = {'bottleneck': c1_150['model_state_dict']['fusion_modules.bottleneck.gate'].item(),
               'low_res': c1_150['model_state_dict']['fusion_modules.low_res.gate'].item()}
c1_150_activated = {'bottleneck': sigmoid(c1_150_raw['bottleneck']).item(),
                     'low_res': sigmoid(c1_150_raw['low_res']).item()}

# Load L.8 C1 step 200
c1_200 = torch.load(base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/C1/checkpoint_step_000200.pt", map_location='cpu')
c1_200_raw = {'bottleneck': c1_200['model_state_dict']['fusion_modules.bottleneck.gate'].item(),
               'low_res': c1_200['model_state_dict']['fusion_modules.low_res.gate'].item()}
c1_200_activated = {'bottleneck': sigmoid(c1_200_raw['bottleneck']).item(),
                     'low_res': sigmoid(c1_200_raw['low_res']).item()}

# Calculate changes
def calc_change(initial, final):
    return ((final - initial) / initial) * 100 if initial != 0 else 0

gate_progression = {
    'C1': {
        'step_100': {'raw': c1_100_raw, 'activated': c1_100_activated},
        'step_150': {'raw': c1_150_raw, 'activated': c1_150_activated},
        'step_200': {'raw': c1_200_raw, 'activated': c1_200_activated},
        'change_100_to_150': {
            'bottleneck_percent': calc_change(c1_100_activated['bottleneck'], c1_150_activated['bottleneck']),
            'low_res_percent': calc_change(c1_100_activated['low_res'], c1_150_activated['low_res'])
        },
        'change_100_to_200': {
            'bottleneck_percent': calc_change(c1_100_activated['bottleneck'], c1_200_activated['bottleneck']),
            'low_res_percent': calc_change(c1_100_activated['low_res'], c1_200_activated['low_res'])
        }
    },
    'C25': {
        'step_100': {'raw': c25_100_raw, 'activated': c25_100_activated},
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

print("\nC1 Gate Progression (Activated Values):")
print(f"  Step 100: bottleneck={c1_100_activated['bottleneck']:.6f}, low_res={c1_100_activated['low_res']:.6f}")
print(f"  Step 150: bottleneck={c1_150_activated['bottleneck']:.6f}, low_res={c1_150_activated['low_res']:.6f}")
print(f"  Step 200: bottleneck={c1_200_activated['bottleneck']:.6f}, low_res={c1_200_activated['low_res']:.6f}")
print(f"  Change 100->150: bottleneck={gate_progression['C1']['change_100_to_150']['bottleneck_percent']:.2f}%, low_res={gate_progression['C1']['change_100_to_150']['low_res_percent']:.2f}%")
print(f"  Change 100->200: bottleneck={gate_progression['C1']['change_100_to_200']['bottleneck_percent']:.2f}%, low_res={gate_progression['C1']['change_100_to_200']['low_res_percent']:.2f}%")

print("\nC25 Gate Values (step 100 only):")
print(f"  Step 100: bottleneck={c25_100_activated['bottleneck']:.6f}, low_res={c25_100_activated['low_res']:.6f}")
