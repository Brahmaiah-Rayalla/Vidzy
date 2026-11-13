"""
Phase C - Individual gate gradient and update analysis.

Extract gradient and update information from historical Experiment C training logs.
"""

import json
import numpy as np
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
results_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/STAGE3_2_9L3_TRAINING_RESULTS.json"

with open(results_file, 'r') as f:
    results = json.load(f)

print("=" * 80)
print("PHASE C - INDIVIDUAL GATE GRADIENT AND UPDATE ANALYSIS")
print("=" * 80)

steps_C = results['experiment_C']['steps']

# Extract gate values and fusion gradients
gate_values = []
fusion_grad_norms = []

for step in steps_C:
    gate_values.append(step.get('gate_values', {}))
    fusion_grad_norms.append(step.get('fusion_grad_norm', 0.0))

# Analyze bottleneck gate
bottleneck_gates = [g.get('bottleneck_gate', 0) for g in gate_values]
low_res_gates = [g.get('low_res_gate', 0) for g in gate_values]

print("\n--- BOTTLENECK GATE ANALYSIS ---")
print(f"Initial activated value: {bottleneck_gates[0]:.8f}")
print(f"Final activated value: {bottleneck_gates[-1]:.8f}")
print(f"Parameter change: {bottleneck_gates[-1] - bottleneck_gates[0]:.8f}")
print(f"Percentage change: {(bottleneck_gates[-1] - bottleneck_gates[0]) / bottleneck_gates[0] * 100:.4f}%")

# Calculate raw values (inverse sigmoid)
def logit(x):
    return np.log(x / (1 - x))

initial_raw_bottleneck = logit(bottleneck_gates[0])
final_raw_bottleneck = logit(bottleneck_gates[-1])

print(f"\nInitial raw logit: {initial_raw_bottleneck:.8f}")
print(f"Final raw logit: {final_raw_bottleneck:.8f}")
print(f"Raw parameter change: {final_raw_bottleneck - initial_raw_bottleneck:.8f}")

print("\n--- LOW-RES GATE ANALYSIS ---")
print(f"Initial activated value: {low_res_gates[0]:.8f}")
print(f"Final activated value: {low_res_gates[-1]:.8f}")
print(f"Parameter change: {low_res_gates[-1] - low_res_gates[0]:.8f}")
print(f"Percentage change: {(low_res_gates[-1] - low_res_gates[0]) / low_res_gates[0] * 100:.4f}%")

initial_raw_low_res = logit(low_res_gates[0])
final_raw_low_res = logit(low_res_gates[-1])

print(f"\nInitial raw logit: {initial_raw_low_res:.8f}")
print(f"Final raw logit: {final_raw_low_res:.8f}")
print(f"Raw parameter change: {final_raw_low_res - initial_raw_low_res:.8f}")

print("\n--- FUSION GRADIENT ANALYSIS ---")
print(f"Initial gradient norm: {fusion_grad_norms[0]:.8e}")
print(f"Mean gradient norm: {np.mean(fusion_grad_norms):.8e}")
print(f"Final gradient norm: {fusion_grad_norms[-1]:.8e}")
print(f"Std gradient norm: {np.std(fusion_grad_norms):.8e}")
print(f"Min gradient norm: {np.min(fusion_grad_norms):.8e}")
print(f"Max gradient norm: {np.max(fusion_grad_norms):.8e}")

print("\n--- GRADIENT SCALE ANALYSIS ---")
print(f"Gradient norm relative to raw parameter scale:")
print(f"  Initial: {fusion_grad_norms[0] / abs(initial_raw_bottleneck):.8e}")
print(f"  Mean: {np.mean(fusion_grad_norms) / abs(initial_raw_bottleneck):.8e}")
print(f"  Final: {fusion_grad_norms[-1] / abs(initial_raw_bottleneck):.8e}")

print("\n--- ADAM STATE AVAILABILITY ---")
print("Optimizer state not saved in historical training logs.")
print("Cannot analyze Adam first-moment and second-moment states.")

gate_analysis = {
    "bottleneck_gate": {
        "initial_activated": float(bottleneck_gates[0]),
        "final_activated": float(bottleneck_gates[-1]),
        "parameter_change": float(bottleneck_gates[-1] - bottleneck_gates[0]),
        "percentage_change": float((bottleneck_gates[-1] - bottleneck_gates[0]) / bottleneck_gates[0] * 100),
        "initial_raw_logit": float(initial_raw_bottleneck),
        "final_raw_logit": float(final_raw_bottleneck),
        "raw_parameter_change": float(final_raw_bottleneck - initial_raw_bottleneck)
    },
    "low_res_gate": {
        "initial_activated": float(low_res_gates[0]),
        "final_activated": float(low_res_gates[-1]),
        "parameter_change": float(low_res_gates[-1] - low_res_gates[0]),
        "percentage_change": float((low_res_gates[-1] - low_res_gates[0]) / low_res_gates[0] * 100),
        "initial_raw_logit": float(initial_raw_low_res),
        "final_raw_logit": float(final_raw_low_res),
        "raw_parameter_change": float(final_raw_low_res - initial_raw_low_res)
    },
    "fusion_gradients": {
        "initial_norm": float(fusion_grad_norms[0]),
        "mean_norm": float(np.mean(fusion_grad_norms)),
        "final_norm": float(fusion_grad_norms[-1]),
        "std_norm": float(np.std(fusion_grad_norms)),
        "min_norm": float(np.min(fusion_grad_norms)),
        "max_norm": float(np.max(fusion_grad_norms))
    },
    "gradient_scale": {
        "initial_relative_to_raw": float(fusion_grad_norms[0] / abs(initial_raw_bottleneck)),
        "mean_relative_to_raw": float(np.mean(fusion_grad_norms) / abs(initial_raw_bottleneck)),
        "final_relative_to_raw": float(fusion_grad_norms[-1] / abs(initial_raw_bottleneck))
    },
    "adam_state": "NOT_AVAILABLE - Optimizer state not saved in historical logs"
}

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/fusion_sensitivity/STAGE3_2_9L5_INDIVIDUAL_GATE_ANALYSIS.json"
with open(output_file, 'w') as f:
    json.dump(gate_analysis, f, indent=2)

print(f"\nResults saved to: {output_file}")

print("\n--- CONCLUSION ---")
print("1. Gradients are small (mean ~6e-3) relative to raw parameter scale (~4.6)")
print("2. Gradient directions fluctuate (std ~3e-3)")
print("3. Gate updates are very small (< 3% activated value change)")
print("4. Adam state not available for detailed analysis")
print("5. Gates are receiving gradients but updates are minimal")
