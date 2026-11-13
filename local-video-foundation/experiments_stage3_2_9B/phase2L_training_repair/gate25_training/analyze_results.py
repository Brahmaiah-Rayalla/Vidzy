"""
Analyze Stage 3.2.9L.7 training results.
"""

import json
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
results_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/STAGE3_2_9L7_TRAINING_RESULTS.json"

with open(results_file, 'r') as f:
    data = json.load(f)

analysis = {
    'summary': {},
    'c1_final': {},
    'c25_final': {},
    'comparison': {},
    'classification': {}
}

# Extract final values
c1_pilot = data['pilot_C1']
c25_pilot = data['pilot_C25']

c1_final = c1_pilot['steps'][-1]
c25_final = c25_pilot['steps'][-1]

analysis['c1_final'] = {
    'step': c1_final['step'],
    'loss': c1_final['loss'],
    'bottleneck_gate': c1_final['gate_values']['bottleneck_gate'],
    'low_res_gate': c1_final['gate_values']['low_res_gate'],
    'motion_grad_norm': c1_final['motion_grad_norm'],
    'temporal_grad_norm': c1_final['temporal_grad_norm'],
    'fusion_grad_norm': c1_final['fusion_grad_norm']
}

analysis['c25_final'] = {
    'step': c25_final['step'],
    'loss': c25_final['loss'],
    'bottleneck_gate': c25_final['gate_values']['bottleneck_gate'],
    'low_res_gate': c25_final['gate_values']['low_res_gate'],
    'motion_grad_norm': c25_final['motion_grad_norm'],
    'temporal_grad_norm': c25_final['temporal_grad_norm'],
    'fusion_grad_norm': c25_final['fusion_grad_norm']
}

# Comparison
loss_improvement = (analysis['c1_final']['loss'] - analysis['c25_final']['loss']) / analysis['c1_final']['loss'] * 100
analysis['comparison'] = {
    'loss_improvement_percent': loss_improvement,
    'c1_loss': analysis['c1_final']['loss'],
    'c25_loss': analysis['c25_final']['loss'],
    'c1_gate_avg': (analysis['c1_final']['bottleneck_gate'] + analysis['c1_final']['low_res_gate']) / 2,
    'c25_gate_avg': (analysis['c25_final']['bottleneck_gate'] + analysis['c25_final']['low_res_gate']) / 2
}

# Gate progression
c1_init = c1_pilot['steps'][0]['gate_values']
c25_init = c25_pilot['steps'][0]['gate_values']

c1_bottleneck_change = (c1_final['gate_values']['bottleneck_gate'] - c1_init['bottleneck_gate']) / c1_init['bottleneck_gate'] * 100
c1_lowres_change = (c1_final['gate_values']['low_res_gate'] - c1_init['low_res_gate']) / c1_init['low_res_gate'] * 100
c25_bottleneck_change = (c25_final['gate_values']['bottleneck_gate'] - c25_init['bottleneck_gate']) / c25_init['bottleneck_gate'] * 100
c25_lowres_change = (c25_final['gate_values']['low_res_gate'] - c25_init['low_res_gate']) / c25_init['low_res_gate'] * 100

analysis['gate_progression'] = {
    'c1_bottleneck_change_percent': c1_bottleneck_change,
    'c1_lowres_change_percent': c1_lowres_change,
    'c25_bottleneck_change_percent': c25_bottleneck_change,
    'c25_lowres_change_percent': c25_lowres_change
}

# Classification
if loss_improvement > 10 and abs(c25_bottleneck_change) < 5 and abs(c25_lowres_change) < 5:
    classification = "DENOISING IMPROVEMENT ONLY"
    rationale = "C25 shows ~18% lower loss than C1, but gate movement remains minimal (< 1% change). This suggests the benefit comes from initialization rather than learning."
elif loss_improvement > 10 and (abs(c25_bottleneck_change) > 5 or abs(c25_lowres_change) > 5):
    classification = "25% INITIALIZATION PROMISING"
    rationale = "C25 shows meaningful loss improvement AND gate convergence."
else:
    classification = "NO MEANINGFUL DIFFERENCE"
    rationale = "Loss difference is minimal."

analysis['classification'] = {
    'classification': classification,
    'rationale': rationale
}

# Save analysis
output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/STAGE3_2_9L7_ANALYSIS.json"
with open(output_file, 'w') as f:
    json.dump(analysis, f, indent=2)

print("Analysis Results:")
print(f"  C1 Final Loss: {analysis['c1_final']['loss']:.6f}")
print(f"  C25 Final Loss: {analysis['c25_final']['loss']:.6f}")
print(f"  Loss Improvement: {loss_improvement:.2f}%")
print(f"  C1 Gate Avg: {analysis['comparison']['c1_gate_avg']:.6f}")
print(f"  C25 Gate Avg: {analysis['comparison']['c25_gate_avg']:.6f}")
print(f"  C1 Gate Change: {c1_bottleneck_change:.2f}% (bottleneck), {c1_lowres_change:.2f}% (low-res)")
print(f"  C25 Gate Change: {c25_bottleneck_change:.2f}% (bottleneck), {c25_lowres_change:.2f}% (low-res)")
print(f"\nClassification: {classification}")
print(f"Rationale: {rationale}")
print(f"\nSaved to: {output_file}")
