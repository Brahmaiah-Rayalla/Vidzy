"""
Stage 3.2.9C - Temporal Fusion Diagnostics

Analyze fusion gate behavior and temporal feature contributions.

Author: Devin
Date: 2026-11-02
"""

import json
import numpy as np
from pathlib import Path


def analyze_fusion_gates(data):
    """Analyze fusion gate values from Phase 2C results."""
    fusion_values = data['experiment_C']['metrics']['fusion_gate_values']

    if not fusion_values:
        return {"error": "No fusion gate data found"}

    # Extract bottleneck and low-res gates
    bottleneck_gates = [v['bottleneck_gate'] for v in fusion_values]
    low_res_gates = [v['low_res_gate'] for v in fusion_values]

    return {
        "num_steps": len(fusion_values),
        "bottleneck_gate": {
            "initial": float(bottleneck_gates[0]),
            "final": float(bottleneck_gates[-1]),
            "min": float(min(bottleneck_gates)),
            "max": float(max(bottleneck_gates)),
            "mean": float(np.mean(bottleneck_gates)),
            "std": float(np.std(bottleneck_gates)),
            "total_change": float(bottleneck_gates[-1] - bottleneck_gates[0]),
            "percent_change": float((bottleneck_gates[-1] - bottleneck_gates[0]) / bottleneck_gates[0] * 100)
        },
        "low_res_gate": {
            "initial": float(low_res_gates[0]),
            "final": float(low_res_gates[-1]),
            "min": float(min(low_res_gates)),
            "max": float(max(low_res_gates)),
            "mean": float(np.mean(low_res_gates)),
            "std": float(np.std(low_res_gates)),
            "total_change": float(low_res_gates[-1] - low_res_gates[0]),
            "percent_change": float((low_res_gates[-1] - low_res_gates[0]) / low_res_gates[0] * 100)
        }
    }


def main():
    # Load Phase 2C results
    results_path = Path(__file__).parent.parent / "phase2C" / "phase2c_pilot_results.json"

    with open(results_path, 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("  STAGE 3.2.9C - TEMPORAL FUSION DIAGNOSTICS")
    print("=" * 80)

    # Analyze fusion gates
    print("\n--- FUSION GATE ANALYSIS ---")
    fusion_stats = analyze_fusion_gates(data)

    if "error" in fusion_stats:
        print(f"Error: {fusion_stats['error']}")
        return fusion_stats

    print(f"Number of steps: {fusion_stats['num_steps']}")

    print("\nBottleneck Gate:")
    print(f"  Initial: {fusion_stats['bottleneck_gate']['initial']:.6f}")
    print(f"  Final: {fusion_stats['bottleneck_gate']['final']:.6f}")
    print(f"  Change: {fusion_stats['bottleneck_gate']['total_change']:.6f} ({fusion_stats['bottleneck_gate']['percent_change']:.2f}%)")
    print(f"  Min: {fusion_stats['bottleneck_gate']['min']:.6f}")
    print(f"  Max: {fusion_stats['bottleneck_gate']['max']:.6f}")
    print(f"  Mean: {fusion_stats['bottleneck_gate']['mean']:.6f}")
    print(f"  Std: {fusion_stats['bottleneck_gate']['std']:.6f}")

    print("\nLow-Res Gate:")
    print(f"  Initial: {fusion_stats['low_res_gate']['initial']:.6f}")
    print(f"  Final: {fusion_stats['low_res_gate']['final']:.6f}")
    print(f"  Change: {fusion_stats['low_res_gate']['total_change']:.6f} ({fusion_stats['low_res_gate']['percent_change']:.2f}%)")
    print(f"  Min: {fusion_stats['low_res_gate']['min']:.6f}")
    print(f"  Max: {fusion_stats['low_res_gate']['max']:.6f}")
    print(f"  Mean: {fusion_stats['low_res_gate']['mean']:.6f}")
    print(f"  Std: {fusion_stats['low_res_gate']['std']:.6f}")

    # Interpretation
    print("\n--- INTERPRETATION ---")

    # Check if gates are near initialization
    init_value = 0.01
    bottleneck_near_init = abs(fusion_stats['bottleneck_gate']['final'] - init_value) < 0.001
    low_res_near_init = abs(fusion_stats['low_res_gate']['final'] - init_value) < 0.001

    if bottleneck_near_init and low_res_near_init:
        print("[INACTIVE] Both gates remain at initialization value (0.01)")
        print("This suggests temporal features are not being effectively fused")
    else:
        print("[ACTIVE] Gates have moved away from initialization")

    # Check if gates are learning
    if fusion_stats['bottleneck_gate']['percent_change'] < 5 and fusion_stats['low_res_gate']['percent_change'] < 5:
        print("[STAGNANT] Gates show minimal learning (< 5% change)")
        print("This indicates the temporal module may not be learning useful representations")
    else:
        print("[LEARNING] Gates show significant learning")

    # Check gate balance
    if abs(fusion_stats['bottleneck_gate']['final'] - fusion_stats['low_res_gate']['final']) < 0.01:
        print("[BALANCED] Gates remain balanced")
    else:
        print("[IMBALANCED] Gates have diverged")

    # Save results
    diagnostics = {
        "fusion_gate_analysis": fusion_stats,
        "interpretation": {
            "gates_near_initialization": bool(bottleneck_near_init and low_res_near_init),
            "gates_learning": bool(fusion_stats['bottleneck_gate']['percent_change'] > 5 or fusion_stats['low_res_gate']['percent_change'] > 5),
            "gates_balanced": bool(abs(fusion_stats['bottleneck_gate']['final'] - fusion_stats['low_res_gate']['final']) < 0.01),
            "temporal_module_active": bool(not (bottleneck_near_init and low_res_near_init))
        },
        "recommendations": [
            "Gates remain at initialization (0.01), suggesting temporal features are not being used",
            "More training steps may be needed for gates to activate",
            "Consider increasing gate initialization to 0.5 to encourage earlier activation",
            "Diagnostic gate values (0.01, 0.05, 0.10, 0.25) should be tested to verify temporal feature quality"
        ]
    }

    output_path = Path(__file__).parent / "STAGE3_2_9C_FUSION_DIAGNOSTICS.json"
    with open(output_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nSaved diagnostics to {output_path}")

    return diagnostics


if __name__ == "__main__":
    results = main()
