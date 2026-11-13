"""
Stage 3.2.9C - Loss Analysis

Investigate the loss discrepancy between Experiment B and Experiment C.

Author: Devin
Date: 2026-11-02
"""

import json
import numpy as np
from pathlib import Path


def analyze_loss_data(losses, experiment_name):
    """Analyze loss data for an experiment."""
    if not losses:
        return {"error": "No loss data found"}

    losses_array = np.array(losses)

    return {
        "num_steps": len(losses),
        "initial_loss": float(losses_array[0]),
        "final_loss": float(losses_array[-1]),
        "min_loss": float(losses_array.min()),
        "max_loss": float(losses_array.max()),
        "mean_loss": float(losses_array.mean()),
        "std_loss": float(losses_array.std()),
        "loss_range": [float(losses_array.min()), float(losses_array.max())],
        "loss_reduction": float(losses_array[0] - losses_array[-1]),
        "loss_reduction_percent": float((losses_array[0] - losses_array[-1]) / losses_array[0] * 100)
    }


def main():
    # Load Phase 2C results
    results_path = Path(__file__).parent.parent / "phase2C" / "phase2c_pilot_results.json"

    with open(results_path, 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("  STAGE 3.2.9C - LOSS ANALYSIS")
    print("=" * 80)

    # Analyze Experiment B
    print("\n--- EXPERIMENT B LOSS ANALYSIS ---")
    exp_B_losses = data['experiment_B']['losses']
    exp_B_stats = analyze_loss_data(exp_B_losses, "B")

    print(f"Number of steps: {exp_B_stats['num_steps']}")
    print(f"Initial loss: {exp_B_stats['initial_loss']:.6f}")
    print(f"Final loss: {exp_B_stats['final_loss']:.6f}")
    print(f"Min loss: {exp_B_stats['min_loss']:.6f}")
    print(f"Max loss: {exp_B_stats['max_loss']:.6f}")
    print(f"Mean loss: {exp_B_stats['mean_loss']:.6f}")
    print(f"Std loss: {exp_B_stats['std_loss']:.6f}")
    print(f"Loss reduction: {exp_B_stats['loss_reduction']:.6f} ({exp_B_stats['loss_reduction_percent']:.2f}%)")

    # Analyze Experiment C
    print("\n--- EXPERIMENT C LOSS ANALYSIS ---")
    exp_C_losses = data['experiment_C']['losses']
    exp_C_stats = analyze_loss_data(exp_C_losses, "C")

    print(f"Number of steps: {exp_C_stats['num_steps']}")
    print(f"Initial loss: {exp_C_stats['initial_loss']:.6f}")
    print(f"Final loss: {exp_C_stats['final_loss']:.6f}")
    print(f"Min loss: {exp_C_stats['min_loss']:.6f}")
    print(f"Max loss: {exp_C_stats['max_loss']:.6f}")
    print(f"Mean loss: {exp_C_stats['mean_loss']:.6f}")
    print(f"Std loss: {exp_C_stats['std_loss']:.6f}")
    print(f"Loss reduction: {exp_C_stats['loss_reduction']:.6f} ({exp_C_stats['loss_reduction_percent']:.2f}%)")

    # Compare
    print("\n--- LOSS COMPARISON ---")
    print(f"Loss scale difference: {exp_C_stats['initial_loss'] / exp_B_stats['initial_loss']:.2f}x")
    print(f"Final loss ratio (C/B): {exp_C_stats['final_loss'] / exp_B_stats['final_loss']:.2f}x")
    print(f"Loss reduction ratio: {exp_C_stats['loss_reduction_percent'] / exp_B_stats['loss_reduction_percent']:.2f}x")

    # Parameter count comparison
    exp_B_params = data['experiment_B']['config']['trainable_params']
    exp_C_params = data['experiment_C']['config']['trainable_params']
    print(f"\nTrainable parameters:")
    print(f"  Experiment B: {exp_B_params:,}")
    print(f"  Experiment C: {exp_C_params:,}")
    print(f"  Ratio (C/B): {exp_C_params / exp_B_params:.2f}x")

    # Check if losses are on different scales
    print("\n--- SCALE ANALYSIS ---")
    if exp_C_stats['initial_loss'] > 10 * exp_B_stats['initial_loss']:
        print("[ISSUE] Experiment C losses are on a different scale than Experiment B")
        print("This suggests the loss function or prediction target may differ between experiments")
    else:
        print("[OK] Losses appear to be on comparable scales")

    # Loss stability
    print("\n--- LOSS STABILITY ---")
    exp_B_cv = exp_B_stats['std_loss'] / exp_B_stats['mean_loss']
    exp_C_cv = exp_C_stats['std_loss'] / exp_C_stats['mean_loss']
    print(f"Experiment B coefficient of variation: {exp_B_cv:.4f}")
    print(f"Experiment C coefficient of variation: {exp_C_cv:.4f}")

    if exp_C_cv > 2 * exp_B_cv:
        print("[OBSERVATION] Experiment C has higher loss variance")
        print("This could indicate more complex learning dynamics with temporal parameters")

    # Save results
    analysis_results = {
        "experiment_B": exp_B_stats,
        "experiment_C": exp_C_stats,
        "comparison": {
            "loss_scale_difference": float(exp_C_stats['initial_loss'] / exp_B_stats['initial_loss']),
            "final_loss_ratio": float(exp_C_stats['final_loss'] / exp_B_stats['final_loss']),
            "loss_reduction_ratio": float(exp_C_stats['loss_reduction_percent'] / exp_B_stats['loss_reduction_percent']),
            "parameter_ratio": float(exp_C_params / exp_B_params)
        },
        "scale_analysis": {
            "different_scales": bool(exp_C_stats['initial_loss'] > 10 * exp_B_stats['initial_loss'])
        },
        "stability_analysis": {
            "exp_B_cv": float(exp_B_cv),
            "exp_C_cv": float(exp_C_cv),
            "exp_C_more_volatile": bool(exp_C_cv > 2 * exp_B_cv)
        }
    }

    output_path = Path(__file__).parent / "loss_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\nSaved analysis results to {output_path}")

    return analysis_results


if __name__ == "__main__":
    results = main()
