"""
Stage 3.2.9B - Phase 2B: Outlier Analysis

Analyze the 301-second outlier from Phase 2A smoke test.

Author: Devin
Date: 2026-11-02
"""

import json
import statistics
from pathlib import Path

def analyze_outlier():
    """Analyze the outlier in Experiment C timing data."""

    # Load smoke test results
    smoke_test_dir = Path(__file__).parent.parent / "smoke_test"
    results_path = smoke_test_dir / "STAGE3_2_9B_SMOKE_TEST_RESULTS.json"

    with open(results_path, 'r') as f:
        data = json.load(f)

    timing = data['experiment_C']['timing']

    print("=" * 80)
    print("  OUTLIER ANALYSIS - EXPERIMENT C")
    print("=" * 80)

    # Total step times
    total_times = timing['total_step']
    print("\nTotal Step Times:")
    for i, t in enumerate(total_times):
        print(f"  Step {i+1}: {t:.2f}s")

    # Calculate statistics
    median = statistics.median(total_times)
    mean = statistics.mean(total_times)
    std = statistics.stdev(total_times) if len(total_times) > 1 else 0

    print(f"\nStatistics:")
    print(f"  Mean: {mean:.2f}s")
    print(f"  Median: {median:.2f}s")
    print(f"  Std Dev: {std:.2f}s")

    # Identify outlier (step 6)
    outlier_step = 5  # 0-indexed
    outlier_time = total_times[outlier_step]
    slowdown_factor = outlier_time / median

    print(f"\nOutlier Detection:")
    print(f"  Step {outlier_step + 1}: {outlier_time:.2f}s")
    print(f"  Median: {median:.2f}s")
    print(f"  Slowdown factor: {slowdown_factor:.2f}x")

    # Breakdown of outlier step
    print(f"\nOutlier Step Breakdown:")
    print(f"  Data loading: {timing['data_loading'][outlier_step]:.4f}s")
    print(f"  Forward pass: {timing['forward_pass'][outlier_step]:.2f}s")
    print(f"  Backward pass: {timing['backward_pass'][outlier_step]:.2f}s")
    print(f"  Optimizer step: {timing['optimizer_step'][outlier_step]:.4f}s")
    print(f"  Total: {timing['total_step'][outlier_step]:.2f}s")

    # Compare with normal step (e.g., step 5)
    normal_step = 4  # 0-indexed
    print(f"\nNormal Step Comparison (Step {normal_step + 1}):")
    print(f"  Data loading: {timing['data_loading'][normal_step]:.4f}s")
    print(f"  Forward pass: {timing['forward_pass'][normal_step]:.2f}s")
    print(f"  Backward pass: {timing['backward_pass'][normal_step]:.2f}s")
    print(f"  Optimizer step: {timing['optimizer_step'][normal_step]:.4f}s")
    print(f"  Total: {timing['total_step'][normal_step]:.2f}s")

    # Calculate where the slowdown occurred
    forward_slowdown = timing['forward_pass'][outlier_step] / timing['forward_pass'][normal_step]
    backward_slowdown = timing['backward_pass'][outlier_step] / timing['backward_pass'][normal_step]

    print(f"\nSlowdown Breakdown:")
    print(f"  Forward pass: {forward_slowdown:.2f}x")
    print(f"  Backward pass: {backward_slowdown:.2f}x")

    # Analyze Experiment B for comparison
    timing_b = data['experiment_B']['timing']
    total_times_b = timing_b['total_step']
    median_b = statistics.median(total_times_b)

    print(f"\nExperiment B Comparison (for context):")
    print(f"  Median step time: {median_b:.2f}s")
    print(f"  Experiment C median: {median:.2f}s")
    print(f"  Temporal overhead: {median / median_b:.2f}x")

    # Save analysis results
    analysis = {
        'outlier_step': outlier_step + 1,
        'outlier_time': outlier_time,
        'median_time': median,
        'slowdown_factor': slowdown_factor,
        'mean_time': mean,
        'std_dev': std,
        'outlier_breakdown': {
            'data_loading': timing['data_loading'][outlier_step],
            'forward_pass': timing['forward_pass'][outlier_step],
            'backward_pass': timing['backward_pass'][outlier_step],
            'optimizer_step': timing['optimizer_step'][outlier_step]
        },
        'normal_step_comparison': {
            'step': normal_step + 1,
            'data_loading': timing['data_loading'][normal_step],
            'forward_pass': timing['forward_pass'][normal_step],
            'backward_pass': timing['backward_pass'][normal_step],
            'optimizer_step': timing['optimizer_step'][normal_step]
        },
        'slowdown_breakdown': {
            'forward_slowdown': forward_slowdown,
            'backward_slowdown': backward_slowdown
        },
        'experiment_b_median': median_b,
        'temporal_overhead': median / median_b
    }

    output_path = Path(__file__).parent / "outlier_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to {output_path}")

    return analysis


if __name__ == "__main__":
    analyze_outlier()
