"""
Stage 3.2.9C - Timing Audit

Investigate the Experiment B timing inconsistency.
Recalculate all statistics from raw records.

Author: Devin
Date: 2026-11-02
"""

import json
import numpy as np
from pathlib import Path


def analyze_timing(timing_data, experiment_name):
    """Analyze timing data for an experiment."""
    total_step_times = timing_data.get('total_step', [])

    if not total_step_times:
        return {
            "error": "No timing data found"
        }

    # Calculate statistics
    total_time = sum(total_step_times)
    num_steps = len(total_step_times)
    mean_time = np.mean(total_step_times)
    median_time = np.median(total_step_times)
    p90_time = np.percentile(total_step_times, 90)
    max_time = np.max(total_step_times)
    min_time = np.min(total_step_times)
    std_time = np.std(total_step_times)

    # Find outlier steps (> 3 std from mean)
    outlier_threshold = mean_time + 3 * std_time
    outliers = [(i, t) for i, t in enumerate(total_step_times) if t > outlier_threshold]

    # Check if max is an outlier
    max_index = np.argmax(total_step_times)
    max_is_outlier = total_step_times[max_index] > outlier_threshold

    return {
        "num_steps": int(num_steps),
        "total_time_seconds": float(total_time),
        "total_time_minutes": float(total_time / 60),
        "mean_step_time": float(mean_time),
        "median_step_time": float(median_time),
        "p90_step_time": float(p90_time),
        "max_step_time": float(max_time),
        "min_step_time": float(min_time),
        "std_step_time": float(std_time),
        "outlier_threshold": float(outlier_threshold),
        "num_outliers": int(len(outliers)),
        "outlier_indices": [int(o[0]) for o in outliers],
        "outlier_values": [float(o[1]) for o in outliers],
        "max_step_index": int(max_index),
        "max_is_outlier": bool(max_is_outlier),
        "outlier_rate": float(len(outliers) / num_steps if num_steps > 0 else 0)
    }


def main():
    # Load Phase 2C results
    results_path = Path(__file__).parent.parent / "phase2C" / "phase2c_pilot_results.json"

    with open(results_path, 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("  STAGE 3.2.9C - TIMING AUDIT")
    print("=" * 80)

    # Analyze Experiment B
    print("\n--- EXPERIMENT B TIMING ANALYSIS ---")
    exp_B_timing = data['experiment_B']['timing']
    exp_B_stats = analyze_timing(exp_B_timing, "B")

    print(f"Number of steps: {exp_B_stats['num_steps']}")
    print(f"Total time: {exp_B_stats['total_time_seconds']:.2f}s ({exp_B_stats['total_time_minutes']:.2f} min)")
    print(f"Mean step time: {exp_B_stats['mean_step_time']:.2f}s")
    print(f"Median step time: {exp_B_stats['median_step_time']:.2f}s")
    print(f"P90 step time: {exp_B_stats['p90_step_time']:.2f}s")
    print(f"Max step time: {exp_B_stats['max_step_time']:.2f}s")
    print(f"Min step time: {exp_B_stats['min_step_time']:.2f}s")
    print(f"Std step time: {exp_B_stats['std_step_time']:.2f}s")
    print(f"Outlier threshold: {exp_B_stats['outlier_threshold']:.2f}s")
    print(f"Number of outliers: {exp_B_stats['num_outliers']}")
    print(f"Outlier rate: {exp_B_stats['outlier_rate']:.2%}")
    print(f"Max step index: {exp_B_stats['max_step_index']}")
    print(f"Max is outlier: {exp_B_stats['max_is_outlier']}")

    if exp_B_stats['num_outliers'] > 0:
        print(f"\nOutlier details:")
        for idx, val in zip(exp_B_stats['outlier_indices'], exp_B_stats['outlier_values']):
            print(f"  Step {idx}: {val:.2f}s")

    # Analyze Experiment C
    print("\n--- EXPERIMENT C TIMING ANALYSIS ---")
    exp_C_timing = data['experiment_C']['timing']
    exp_C_stats = analyze_timing(exp_C_timing, "C")

    print(f"Number of steps: {exp_C_stats['num_steps']}")
    print(f"Total time: {exp_C_stats['total_time_seconds']:.2f}s ({exp_C_stats['total_time_minutes']:.2f} min)")
    print(f"Mean step time: {exp_C_stats['mean_step_time']:.2f}s")
    print(f"Median step time: {exp_C_stats['median_step_time']:.2f}s")
    print(f"P90 step time: {exp_C_stats['p90_step_time']:.2f}s")
    print(f"Max step time: {exp_C_stats['max_step_time']:.2f}s")
    print(f"Min step time: {exp_C_stats['min_step_time']:.2f}s")
    print(f"Std step time: {exp_C_stats['std_step_time']:.2f}s")
    print(f"Outlier threshold: {exp_C_stats['outlier_threshold']:.2f}s")
    print(f"Number of outliers: {exp_C_stats['num_outliers']}")
    print(f"Outlier rate: {exp_C_stats['outlier_rate']:.2%}")

    # Compare with Phase 2C report
    print("\n--- COMPARISON WITH PHASE 2C REPORT ---")
    print("Phase 2C Report (Experiment B):")
    print("  Total time: 354s (5.9 min)")
    print("  Mean step time: 3.54s")
    print("  Median step time: 3.31s")
    print("  Max step time: 941.87s")
    print("  Outlier at step 62")

    print("\nRecalculated from raw data (Experiment B):")
    print(f"  Total time: {exp_B_stats['total_time_seconds']:.2f}s ({exp_B_stats['total_time_minutes']:.2f} min)")
    print(f"  Mean step time: {exp_B_stats['mean_step_time']:.2f}s")
    print(f"  Median step time: {exp_B_stats['median_step_time']:.2f}s")
    print(f"  Max step time: {exp_B_stats['max_step_time']:.2f}s")
    print(f"  Max step index: {exp_B_stats['max_step_index']}")
    print(f"  Number of outliers: {exp_B_stats['num_outliers']}")

    # Consistency check
    print("\n--- CONSISTENCY CHECK ---")
    if abs(exp_B_stats['total_time_seconds'] - 354) < 1:
        print("[OK] Total time matches Phase 2C report")
    else:
        print(f"[MISMATCH] Total time difference: {exp_B_stats['total_time_seconds'] - 354:.2f}s")

    if abs(exp_B_stats['mean_step_time'] - 3.54) < 0.1:
        print("[OK] Mean step time matches Phase 2C report")
    else:
        print(f"[MISMATCH] Mean step time difference: {exp_B_stats['mean_step_time'] - 3.54:.2f}s")

    if abs(exp_B_stats['median_step_time'] - 3.31) < 0.1:
        print("[OK] Median step time matches Phase 2C report")
    else:
        print(f"[MISMATCH] Median step time difference: {exp_B_stats['median_step_time'] - 3.31:.2f}s")

    if abs(exp_B_stats['max_step_time'] - 941.87) < 1:
        print("[OK] Max step time matches Phase 2C report")
    else:
        print(f"[MISMATCH] Max step time difference: {exp_B_stats['max_step_time'] - 941.87:.2f}s")

    # Mathematical consistency check
    print("\n--- MATHEMATICAL CONSISTENCY CHECK ---")
    expected_total_from_mean = exp_B_stats['mean_step_time'] * exp_B_stats['num_steps']
    print(f"Expected total from mean * steps: {expected_total_from_mean:.2f}s")
    print(f"Actual total: {exp_B_stats['total_time_seconds']:.2f}s")
    print(f"Difference: {abs(expected_total_from_mean - exp_B_stats['total_time_seconds']):.2f}s")

    if exp_B_stats['max_step_time'] > exp_B_stats['total_time_seconds']:
        print(f"[INCONSISTENCY] Max step time ({exp_B_stats['max_step_time']:.2f}s) > Total time ({exp_B_stats['total_time_seconds']:.2f}s)")
        print("This is mathematically impossible for a single step in a multi-step run.")

    # Save results
    audit_results = {
        "experiment_B": exp_B_stats,
        "experiment_C": exp_C_stats,
        "phase2c_report": {
            "experiment_B": {
                "total_time": 354,
                "mean_step_time": 3.54,
                "median_step_time": 3.31,
                "max_step_time": 941.87,
                "outlier_step": 62
            }
        },
        "consistency_check": {
            "total_time_match": bool(abs(exp_B_stats['total_time_seconds'] - 354) < 1),
            "mean_step_time_match": bool(abs(exp_B_stats['mean_step_time'] - 3.54) < 0.1),
            "median_step_time_match": bool(abs(exp_B_stats['median_step_time'] - 3.31) < 0.1),
            "max_step_time_match": bool(abs(exp_B_stats['max_step_time'] - 941.87) < 1)
        },
        "mathematical_consistency": {
            "expected_total_from_mean": expected_total_from_mean,
            "actual_total": exp_B_stats['total_time_seconds'],
            "max_greater_than_total": bool(exp_B_stats['max_step_time'] > exp_B_stats['total_time_seconds'])
        }
    }

    output_path = Path(__file__).parent / "timing_audit_results.json"
    with open(output_path, 'w') as f:
        json.dump(audit_results, f, indent=2)

    print(f"\nSaved audit results to {output_path}")

    return audit_results


if __name__ == "__main__":
    results = main()
