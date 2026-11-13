"""
Stage 3.2.9K - Evaluation Audit

Audits Stage 3.2.9J evaluation results to verify metric calculations.

Author: Devin
Date: 2026-11-03
"""

import json
import numpy as np
from pathlib import Path

def audit_evaluation_results():
    """Audit Stage 3.2.9J evaluation results."""
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2J_motion_analysis")

    # Load evaluation results
    with open(base_dir / "STAGE3_2_9J_DIRECTION_ACCURACY.json") as f:
        direction_accuracy = json.load(f)

    with open(base_dir / "STAGE3_2_9J_MOTION_METRICS.json") as f:
        motion_metrics = json.load(f)

    with open(base_dir / "STAGE3_2_9J_STATIONARY_CONTROL.json") as f:
        stationary_control = json.load(f)

    with open(base_dir / "STAGE3_2_9J_GENERATION_MATRIX_AUDIT.json") as f:
        generation_audit = json.load(f)

    audit_results = {
        'direction_accuracy': {},
        'stationary_control': {},
        'verification': {},
        'discrepancies': []
    }

    # Verify total count
    total_videos = len(direction_accuracy)
    expected_total = 36

    audit_results['verification']['total_videos'] = total_videos
    audit_results['verification']['expected_total'] = expected_total
    audit_results['verification']['count_match'] = bool(total_videos == expected_total)

    if total_videos != expected_total:
        audit_results['discrepancies'].append({
            'issue': 'Video count mismatch',
            'expected': expected_total,
            'actual': total_videos
        })

    # Recalculate overall direction accuracy
    correct_count = 0
    total_count = 0
    failed_detections = 0

    for key, data in direction_accuracy.items():
        if 'accuracy' in data:
            total_count += 1
            if data['accuracy'] > 0.5:
                correct_count += 1
            # Check for failed detections (if accuracy is None or error)
            if data.get('accuracy') is None or 'error' in data:
                failed_detections += 1

    recalculated_accuracy = correct_count / total_count if total_count > 0 else 0.0
    reported_accuracy = 0.19444444444444445  # From Stage 3.2.9J report

    audit_results['direction_accuracy']['recalculated'] = {
        'correct_count': correct_count,
        'total_count': total_count,
        'accuracy': recalculated_accuracy
    }

    audit_results['direction_accuracy']['reported'] = {
        'accuracy': reported_accuracy
    }

    audit_results['direction_accuracy']['match'] = bool(abs(recalculated_accuracy - reported_accuracy) < 0.001)

    if not audit_results['direction_accuracy']['match']:
        audit_results['discrepancies'].append({
            'issue': 'Direction accuracy mismatch',
            'reported': reported_accuracy,
            'recalculated': recalculated_accuracy
        })

    # Per-experiment accuracy
    per_experiment = {'B': {'correct': 0, 'total': 0},
                      'C': {'correct': 0, 'total': 0},
                      'D': {'correct': 0, 'total': 0}}

    for key, data in direction_accuracy.items():
        config = data.get('config')
        if config in per_experiment:
            per_experiment[config]['total'] += 1
            if data.get('accuracy', 0) > 0.5:
                per_experiment[config]['correct'] += 1

    for config in ['B', 'C', 'D']:
        acc = per_experiment[config]['correct'] / per_experiment[config]['total'] if per_experiment[config]['total'] > 0 else 0.0
        audit_results['direction_accuracy'][f'{config}_accuracy'] = {
            'correct': per_experiment[config]['correct'],
            'total': per_experiment[config]['total'],
            'accuracy': acc
        }

    # Verify reported per-experiment accuracies
    reported_B = 0.167
    reported_C = 0.250
    reported_D = 0.167

    if abs(audit_results['direction_accuracy']['B_accuracy']['accuracy'] - reported_B) > 0.01:
        audit_results['discrepancies'].append({
            'issue': 'B accuracy mismatch',
            'reported': reported_B,
            'recalculated': audit_results['direction_accuracy']['B_accuracy']['accuracy']
        })

    if abs(audit_results['direction_accuracy']['C_accuracy']['accuracy'] - reported_C) > 0.01:
        audit_results['discrepancies'].append({
            'issue': 'C accuracy mismatch',
            'reported': reported_C,
            'recalculated': audit_results['direction_accuracy']['C_accuracy']['accuracy']
        })

    if abs(audit_results['direction_accuracy']['D_accuracy']['accuracy'] - reported_D) > 0.01:
        audit_results['discrepancies'].append({
            'issue': 'D accuracy mismatch',
            'reported': reported_D,
            'recalculated': audit_results['direction_accuracy']['D_accuracy']['accuracy']
        })

    # Per-direction accuracy
    per_direction = {0: {'correct': 0, 'total': 0},  # moving left
                     1: {'correct': 0, 'total': 0},  # moving right
                     2: {'correct': 0, 'total': 0},  # moving diagonal
                     3: {'correct': 0, 'total': 0}}  # stationary

    for key, data in direction_accuracy.items():
        prompt_idx = data.get('prompt_idx')
        if prompt_idx in per_direction:
            per_direction[prompt_idx]['total'] += 1
            if data.get('accuracy', 0) > 0.5:
                per_direction[prompt_idx]['correct'] += 1

    for prompt_idx in range(4):
        acc = per_direction[prompt_idx]['correct'] / per_direction[prompt_idx]['total'] if per_direction[prompt_idx]['total'] > 0 else 0.0
        audit_results['direction_accuracy'][f'prompt_{prompt_idx}_accuracy'] = {
            'correct': per_direction[prompt_idx]['correct'],
            'total': per_direction[prompt_idx]['total'],
            'accuracy': acc
        }

    # Recalculate stationary drift
    stationary_drifts = []
    for key, data in stationary_control.items():
        if 'displacement_magnitude' in data:
            stationary_drifts.append(data['displacement_magnitude'])

    recalculated_drift = np.mean(stationary_drifts) if stationary_drifts else 0.0
    reported_drift = 25.488747534374298

    audit_results['stationary_control']['recalculated'] = {
        'mean_drift': float(recalculated_drift),
        'count': len(stationary_drifts),
        'min': float(np.min(stationary_drifts)) if stationary_drifts else 0.0,
        'max': float(np.max(stationary_drifts)) if stationary_drifts else 0.0
    }

    audit_results['stationary_control']['reported'] = {
        'mean_drift': reported_drift
    }

    audit_results['stationary_control']['match'] = bool(abs(recalculated_drift - reported_drift) < 0.1)

    if not audit_results['stationary_control']['match']:
        audit_results['discrepancies'].append({
            'issue': 'Stationary drift mismatch',
            'reported': reported_drift,
            'recalculated': recalculated_drift
        })

    # Verify 5-pixel stationary threshold
    threshold = 5.0
    stationary_count = sum(1 for d in stationary_drifts if d < threshold)
    audit_results['stationary_control']['threshold_verification'] = {
        'threshold': threshold,
        'stationary_below_threshold': stationary_count,
        'total_stationary': len(stationary_drifts),
        'stationary_rate': stationary_count / len(stationary_drifts) if stationary_drifts else 0.0
    }

    # Per-seed metrics
    per_seed = {42: {'correct': 0, 'total': 0},
                123: {'correct': 0, 'total': 0},
                456: {'correct': 0, 'total': 0}}

    for key, data in direction_accuracy.items():
        seed = data.get('seed')
        if seed in per_seed:
            per_seed[seed]['total'] += 1
            if data.get('accuracy', 0) > 0.5:
                per_seed[seed]['correct'] += 1

    for seed in [42, 123, 456]:
        acc = per_seed[seed]['correct'] / per_seed[seed]['total'] if per_seed[seed]['total'] > 0 else 0.0
        audit_results['direction_accuracy'][f'seed_{seed}_accuracy'] = {
            'correct': per_seed[seed]['correct'],
            'total': per_seed[seed]['total'],
            'accuracy': acc
        }

    # Mean and median displacement
    displacements = []
    for key, data in motion_metrics.items():
        if 'total_displacement' in data and data['total_displacement']:
            dx = data['total_displacement'][1]
            dy = data['total_displacement'][0]
            displacement_magnitude = np.sqrt(dx**2 + dy**2)
            displacements.append(displacement_magnitude)

    audit_results['verification']['mean_displacement'] = float(np.mean(displacements)) if displacements else 0.0
    audit_results['verification']['median_displacement'] = float(np.median(displacements)) if displacements else 0.0

    # Overall verification
    audit_results['verification']['overall_success'] = bool(
        audit_results['verification']['count_match'] and
        audit_results['direction_accuracy']['match'] and
        audit_results['stationary_control']['match'] and
        len(audit_results['discrepancies']) == 0
    )

    return audit_results

def main():
    """Run evaluation audit."""
    print("=" * 80)
    print("STAGE 3.2.9K - EVALUATION AUDIT")
    print("=" * 80)

    audit_results = audit_evaluation_results()

    print("\nVerification Results:")
    print(f"  Total videos: {audit_results['verification']['total_videos']} (expected: {audit_results['verification']['expected_total']})")
    print(f"  Count match: {audit_results['verification']['count_match']}")
    print(f"  Direction accuracy match: {audit_results['direction_accuracy']['match']}")
    print(f"  Stationary drift match: {audit_results['stationary_control']['match']}")
    print(f"  Overall success: {audit_results['verification']['overall_success']}")

    print("\nDirection Accuracy:")
    print(f"  Reported: {audit_results['direction_accuracy']['reported']['accuracy']:.4f}")
    print(f"  Recalculated: {audit_results['direction_accuracy']['recalculated']['accuracy']:.4f}")
    print(f"  Correct/Total: {audit_results['direction_accuracy']['recalculated']['correct_count']}/{audit_results['direction_accuracy']['recalculated']['total_count']}")

    print("\nPer-Experiment Accuracy:")
    for config in ['B', 'C', 'D']:
        acc = audit_results['direction_accuracy'][f'{config}_accuracy']
        print(f"  {config}: {acc['accuracy']:.3f} ({acc['correct']}/{acc['total']})")

    print("\nPer-Direction Accuracy:")
    for prompt_idx in range(4):
        acc = audit_results['direction_accuracy'][f'prompt_{prompt_idx}_accuracy']
        print(f"  Prompt {prompt_idx}: {acc['accuracy']:.3f} ({acc['correct']}/{acc['total']})")

    print("\nPer-Seed Accuracy:")
    for seed in [42, 123, 456]:
        acc = audit_results['direction_accuracy'][f'seed_{seed}_accuracy']
        print(f"  Seed {seed}: {acc['accuracy']:.3f} ({acc['correct']}/{acc['total']})")

    print("\nStationary Control:")
    print(f"  Reported drift: {audit_results['stationary_control']['reported']['mean_drift']:.2f} pixels")
    print(f"  Recalculated drift: {audit_results['stationary_control']['recalculated']['mean_drift']:.2f} pixels")
    print(f"  Threshold (5px): {audit_results['stationary_control']['threshold_verification']['stationary_below_threshold']}/{audit_results['stationary_control']['threshold_verification']['total_stationary']} below threshold")

    print("\nDisplacement Statistics:")
    print(f"  Mean: {audit_results['verification']['mean_displacement']:.2f} pixels")
    print(f"  Median: {audit_results['verification']['median_displacement']:.2f} pixels")

    if audit_results['discrepancies']:
        print("\nDiscrepancies:")
        for d in audit_results['discrepancies']:
            print(f"  - {d['issue']}: {d}")
    else:
        print("\nNo discrepancies found.")

    # Save results
    output_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2K_motion_conditioning_forensics")

    with open(output_dir / "STAGE3_2_9K_EVALUATION_AUDIT.json", 'w') as f:
        json.dump(audit_results, f, indent=2)

    with open(output_dir / "STAGE3_2_9K_EVALUATION_AUDIT.md", 'w') as f:
        f.write("# Stage 3.2.9K - Evaluation Audit Report\n\n")
        f.write("**Date**: 2026-11-03\n\n")
        f.write("## Verification Results\n\n")
        f.write(f"- Total videos: {audit_results['verification']['total_videos']} (expected: {audit_results['verification']['expected_total']})\n")
        f.write(f"- Count match: {audit_results['verification']['count_match']}\n")
        f.write(f"- Direction accuracy match: {audit_results['direction_accuracy']['match']}\n")
        f.write(f"- Stationary drift match: {audit_results['stationary_control']['match']}\n")
        f.write(f"- Overall success: {audit_results['verification']['overall_success']}\n\n")

        f.write("## Direction Accuracy\n\n")
        f.write(f"- Reported: {audit_results['direction_accuracy']['reported']['accuracy']:.4f}\n")
        f.write(f"- Recalculated: {audit_results['direction_accuracy']['recalculated']['accuracy']:.4f}\n")
        f.write(f"- Correct/Total: {audit_results['direction_accuracy']['recalculated']['correct_count']}/{audit_results['direction_accuracy']['recalculated']['total_count']}\n\n")

        f.write("## Per-Experiment Accuracy\n\n")
        for config in ['B', 'C', 'D']:
            acc = audit_results['direction_accuracy'][f'{config}_accuracy']
            f.write(f"- {config}: {acc['accuracy']:.3f} ({acc['correct']}/{acc['total']})\n")
        f.write("\n")

        f.write("## Per-Direction Accuracy\n\n")
        for prompt_idx in range(4):
            acc = audit_results['direction_accuracy'][f'prompt_{prompt_idx}_accuracy']
            f.write(f"- Prompt {prompt_idx}: {acc['accuracy']:.3f} ({acc['correct']}/{acc['total']})\n")
        f.write("\n")

        f.write("## Per-Seed Accuracy\n\n")
        for seed in [42, 123, 456]:
            acc = audit_results['direction_accuracy'][f'seed_{seed}_accuracy']
            f.write(f"- Seed {seed}: {acc['accuracy']:.3f} ({acc['correct']}/{acc['total']})\n")
        f.write("\n")

        f.write("## Stationary Control\n\n")
        f.write(f"- Reported drift: {audit_results['stationary_control']['reported']['mean_drift']:.2f} pixels\n")
        f.write(f"- Recalculated drift: {audit_results['stationary_control']['recalculated']['mean_drift']:.2f} pixels\n")
        f.write(f"- Threshold (5px): {audit_results['stationary_control']['threshold_verification']['stationary_below_threshold']}/{audit_results['stationary_control']['threshold_verification']['total_stationary']} below threshold\n\n")

        f.write("## Displacement Statistics\n\n")
        f.write(f"- Mean: {audit_results['verification']['mean_displacement']:.2f} pixels\n")
        f.write(f"- Median: {audit_results['verification']['median_displacement']:.2f} pixels\n\n")

        if audit_results['discrepancies']:
            f.write("## Discrepancies\n\n")
            for d in audit_results['discrepancies']:
                f.write(f"- {d['issue']}: {d}\n")
            f.write("\n")
        else:
            f.write("## Discrepancies\n\n")
            f.write("No discrepancies found.\n\n")

        f.write("## Conclusion\n\n")
        if audit_results['verification']['overall_success']:
            f.write("All metric calculations verified successfully. No discrepancies found.\n")
        else:
            f.write("Discrepancies detected. See above for details.\n")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_dir}")

    return audit_results

if __name__ == "__main__":
    main()
