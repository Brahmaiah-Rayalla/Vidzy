"""
Generate comprehensive analysis reports for Stage 3.2.9L.3.

Generates:
- Parameter updates analysis
- Gradient monitoring analysis
- Loss comparison analysis
- CPU performance analysis
"""

import json
import numpy as np
from pathlib import Path

def analyze_training_results():
    """Analyze training results and generate reports."""
    print("=" * 80)
    print("STAGE 3.2.9L.3 - GENERATING ANALYSIS REPORTS")
    print("=" * 80)

    output_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2L_training_repair/controlled_training")

    # Load training results
    with open(output_dir / "STAGE3_2_9L3_TRAINING_RESULTS.json", 'r') as f:
        training_results = json.load(f)

    # Load checkpoint integrity results
    with open(output_dir / "STAGE3_2_9L3_CHECKPOINT_INTEGRITY.json", 'r') as f:
        checkpoint_integrity = json.load(f)

    results_B = training_results['experiment_B']['steps']
    results_C = training_results['experiment_C']['steps']

    # ============================================================================
    # PARAMETER UPDATES ANALYSIS
    # ============================================================================
    print("\nGenerating parameter updates analysis...")

    parameter_updates = {
        'experiment_B': {},
        'experiment_C': {},
        'comparison': {}
    }

    # Experiment B parameter updates
    motion_deltas_B = [float(step.get('motion_delta', {}).get('max', 0)) for step in results_B]
    parameter_updates['experiment_B'] = {
        'final_motion_delta_max': float(results_B[-1].get('motion_delta', {}).get('max', 0)),
        'final_motion_delta_mean': float(results_B[-1].get('motion_delta', {}).get('mean', 0)),
        'motion_delta_progression': motion_deltas_B[-10:]  # Last 10 steps
    }

    # Experiment C parameter updates
    motion_deltas_C = [float(step.get('motion_delta', {}).get('max', 0)) for step in results_C]
    temporal_deltas_C = [float(step.get('temporal_delta', {}).get('max', 0)) for step in results_C]
    fusion_deltas_C = [float(step.get('fusion_delta', {}).get('max', 0)) for step in results_C]

    parameter_updates['experiment_C'] = {
        'final_motion_delta_max': float(results_C[-1].get('motion_delta', {}).get('max', 0)),
        'final_motion_delta_mean': float(results_C[-1].get('motion_delta', {}).get('mean', 0)),
        'final_temporal_delta_max': float(results_C[-1].get('temporal_delta', {}).get('max', 0)),
        'final_temporal_delta_mean': float(results_C[-1].get('temporal_delta', {}).get('mean', 0)),
        'final_fusion_delta_max': float(results_C[-1].get('fusion_delta', {}).get('max', 0)),
        'final_fusion_delta_mean': float(results_C[-1].get('fusion_delta', {}).get('mean', 0)),
        'motion_delta_progression': motion_deltas_C[-10:],
        'temporal_delta_progression': temporal_deltas_C[-10:],
        'fusion_delta_progression': fusion_deltas_C[-10:]
    }

    # Comparison
    parameter_updates['comparison'] = {
        'motion_delta_difference': float(abs(parameter_updates['experiment_B']['final_motion_delta_max'] - parameter_updates['experiment_C']['final_motion_delta_max']))
    }

    with open(output_dir / "STAGE3_2_9L3_PARAMETER_UPDATES.json", 'w') as f:
        json.dump(parameter_updates, f, indent=2)

    print("  Saved: STAGE3_2_9L3_PARAMETER_UPDATES.json")

    # ============================================================================
    # GRADIENT MONITORING ANALYSIS
    # ============================================================================
    print("\nGenerating gradient monitoring analysis...")

    gradient_monitoring = {
        'experiment_B': {},
        'experiment_C': {},
        'comparison': {}
    }

    # Experiment B gradients
    motion_grads_B = [step['motion_grad_norm'] for step in results_B]
    gradient_monitoring['experiment_B'] = {
        'mean_motion_grad_norm': float(np.mean(motion_grads_B)),
        'std_motion_grad_norm': float(np.std(motion_grads_B)),
        'min_motion_grad_norm': float(np.min(motion_grads_B)),
        'max_motion_grad_norm': float(np.max(motion_grads_B)),
        'final_motion_grad_norm': float(motion_grads_B[-1]),
        'gradient_progression': motion_grads_B[-10:]
    }

    # Experiment C gradients
    motion_grads_C = [step['motion_grad_norm'] for step in results_C]
    temporal_grads_C = [step['temporal_grad_norm'] for step in results_C]
    fusion_grads_C = [step['fusion_grad_norm'] for step in results_C]

    gradient_monitoring['experiment_C'] = {
        'mean_motion_grad_norm': float(np.mean(motion_grads_C)),
        'std_motion_grad_norm': float(np.std(motion_grads_C)),
        'min_motion_grad_norm': float(np.min(motion_grads_C)),
        'max_motion_grad_norm': float(np.max(motion_grads_C)),
        'final_motion_grad_norm': float(motion_grads_C[-1]),
        'mean_temporal_grad_norm': float(np.mean(temporal_grads_C)),
        'std_temporal_grad_norm': float(np.std(temporal_grads_C)),
        'min_temporal_grad_norm': float(np.min(temporal_grads_C)),
        'max_temporal_grad_norm': float(np.max(temporal_grads_C)),
        'final_temporal_grad_norm': float(temporal_grads_C[-1]),
        'mean_fusion_grad_norm': float(np.mean(fusion_grads_C)),
        'std_fusion_grad_norm': float(np.std(fusion_grads_C)),
        'min_fusion_grad_norm': float(np.min(fusion_grads_C)),
        'max_fusion_grad_norm': float(np.max(fusion_grads_C)),
        'final_fusion_grad_norm': float(fusion_grads_C[-1]),
        'motion_grad_progression': motion_grads_C[-10:],
        'temporal_grad_progression': temporal_grads_C[-10:],
        'fusion_grad_progression': fusion_grads_C[-10:]
    }

    # Comparison
    gradient_monitoring['comparison'] = {
        'motion_grad_norm_ratio': float(gradient_monitoring['experiment_C']['mean_motion_grad_norm'] / gradient_monitoring['experiment_B']['mean_motion_grad_norm'])
    }

    with open(output_dir / "STAGE3_2_9L3_GRADIENT_MONITORING.json", 'w') as f:
        json.dump(gradient_monitoring, f, indent=2)

    print("  Saved: STAGE3_2_9L3_GRADIENT_MONITORING.json")

    # ============================================================================
    # LOSS COMPARISON ANALYSIS
    # ============================================================================
    print("\nGenerating loss comparison analysis...")

    loss_comparison = {
        'experiment_B': {},
        'experiment_C': {},
        'comparison': {}
    }

    # Experiment B losses
    losses_B = [step['loss'] for step in results_B]
    loss_comparison['experiment_B'] = {
        'mean_loss': float(np.mean(losses_B)),
        'std_loss': float(np.std(losses_B)),
        'min_loss': float(np.min(losses_B)),
        'max_loss': float(np.max(losses_B)),
        'initial_loss': float(losses_B[0]),
        'final_loss': float(losses_B[-1]),
        'loss_change': float(losses_B[-1] - losses_B[0]),
        'loss_progression': losses_B[-10:]
    }

    # Experiment C losses
    losses_C = [step['loss'] for step in results_C]
    loss_comparison['experiment_C'] = {
        'mean_loss': float(np.mean(losses_C)),
        'std_loss': float(np.std(losses_C)),
        'min_loss': float(np.min(losses_C)),
        'max_loss': float(np.max(losses_C)),
        'initial_loss': float(losses_C[0]),
        'final_loss': float(losses_C[-1]),
        'loss_change': float(losses_C[-1] - losses_C[0]),
        'loss_progression': losses_C[-10:]
    }

    # Comparison
    loss_comparison['comparison'] = {
        'initial_loss_difference': float(abs(losses_B[0] - losses_C[0])),
        'final_loss_difference': float(abs(losses_B[-1] - losses_C[-1])),
        'mean_loss_ratio': float(loss_comparison['experiment_C']['mean_loss'] / loss_comparison['experiment_B']['mean_loss'])
    }

    with open(output_dir / "STAGE3_2_9L3_LOSS_COMPARISON.json", 'w') as f:
        json.dump(loss_comparison, f, indent=2)

    print("  Saved: STAGE3_2_9L3_LOSS_COMPARISON.json")

    # ============================================================================
    # CPU PERFORMANCE ANALYSIS
    # ============================================================================
    print("\nGenerating CPU performance analysis...")

    cpu_performance = {
        'experiment_B': {},
        'experiment_C': {},
        'comparison': {}
    }

    # Experiment B performance
    step_times_B = [step['step_time'] for step in results_B]
    ram_usage_B = [step['peak_ram_mb'] for step in results_B]

    cpu_performance['experiment_B'] = {
        'total_time': float(sum(step_times_B)),
        'mean_step_time': float(np.mean(step_times_B)),
        'median_step_time': float(np.median(step_times_B)),
        'p90_step_time': float(np.percentile(step_times_B, 90)),
        'max_step_time': float(np.max(step_times_B)),
        'min_step_time': float(np.min(step_times_B)),
        'outlier_count': int(sum(1 for t in step_times_B if t > np.mean(step_times_B) + 2 * np.std(step_times_B))),
        'peak_ram_mb': float(np.max(ram_usage_B)),
        'mean_ram_mb': float(np.mean(ram_usage_B)),
        'cpu_threads': training_results['config']['cpu_threads']
    }

    # Experiment C performance
    step_times_C = [step['step_time'] for step in results_C]
    ram_usage_C = [step['peak_ram_mb'] for step in results_C]

    cpu_performance['experiment_C'] = {
        'total_time': float(sum(step_times_C)),
        'mean_step_time': float(np.mean(step_times_C)),
        'median_step_time': float(np.median(step_times_C)),
        'p90_step_time': float(np.percentile(step_times_C, 90)),
        'max_step_time': float(np.max(step_times_C)),
        'min_step_time': float(np.min(step_times_C)),
        'outlier_count': int(sum(1 for t in step_times_C if t > np.mean(step_times_C) + 2 * np.std(step_times_C))),
        'peak_ram_mb': float(np.max(ram_usage_C)),
        'mean_ram_mb': float(np.mean(ram_usage_C)),
        'cpu_threads': training_results['config']['cpu_threads']
    }

    # Comparison
    cpu_performance['comparison'] = {
        'time_ratio': float(cpu_performance['experiment_C']['mean_step_time'] / cpu_performance['experiment_B']['mean_step_time']),
        'ram_ratio': float(cpu_performance['experiment_C']['mean_ram_mb'] / cpu_performance['experiment_B']['mean_ram_mb'])
    }

    with open(output_dir / "STAGE3_2_9L3_CPU_PERFORMANCE.json", 'w') as f:
        json.dump(cpu_performance, f, indent=2)

    print("  Saved: STAGE3_2_9L3_CPU_PERFORMANCE.json")

    # ============================================================================
    # MOTION EVALUATION (Placeholder - will be filled with actual evaluation)
    # ============================================================================
    print("\nGenerating motion evaluation report...")

    motion_evaluation = {
        'status': 'NOT_EVALUATED',
        'note': 'Motion evaluation requires inference pipeline and generated samples. This is a placeholder for future evaluation.',
        'planned_evaluations': [
            'Direction accuracy',
            'Object-centre displacement',
            'Direction consistency',
            'Stationary drift',
            'Object detection rate',
            'Shape and colour preservation'
        ]
    }

    with open(output_dir / "STAGE3_2_9L3_MOTION_EVALUATION.json", 'w') as f:
        json.dump(motion_evaluation, f, indent=2)

    print("  Saved: STAGE3_2_9L3_MOTION_EVALUATION.json (placeholder)")

    print("\n" + "=" * 80)
    print("ANALYSIS REPORTS GENERATED SUCCESSFULLY")
    print("=" * 80)

    return {
        'parameter_updates': parameter_updates,
        'gradient_monitoring': gradient_monitoring,
        'loss_comparison': loss_comparison,
        'cpu_performance': cpu_performance,
        'motion_evaluation': motion_evaluation
    }

if __name__ == "__main__":
    analyze_training_results()
