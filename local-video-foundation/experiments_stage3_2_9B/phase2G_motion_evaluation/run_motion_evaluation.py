"""
Stage 3.2.9F - Run Motion Evaluation

Run quantitative motion evaluation on all generated comparison videos.

Author: Devin
Date: 2026-11-02
"""

import json
from pathlib import Path
from typing import Dict, Any
from motion_evaluator import MotionEvaluator


def parse_direction_from_prompt(prompt: str) -> str:
    """Extract expected direction from prompt."""
    prompt_lower = prompt.lower()
    
    if 'left' in prompt_lower:
        return 'left'
    elif 'right' in prompt_lower:
        return 'right'
    elif 'up' in prompt_lower or 'upward' in prompt_lower:
        return 'up'
    elif 'down' in prompt_lower or 'downward' in prompt_lower:
        return 'down'
    elif 'stationary' in prompt_lower:
        return 'stationary'
    else:
        return 'unknown'


def main():
    """Main evaluation function."""
    base_dir = Path(__file__).parent
    output_dir = base_dir / "comparison_outputs"
    results_path = output_dir / "comparison_results.json"
    
    # Load comparison results
    with open(results_path, 'r') as f:
        comparison_results = json.load(f)
    
    # Initialize evaluator
    evaluator = MotionEvaluator()
    
    # Evaluate all videos
    all_evaluations = {}
    
    for sample_key, sample_data in comparison_results.items():
        print(f"\nEvaluating: {sample_key}")
        
        # Parse prompt and seed from key
        # Format: "Red_object_moving_left_seed42"
        parts = sample_key.rsplit('_seed', 1)
        prompt = parts[0].replace('_', ' ')
        seed = int(parts[1])
        
        expected_direction = parse_direction_from_prompt(prompt)
        
        sample_evaluations = {}
        
        for exp_type, exp_data in sample_data.items():
            frames_dir = Path(exp_data['frames_dir'])
            
            print(f"  Experiment {exp_type}: evaluating...")
            
            evaluation = evaluator.evaluate_video(frames_dir, expected_direction)
            sample_evaluations[exp_type] = evaluation
        
        all_evaluations[sample_key] = {
            'prompt': prompt,
            'seed': seed,
            'expected_direction': expected_direction,
            'evaluations': sample_evaluations
        }
    
    # Save evaluation results
    eval_path = output_dir / "motion_evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(all_evaluations, f, indent=2)
    
    print(f"\n{'='*80}")
    print("MOTION EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to {eval_path}")
    
    # Generate summary statistics
    generate_summary_statistics(all_evaluations, output_dir)


def generate_summary_statistics(evaluations: Dict[str, Any], output_dir: Path):
    """Generate summary statistics across all experiments."""
    
    summary = {
        'experiment_comparison': {},
        'by_direction': {},
        'overall_statistics': {}
    }
    
    # Initialize per-experiment statistics
    for exp_type in ['B', 'C', 'D']:
        summary['experiment_comparison'][exp_type] = {
            'direction_accuracy_sum': 0,
            'direction_accuracy_count': 0,
            'detection_rate_sum': 0,
            'detection_rate_count': 0,
            'avg_displacement_sum': 0,
            'avg_displacement_count': 0,
            'smoothness_sum': 0,
            'smoothness_count': 0
        }
    
    # Initialize per-direction statistics
    directions = ['left', 'right', 'up', 'down', 'stationary']
    for direction in directions:
        summary['by_direction'][direction] = {
            'B': {'direction_accuracy_sum': 0, 'count': 0},
            'C': {'direction_accuracy_sum': 0, 'count': 0},
            'D': {'direction_accuracy_sum': 0, 'count': 0}
        }
    
    # Aggregate statistics
    for sample_key, sample_data in evaluations.items():
        expected_direction = sample_data['expected_direction']
        
        for exp_type in ['B', 'C', 'D']:
            if exp_type not in sample_data['evaluations']:
                continue
            
            eval_data = sample_data['evaluations'][exp_type]
            
            if not eval_data.get('valid', False):
                continue
            
            # Overall statistics
            summary['experiment_comparison'][exp_type]['direction_accuracy_sum'] += \
                eval_data['direction']['direction_accuracy']
            summary['experiment_comparison'][exp_type]['direction_accuracy_count'] += 1
            
            summary['experiment_comparison'][exp_type]['detection_rate_sum'] += \
                eval_data['tracking']['detection_rate']
            summary['experiment_comparison'][exp_type]['detection_rate_count'] += 1
            
            summary['experiment_comparison'][exp_type]['avg_displacement_sum'] += \
                eval_data['speed']['avg_pixel_displacement']
            summary['experiment_comparison'][exp_type]['avg_displacement_count'] += 1
            
            summary['experiment_comparison'][exp_type]['smoothness_sum'] += \
                eval_data['temporal']['smoothness']
            summary['experiment_comparison'][exp_type]['smoothness_count'] += 1
            
            # Per-direction statistics
            if expected_direction in summary['by_direction']:
                summary['by_direction'][expected_direction][exp_type]['direction_accuracy_sum'] += \
                    eval_data['direction']['direction_accuracy']
                summary['by_direction'][expected_direction][exp_type]['count'] += 1
    
    # Compute averages
    for exp_type in ['B', 'C', 'D']:
        stats = summary['experiment_comparison'][exp_type]
        
        if stats['direction_accuracy_count'] > 0:
            stats['avg_direction_accuracy'] = \
                stats['direction_accuracy_sum'] / stats['direction_accuracy_count']
        else:
            stats['avg_direction_accuracy'] = 0
        
        if stats['detection_rate_count'] > 0:
            stats['avg_detection_rate'] = \
                stats['detection_rate_sum'] / stats['detection_rate_count']
        else:
            stats['avg_detection_rate'] = 0
        
        if stats['avg_displacement_count'] > 0:
            stats['avg_displacement'] = \
                stats['avg_displacement_sum'] / stats['avg_displacement_count']
        else:
            stats['avg_displacement'] = 0
        
        if stats['smoothness_count'] > 0:
            stats['avg_smoothness'] = \
                stats['smoothness_sum'] / stats['smoothness_count']
        else:
            stats['avg_smoothness'] = 0
    
    # Compute per-direction averages
    for direction in directions:
        for exp_type in ['B', 'C', 'D']:
            dir_stats = summary['by_direction'][direction][exp_type]
            if dir_stats['count'] > 0:
                dir_stats['avg_direction_accuracy'] = \
                    dir_stats['direction_accuracy_sum'] / dir_stats['count']
            else:
                dir_stats['avg_direction_accuracy'] = 0
    
    # Save summary
    summary_path = output_dir / "motion_evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary statistics saved to {summary_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    for exp_type in ['B', 'C', 'D']:
        stats = summary['experiment_comparison'][exp_type]
        print(f"\nExperiment {exp_type}:")
        print(f"  Avg Direction Accuracy: {stats['avg_direction_accuracy']:.3f}")
        print(f"  Avg Detection Rate: {stats['avg_detection_rate']:.3f}")
        print(f"  Avg Displacement: {stats['avg_displacement']:.3f} pixels")
        print(f"  Avg Smoothness: {stats['avg_smoothness']:.3f}")


if __name__ == "__main__":
    main()
