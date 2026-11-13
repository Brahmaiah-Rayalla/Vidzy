"""
Stage 3.2.9J - Quantitative Motion Evaluation

Performs object tracking, direction accuracy, motion magnitude, and quality evaluation.

Author: Devin
Date: 2026-11-03
"""

import json
import numpy as np
from pathlib import Path
try:
    import imageio
except ImportError:
    imageio = None
from scipy import ndimage

def track_object(video):
    """
    Track the main object in a video using simple centroid tracking.
    Assumes the object is the brightest/most prominent region.
    """
    if imageio is None:
        return None

    frames = np.array(video)  # [T, H, W, C]

    # Convert to grayscale for tracking
    if frames.shape[-1] == 3:
        gray = frames.mean(axis=-1)  # [T, H, W]
    else:
        gray = frames

    # Threshold to find object (assuming object is brighter than background)
    # Use adaptive threshold based on frame statistics
    threshold = gray.mean() + 0.5 * gray.std()

    trajectories = []
    for t in range(len(frames)):
        frame = gray[t]

        # Create binary mask
        mask = frame > threshold

        if mask.sum() == 0:
            # No object detected, use previous position or center
            if trajectories:
                trajectories.append(trajectories[-1])
            else:
                trajectories.append((32, 32))  # Center of 64x64
            continue

        # Find centroid
        labeled, num_features = ndimage.label(mask)

        if num_features == 0:
            if trajectories:
                trajectories.append(trajectories[-1])
            else:
                trajectories.append((32, 32))
            continue

        # Find largest connected component (the object)
        sizes = ndimage.sum(mask, labeled, range(num_features + 1))
        max_label = sizes[1:].argmax() + 1

        # Get centroid of largest component
        coords = ndimage.center_of_mass(mask, labeled, max_label)
        trajectories.append(coords)

    return np.array(trajectories)  # [T, 2] - (y, x) coordinates

def calculate_displacement(trajectory):
    """Calculate frame-to-frame displacements."""
    if len(trajectory) < 2:
        return np.array([])

    displacements = np.diff(trajectory, axis=0)  # [T-1, 2] - (dy, dx)
    return displacements

def calculate_metrics(trajectory, prompt, prompt_idx):
    """Calculate motion metrics for a video."""
    if trajectory is None or len(trajectory) < 2:
        return None

    displacements = calculate_displacement(trajectory)

    # Total displacement (net)
    total_displacement = trajectory[-1] - trajectory[0]  # (dy, dx)

    # Trajectory length (sum of absolute frame-to-frame displacements)
    trajectory_length = np.abs(displacements).sum(axis=0)  # (total_dy, total_dx)

    # Average displacement per frame
    avg_displacement = displacements.mean(axis=0)  # (avg_dy, avg_dx)

    # Direction consistency (sign agreement ratio)
    if len(displacements) > 0:
        x_signs = np.sign(displacements[:, 1])
        y_signs = np.sign(displacements[:, 0])

        # Ignore zero displacements
        x_signs = x_signs[x_signs != 0]
        y_signs = y_signs[y_signs != 0]

        x_consistency = 0.0
        y_consistency = 0.0

        if len(x_signs) > 0:
            x_consistency = (x_signs == x_signs[0]).sum() / len(x_signs)
        if len(y_signs) > 0:
            y_consistency = (y_signs == y_signs[0]).sum() / len(y_signs)
    else:
        x_consistency = 0.0
        y_consistency = 0.0

    # Expected direction based on prompt
    expected_dx = None
    expected_dy = None

    if prompt_idx == 0:  # "A red object moving left"
        expected_dx = -1  # Negative X (left in image coordinates)
        expected_dy = 0
    elif prompt_idx == 1:  # "A red object moving right"
        expected_dx = 1  # Positive X (right in image coordinates)
        expected_dy = 0
    elif prompt_idx == 2:  # "A blue object moving diagonally"
        expected_dx = 1  # Positive X
        expected_dy = 1  # Positive Y (down in image coordinates)
    elif prompt_idx == 3:  # "A yellow object remaining stationary"
        expected_dx = 0
        expected_dy = 0

    # Direction accuracy
    direction_accuracy = 0.0
    if expected_dx is not None and expected_dy is not None:
        if expected_dx == 0 and expected_dy == 0:
            # Stationary: check if total displacement is small
            displacement_magnitude = np.sqrt(total_displacement[0]**2 + total_displacement[1]**2)
            direction_accuracy = 1.0 if displacement_magnitude < 5.0 else 0.0
        else:
            # Moving: check if direction matches
            actual_dx_sign = np.sign(total_displacement[1])
            actual_dy_sign = np.sign(total_displacement[0])

            dx_match = (actual_dx_sign == expected_dx) if expected_dx != 0 else True
            dy_match = (actual_dy_sign == expected_dy) if expected_dy != 0 else True

            direction_accuracy = 1.0 if (dx_match and dy_match) else 0.0

    return {
        'trajectory': trajectory.tolist(),
        'total_displacement': total_displacement.tolist(),  # [dy, dx]
        'trajectory_length': trajectory_length.tolist(),  # [total_dy, total_dx]
        'avg_displacement': avg_displacement.tolist(),  # [avg_dy, avg_dx]
        'x_consistency': float(x_consistency),
        'y_consistency': float(y_consistency),
        'direction_accuracy': float(direction_accuracy),
        'expected_dx': expected_dx,
        'expected_dy': expected_dy
    }

def evaluate_appearance_quality(video):
    """Evaluate appearance quality metrics."""
    if imageio is None:
        return None

    frames = np.array(video).astype(float) / 255.0  # Normalize to [0, 1]

    # Color consistency (std of mean color across frames)
    mean_colors = frames.mean(axis=(1, 2))  # [T, 3]
    color_consistency = 1.0 - mean_colors.std(axis=0).mean()  # Higher is better

    # Object visibility (contrast between object and background)
    # Use std as proxy for visibility
    frame_stds = frames.std(axis=(1, 2))  # [T, 3]
    visibility = frame_stds.mean()

    # Size variation (std of frame means - proxy for object size)
    frame_means = frames.mean(axis=(1, 2, 3))  # [T]
    size_variation = frame_means.std()

    # Background stability (std of frame-to-frame differences)
    frame_diffs = np.diff(frames, axis=0)
    background_stability = 1.0 - frame_diffs.std()  # Higher is better

    return {
        'color_consistency': float(color_consistency),
        'visibility': float(visibility),
        'size_variation': float(size_variation),
        'background_stability': float(background_stability)
    }

def evaluate_temporal_quality(video):
    """Evaluate temporal quality metrics."""
    if imageio is None:
        return None

    frames = np.array(video).astype(float)

    # Frame-to-frame flicker (std of differences)
    frame_diffs = np.diff(frames, axis=0)
    flicker = frame_diffs.std()

    # Motion smoothness (variance of frame-to-frame differences)
    motion_smoothness = 1.0 - frame_diffs.var() / (frames.var() + 1e-8)

    # Unexpected frame variation (high-frequency changes)
    # Use second derivative
    frame_accels = np.diff(frame_diffs, axis=0)
    unexpected_variation = frame_accels.std()

    return {
        'flicker': float(flicker),
        'motion_smoothness': float(motion_smoothness),
        'unexpected_variation': float(unexpected_variation)
    }

def evaluate_all_videos():
    """Evaluate all videos."""
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2J_corrected_motion_evaluation/outputs_evaluation")

    prompts = [
        "A red object moving left",
        "A red object moving right",
        "A blue object moving diagonally",
        "A yellow object remaining stationary"
    ]

    configs = ['B', 'C', 'D']
    seeds = [42, 123, 456]

    results = {
        'direction_accuracy': {},
        'motion_metrics': {},
        'stationary_control': {},
        'appearance_evaluation': {},
        'temporal_comparison': {}
    }

    for config in configs:
        for seed in seeds:
            for prompt_idx, prompt in enumerate(prompts):
                video_path = base_dir / config / f"seed_{seed}" / f"prompt_{prompt_idx:02d}" / "video.gif"
                key = f"{config}_seed{seed}_prompt{prompt_idx}"

                if not video_path.exists():
                    continue

                try:
                    # Load video
                    video = imageio.mimread(video_path)

                    # Track object
                    trajectory = track_object(video)

                    # Calculate motion metrics
                    motion_metrics = calculate_metrics(trajectory, prompt, prompt_idx)

                    # Evaluate appearance quality
                    appearance_metrics = evaluate_appearance_quality(video)

                    # Evaluate temporal quality
                    temporal_metrics = evaluate_temporal_quality(video)

                    # Store results
                    results['motion_metrics'][key] = {
                        'config': config,
                        'seed': seed,
                        'prompt_idx': prompt_idx,
                        'prompt': prompt,
                        **motion_metrics
                    }

                    results['appearance_evaluation'][key] = {
                        'config': config,
                        'seed': seed,
                        'prompt_idx': prompt_idx,
                        'prompt': prompt,
                        **appearance_metrics
                    }

                    results['temporal_comparison'][key] = {
                        'config': config,
                        'seed': seed,
                        'prompt_idx': prompt_idx,
                        'prompt': prompt,
                        **temporal_metrics
                    }

                    # Direction accuracy
                    if motion_metrics:
                        results['direction_accuracy'][key] = {
                            'config': config,
                            'seed': seed,
                            'prompt_idx': prompt_idx,
                            'prompt': prompt,
                            'accuracy': motion_metrics['direction_accuracy'],
                            'expected_dx': motion_metrics['expected_dx'],
                            'expected_dy': motion_metrics['expected_dy'],
                            'actual_dx': motion_metrics['total_displacement'][1],
                            'actual_dy': motion_metrics['total_displacement'][0]
                        }

                    # Stationary control (for prompt 3)
                    if prompt_idx == 3 and motion_metrics:
                        displacement_magnitude = np.sqrt(
                            motion_metrics['total_displacement'][0]**2 +
                            motion_metrics['total_displacement'][1]**2
                        )
                        results['stationary_control'][key] = {
                            'config': config,
                            'seed': seed,
                            'prompt': prompt,
                            'displacement_magnitude': float(displacement_magnitude),
                            'drift': float(displacement_magnitude),
                            'stationary': bool(displacement_magnitude < 5.0)
                        }

                except Exception as e:
                    print(f"Error processing {key}: {e}")
                    results['motion_metrics'][key] = {
                        'config': config,
                        'seed': seed,
                        'prompt_idx': prompt_idx,
                        'prompt': prompt,
                        'error': str(e)
                    }

    return results

def compute_statistics(metrics_dict, metric_name):
    """Compute statistics for a metric across all samples."""
    values = []
    for key, data in metrics_dict.items():
        if metric_name in data and isinstance(data[metric_name], (int, float)):
            values.append(data[metric_name])

    if not values:
        return None

    values = np.array(values)

    return {
        'count': len(values),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'std': float(values.std()),
        'min': float(values.min()),
        'max': float(values.max())
    }

def main():
    """Run motion evaluation."""
    print("=" * 80)
    print("STAGE 3.2.9J - QUANTITATIVE MOTION EVALUATION")
    print("=" * 80)

    if imageio is None:
        print("ERROR: imageio not available. Cannot proceed with evaluation.")
        return

    results = evaluate_all_videos()

    # Compute statistics
    print("\nComputing statistics...")

    # Direction accuracy statistics
    dir_stats = compute_statistics(results['direction_accuracy'], 'accuracy')
    print(f"Direction accuracy: {dir_stats}")

    # Motion magnitude statistics
    motion_stats = {}
    for metric in ['x_consistency', 'y_consistency']:
        stats = compute_statistics(results['motion_metrics'], metric)
        motion_stats[metric] = stats
        print(f"{metric}: {stats}")

    # Appearance statistics
    appearance_stats = {}
    for metric in ['color_consistency', 'visibility', 'size_variation', 'background_stability']:
        stats = compute_statistics(results['appearance_evaluation'], metric)
        appearance_stats[metric] = stats
        print(f"{metric}: {stats}")

    # Temporal statistics
    temporal_stats = {}
    for metric in ['flicker', 'motion_smoothness', 'unexpected_variation']:
        stats = compute_statistics(results['temporal_comparison'], metric)
        temporal_stats[metric] = stats
        print(f"{metric}: {stats}")

    # Stationary statistics
    stationary_stats = compute_statistics(results['stationary_control'], 'displacement_magnitude')
    print(f"Stationary drift: {stationary_stats}")

    # Save results
    output_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2J_motion_analysis")

    with open(output_dir / "STAGE3_2_9J_DIRECTION_ACCURACY.json", 'w') as f:
        json.dump(results['direction_accuracy'], f, indent=2)

    with open(output_dir / "STAGE3_2_9J_MOTION_METRICS.json", 'w') as f:
        json.dump(results['motion_metrics'], f, indent=2)

    with open(output_dir / "STAGE3_2_9J_STATIONARY_CONTROL.json", 'w') as f:
        json.dump(results['stationary_control'], f, indent=2)

    with open(output_dir / "STAGE3_2_9J_APPEARANCE_EVALUATION.json", 'w') as f:
        json.dump(results['appearance_evaluation'], f, indent=2)

    with open(output_dir / "STAGE3_2_9J_TEMPORAL_COMPARISON.json", 'w') as f:
        json.dump(results['temporal_comparison'], f, indent=2)

    # Save statistics
    stats_summary = {
        'direction_accuracy': dir_stats,
        'motion_metrics': motion_stats,
        'appearance_metrics': appearance_stats,
        'temporal_metrics': temporal_stats,
        'stationary_control': stationary_stats
    }

    with open(output_dir / "STAGE3_2_9J_STATISTICS.json", 'w') as f:
        json.dump(stats_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_dir}")

    return results

if __name__ == "__main__":
    main()
