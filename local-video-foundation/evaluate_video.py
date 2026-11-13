"""
Evaluation metrics for video generation.

Implements metrics to assess the quality and consistency of generated videos.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import json
from typing import List, Dict, Tuple
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_frames(frame_dir: str) -> np.ndarray:
    """
    Load frames from a directory into a numpy array.

    Args:
        frame_dir: Directory containing frame_*.png files

    Returns:
        Array of shape [num_frames, height, width, channels]
    """
    frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
    frames = []

    for frame_file in frame_files:
        img = Image.open(frame_file)
        frames.append(np.array(img))

    return np.stack(frames)  # [frames, h, w, c]


def compute_temporal_consistency(frames: np.ndarray) -> Dict[str, float]:
    """
    Compute temporal consistency metrics.

    Measures how consistent the video is across frames.

    Args:
        frames: Array of shape [num_frames, height, width, channels]

    Returns:
        Dictionary of temporal consistency metrics
    """
    num_frames = frames.shape[0]

    # Compute frame-to-frame differences
    frame_diffs = []
    for i in range(num_frames - 1):
        diff = np.abs(frames[i].astype(float) - frames[i+1].astype(float))
        frame_diffs.append(diff)

    frame_diffs = np.stack(frame_diffs)  # [num_frames-1, h, w, c]

    # Average frame difference (lower is more consistent)
    avg_frame_diff = np.mean(frame_diffs)

    # Maximum frame difference
    max_frame_diff = np.max(frame_diffs)

    # Compute temporal smoothness (standard deviation of differences)
    temporal_smoothness = np.std(frame_diffs)

    return {
        'avg_frame_diff': float(avg_frame_diff),
        'max_frame_diff': float(max_frame_diff),
        'temporal_smoothness': float(temporal_smoothness)
    }


def compute_spatial_quality(frames: np.ndarray) -> Dict[str, float]:
    """
    Compute spatial quality metrics.

    Measures the spatial characteristics of the video.

    Args:
        frames: Array of shape [num_frames, height, width, channels]

    Returns:
        Dictionary of spatial quality metrics
    """
    # Average brightness across all frames
    avg_brightness = np.mean(frames)

    # Contrast (standard deviation of pixel values)
    contrast = np.std(frames)

    # Saturation (average standard deviation across RGB channels)
    saturation = np.mean([np.std(frames[:, :, :, c]) for c in range(3)])

    return {
        'avg_brightness': float(avg_brightness),
        'contrast': float(contrast),
        'saturation': float(saturation)
    }


def compute_motion_metrics(frames: np.ndarray) -> Dict[str, float]:
    """
    Compute motion-related metrics.

    Measures the amount and pattern of motion in the video.

    Args:
        frames: Array of shape [num_frames, height, width, channels]

    Returns:
        Dictionary of motion metrics
    """
    num_frames = frames.shape[0]

    # Compute optical flow approximation using frame differences
    # Simplified: use the magnitude of frame-to-frame differences
    frame_diffs = []
    for i in range(num_frames - 1):
        diff = frames[i].astype(float) - frames[i+1].astype(float)
        diff_magnitude = np.sqrt(np.sum(diff**2, axis=2))  # [h, w]
        frame_diffs.append(diff_magnitude)

    frame_diffs = np.stack(frame_diffs)  # [num_frames-1, h, w]

    # Average motion magnitude
    avg_motion = np.mean(frame_diffs)

    # Maximum motion magnitude
    max_motion = np.max(frame_diffs)

    # Motion consistency (correlation of motion patterns across time)
    if len(frame_diffs) > 1:
        motion_flattened = [diff.flatten() for diff in frame_diffs]
        motion_consistency = np.corrcoef(motion_flattened)
        avg_motion_corr = np.mean(motion_consistency[np.triu_indices_from(motion_consistency, k=1)])
    else:
        avg_motion_corr = 0.0

    return {
        'avg_motion': float(avg_motion),
        'max_motion': float(max_motion),
        'motion_consistency': float(avg_motion_corr)
    }


def compute_diversity(frames: np.ndarray) -> Dict[str, float]:
    """
    Compute diversity metrics.

    Measures how much the video varies across frames.

    Args:
        frames: Array of shape [num_frames, height, width, channels]

    Returns:
        Dictionary of diversity metrics
    """
    num_frames = frames.shape[0]

    # Compute pairwise frame differences
    pairwise_diffs = []
    for i in range(num_frames):
        for j in range(i+1, num_frames):
            diff = np.abs(frames[i].astype(float) - frames[j].astype(float))
            pairwise_diffs.append(np.mean(diff))

    pairwise_diffs = np.array(pairwise_diffs)

    # Average pairwise difference
    avg_pairwise_diff = np.mean(pairwise_diffs)

    # Standard deviation of pairwise differences
    std_pairwise_diff = np.std(pairwise_diffs)

    return {
        'avg_pairwise_diff': float(avg_pairwise_diff),
        'std_pairwise_diff': float(std_pairwise_diff)
    }


def evaluate_video(frame_dir: str, output_file: str = None) -> Dict[str, float]:
    """
    Evaluate a generated video using multiple metrics.

    Args:
        frame_dir: Directory containing frame_*.png files
        output_file: Optional path to save evaluation results

    Returns:
        Dictionary of all evaluation metrics
    """
    print(f"Evaluating video in {frame_dir}...")

    # Load frames
    frames = load_frames(frame_dir)
    print(f"  Loaded {frames.shape[0]} frames of size {frames.shape[1]}x{frames.shape[2]}")

    # Compute metrics
    temporal_metrics = compute_temporal_consistency(frames)
    spatial_metrics = compute_spatial_quality(frames)
    motion_metrics = compute_motion_metrics(frames)
    diversity_metrics = compute_diversity(frames)

    # Combine all metrics
    all_metrics = {
        **temporal_metrics,
        **spatial_metrics,
        **motion_metrics,
        **diversity_metrics
    }

    # Print metrics
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric_name, value in all_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print("=" * 50)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved results to {output_file}")

    return all_metrics


def compare_videos(
    frame_dir1: str,
    frame_dir2: str,
    output_file: str = None
) -> Dict[str, float]:
    """
    Compare two videos and compute similarity metrics.

    Args:
        frame_dir1: Directory containing frames for video 1
        frame_dir2: Directory containing frames for video 2
        output_file: Optional path to save comparison results

    Returns:
        Dictionary of comparison metrics
    """
    print(f"Comparing videos:")
    print(f"  Video 1: {frame_dir1}")
    print(f"  Video 2: {frame_dir2}")

    # Load frames
    frames1 = load_frames(frame_dir1)
    frames2 = load_frames(frame_dir2)

    # Ensure same number of frames
    min_frames = min(frames1.shape[0], frames2.shape[0])
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]

    # Compute frame-wise differences
    frame_diffs = []
    for i in range(min_frames):
        diff = np.abs(frames1[i].astype(float) - frames2[i].astype(float))
        frame_diffs.append(np.mean(diff))

    frame_diffs = np.array(frame_diffs)

    # Metrics
    avg_diff = np.mean(frame_diffs)
    max_diff = np.max(frame_diffs)
    std_diff = np.std(frame_diffs)

    # Structural similarity (simplified)
    ssim_scores = []
    for i in range(min_frames):
        # Simplified SSIM-like metric
        mu1 = np.mean(frames1[i])
        mu2 = np.mean(frames2[i])
        sigma1 = np.std(frames1[i])
        sigma2 = np.std(frames2[i])
        covariance = np.mean((frames1[i] - mu1) * (frames2[i] - mu2))

        # SSIM-like score
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * mu1 * mu2 + c1) * (2 * covariance + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        ssim_scores.append(ssim)

    avg_ssim = np.mean(ssim_scores)

    results = {
        'avg_frame_diff': float(avg_diff),
        'max_frame_diff': float(max_diff),
        'std_frame_diff': float(std_diff),
        'avg_ssim': float(avg_ssim)
    }

    print("\nComparison Results:")
    print("=" * 50)
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")
    print("=" * 50)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated videos using quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single video
  python evaluate_video.py --frame_dir test_ddim_output --output results.json

  # Compare two videos
  python evaluate_video.py --compare --frame_dir1 test_ddim_output --frame_dir2 test_cli_output --output comparison.json
        """
    )

    parser.add_argument(
        "--frame_dir",
        type=str,
        help="Directory containing frame_*.png files"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two videos instead of evaluating one"
    )
    parser.add_argument(
        "--frame_dir1",
        type=str,
        help="First video directory (for comparison)"
    )
    parser.add_argument(
        "--frame_dir2",
        type=str,
        help="Second video directory (for comparison)"
    )

    args = parser.parse_args()

    if args.compare:
        if not args.frame_dir1 or not args.frame_dir2:
            parser.error("--frame_dir1 and --frame_dir2 are required for comparison")
        compare_videos(args.frame_dir1, args.frame_dir2, args.output)
    else:
        if not args.frame_dir:
            parser.error("--frame_dir is required for evaluation")
        evaluate_video(args.frame_dir, args.output)


if __name__ == "__main__":
    main()
