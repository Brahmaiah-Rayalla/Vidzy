"""
Objective temporal evaluation metrics for Stage 3.2.

Measures the quality of generated videos against known synthetic trajectories:
- Object detection across frames
- Object center coordinates per frame
- Motion direction
- Motion magnitude
- Trajectory error
- Object color consistency
- Shape consistency
- Background consistency
- Frame-to-frame movement
- Temporal smoothness

Compares predicted object trajectories against known target trajectories.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_object_center(frame: np.ndarray, target_color: Tuple[int, int, int]) -> Tuple[float, float]:
    """
    Detect the center of an object with a specific color.

    Args:
        frame: RGB image array
        target_color: Target RGB color tuple

    Returns:
        (x, y) center coordinates
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2HSV)

    # Define color range (using a tolerance around the target color)
    target_color_np = np.array(target_color)
    tolerance = 30

    lower_bound = np.clip(target_color_np - tolerance, 0, 255)
    upper_bound = np.clip(target_color_np + tolerance, 0, 255)

    # Create mask
    mask = cv2.inRange(frame, lower_bound, upper_bound)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute center
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    return (cx, cy)


def compute_trajectory(frames: np.ndarray, target_color: Tuple[int, int, int]) -> List[Tuple[float, float]]:
    """
    Compute the trajectory of an object across frames.

    Args:
        frames: Array of shape [num_frames, height, width, channels]
        target_color: Target RGB color tuple

    Returns:
        List of (x, y) center coordinates per frame
    """
    trajectory = []

    for frame in frames:
        center = detect_object_center(frame, target_color)
        if center is not None:
            trajectory.append(center)
        else:
            # If object not detected, use previous position or center of frame
            if trajectory:
                trajectory.append(trajectory[-1])
            else:
                h, w = frame.shape[:2]
                trajectory.append((w/2, h/2))

    return trajectory


def compute_motion_metrics(trajectory: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Compute motion metrics from a trajectory.

    Args:
        trajectory: List of (x, y) coordinates

    Returns:
        Dictionary of motion metrics
    """
    if len(trajectory) < 2:
        return {
            'total_distance': 0.0,
            'avg_speed': 0.0,
            'direction': 'unknown',
            'direction_consistency': 0.0
        }

    # Compute displacements
    displacements = []
    for i in range(len(trajectory) - 1):
        dx = trajectory[i+1][0] - trajectory[i][0]
        dy = trajectory[i+1][1] - trajectory[i][1]
        displacements.append((dx, dy))

    # Total distance
    total_distance = sum(np.sqrt(dx**2 + dy**2) for dx, dy in displacements)

    # Average speed (distance per frame)
    avg_speed = total_distance / (len(trajectory) - 1)

    # Overall direction
    total_dx = sum(dx for dx, dy in displacements)
    total_dy = sum(dy for dx, dy in displacements)

    if abs(total_dx) > abs(total_dy):
        direction = 'left_to_right' if total_dx > 0 else 'right_to_left'
    else:
        direction = 'top_to_bottom' if total_dy > 0 else 'bottom_to_top'

    # Direction consistency (fraction of displacements in the same direction)
    consistent_count = 0
    for dx, dy in displacements:
        if direction == 'left_to_right' and dx > 0:
            consistent_count += 1
        elif direction == 'right_to_left' and dx < 0:
            consistent_count += 1
        elif direction == 'top_to_bottom' and dy > 0:
            consistent_count += 1
        elif direction == 'bottom_to_top' and dy < 0:
            consistent_count += 1

    direction_consistency = consistent_count / len(displacements) if displacements else 0.0

    return {
        'total_distance': float(total_distance),
        'avg_speed': float(avg_speed),
        'direction': direction,
        'direction_consistency': float(direction_consistency)
    }


def compute_trajectory_error(
    predicted_trajectory: List[Tuple[float, float]],
    target_trajectory: List[Tuple[float, float]]
) -> Dict[str, float]:
    """
    Compute error between predicted and target trajectories.

    Args:
        predicted_trajectory: Predicted object centers
        target_trajectory: Ground truth object centers

    Returns:
        Dictionary of error metrics
    """
    if len(predicted_trajectory) != len(target_trajectory):
        # Pad the shorter one
        max_len = max(len(predicted_trajectory), len(target_trajectory))
        predicted_trajectory = predicted_trajectory + [predicted_trajectory[-1]] * (max_len - len(predicted_trajectory))
        target_trajectory = target_trajectory + [target_trajectory[-1]] * (max_len - len(target_trajectory))

    # Compute per-frame errors
    frame_errors = []
    for pred, target in zip(predicted_trajectory, target_trajectory):
        error = np.sqrt((pred[0] - target[0])**2 + (pred[1] - target[1])**2)
        frame_errors.append(error)

    # Overall metrics
    avg_error = np.mean(frame_errors)
    max_error = np.max(frame_errors)
    std_error = np.std(frame_errors)

    return {
        'avg_trajectory_error': float(avg_error),
        'max_trajectory_error': float(max_error),
        'std_trajectory_error': float(std_error),
        'frame_errors': [float(e) for e in frame_errors]
    }


def compute_color_consistency(frames: np.ndarray, target_color: Tuple[int, int, int]) -> Dict[str, float]:
    """
    Compute color consistency metrics.

    Args:
        frames: Array of shape [num_frames, height, width, channels]
        target_color: Target RGB color tuple

    Returns:
        Dictionary of color consistency metrics
    """
    target_color_np = np.array(target_color)
    color_errors = []

    for frame in frames:
        # Sample center of frame
        h, w = frame.shape[:2]
        center_region = frame[h//4:3*h//4, w//4:3*w//4]

        # Compute average color in center region
        avg_color = center_region.mean(axis=(0, 1))

        # Compute error from target
        color_error = np.linalg.norm(avg_color - target_color_np)
        color_errors.append(color_error)

    return {
        'avg_color_error': float(np.mean(color_errors)),
        'std_color_error': float(np.std(color_errors)),
        'max_color_error': float(np.max(color_errors))
    }


def evaluate_video_against_trajectory(
    frame_dir: str,
    target_metadata: Dict
) -> Dict[str, float]:
    """
    Evaluate a generated video against its target trajectory.

    Args:
        frame_dir: Directory containing generated frames
        target_metadata: Ground truth metadata with trajectory

    Returns:
        Dictionary of evaluation metrics
    """
    # Load generated frames
    frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
    frames = []

    for frame_file in frame_files:
        img = Image.open(frame_file)
        frames.append(np.array(img))

    frames = np.stack(frames)  # [frames, h, w, c]

    # Get target color from metadata
    color_map = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'cyan': (0, 255, 255),
        'pink': (255, 192, 203),
    }

    target_color = color_map.get(target_metadata['color'], (128, 128, 128))

    # Compute predicted trajectory
    predicted_trajectory = compute_trajectory(frames, target_color)

    # Get target trajectory from metadata
    target_trajectory = target_metadata['trajectory']

    # Compute motion metrics
    motion_metrics = compute_motion_metrics(predicted_trajectory)

    # Compute trajectory error
    trajectory_error = compute_trajectory_error(predicted_trajectory, target_trajectory)

    # Compute color consistency
    color_consistency = compute_color_consistency(frames, target_color)

    # Combine all metrics
    all_metrics = {
        **motion_metrics,
        **trajectory_error,
        **color_consistency
    }

    return all_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate video against target trajectory")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory with generated frames")
    parser.add_argument("--target_metadata", type=str, required=True, help="Path to target metadata JSON")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")

    args = parser.parse_args()

    # Load target metadata
    with open(args.target_metadata, 'r') as f:
        target_metadata = json.load(f)

    # Evaluate
    metrics = evaluate_video_against_trajectory(args.frame_dir, target_metadata)

    # Print results
    print("Evaluation Results:")
    print("=" * 50)
    for key, value in metrics.items():
        if key != 'frame_errors':
            print(f"  {key}: {value:.4f}")
    print("=" * 50)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
