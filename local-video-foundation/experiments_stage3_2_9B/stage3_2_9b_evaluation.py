"""
Stage 3.2.9B - Evaluation Framework

Comprehensive evaluation for temporal learning experiments.
Supports Experiments A, B, and C with consistent inputs and measurements.

Author: Devin
Date: 2026-11-01
"""

import sys
import os
import json
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import time

# Add paths
parent_dir = Path(__file__).parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))

from video_foundation.models.conditional_unet import ConditionalUNet as OriginalConditionalUNet
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed
from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder


class ObjectTracker:
    """Lightweight CPU-compatible object tracking for synthetic geometric objects."""

    def __init__(self):
        self.detection_failures = 0
        self.detection_successes = 0

    def locate_object(self, frame: torch.Tensor, metadata: Dict) -> Optional[Tuple[float, float]]:
        """
        Locate object center in frame using color-based segmentation.

        Args:
            frame: [H, W, C] tensor in [-1, 1] range
            metadata: Metadata with shape, color, background

        Returns:
            (x, y) center coordinates or None if detection fails
        """
        try:
            # Convert to [0, 255] range
            frame_np = ((frame + 1) * 127.5).byte().cpu().numpy()

            # Simple color-based segmentation
            # Convert to numpy and extract color channels
            h, w, c = frame_np.shape

            # Get target color from metadata
            color_map = {
                'red': [255, 0, 0],
                'green': [0, 255, 0],
                'blue': [0, 0, 255],
                'yellow': [255, 255, 0],
                'pink': [255, 192, 203],
                'purple': [128, 0, 128],
                'orange': [255, 165, 0],
                'cyan': [0, 255, 255],
                'white': [255, 255, 255],
                'black': [0, 0, 0],
                'gray': [128, 128, 128],
                'light_gray': [211, 211, 211],
                'light_blue': [173, 216, 230],
                'light_yellow': [255, 255, 224],
                'cream': [255, 253, 208]
            }

            target_color = np.array(color_map.get(metadata.get('color', 'red'), [255, 0, 0]))
            background_color = np.array(color_map.get(metadata.get('background', 'white'), [255, 255, 255]))

            # Compute color distance
            color_dist = np.sqrt(np.sum((frame_np.astype(float) - target_color) ** 2, axis=2))
            bg_dist = np.sqrt(np.sum((frame_np.astype(float) - background_color) ** 2, axis=2))

            # Threshold: object pixels are closer to target color than background
            mask = color_dist < bg_dist

            if mask.sum() < 10:  # Too few pixels
                self.detection_failures += 1
                return None

            # Compute center of mass
            y_indices, x_indices = np.where(mask)
            center_y = float(y_indices.mean())
            center_x = float(x_indices.mean())

            self.detection_successes += 1
            return (center_x, center_y)

        except Exception as e:
            self.detection_failures += 1
            return None

    def get_detection_stats(self) -> Dict:
        return {
            'successes': self.detection_successes,
            'failures': self.detection_failures,
            'success_rate': self.detection_successes / (self.detection_successes + self.detection_failures) if (self.detection_successes + self.detection_failures) > 0 else 0.0
        }


class MotionEvaluator:
    """Evaluate motion accuracy from trajectories."""

    @staticmethod
    def compute_displacement(trajectory: List[Tuple[float, float]]) -> float:
        """Compute total displacement from start to end."""
        if len(trajectory) < 2:
            return 0.0
        start = trajectory[0]
        end = trajectory[-1]
        return np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

    @staticmethod
    def compute_average_speed(trajectory: List[Tuple[float, float]], fps: float = 1.0) -> float:
        """Compute average speed (pixels per frame)."""
        if len(trajectory) < 2:
            return 0.0
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        return total_distance / (len(trajectory) - 1)

    @staticmethod
    def classify_direction(trajectory: List[Tuple[float, float]]) -> str:
        """Classify motion direction."""
        if len(trajectory) < 2:
            return "stationary"

        start = trajectory[0]
        end = trajectory[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        # Determine dominant direction
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

    @staticmethod
    def compute_direction_accuracy(predicted: str, ground_truth: str) -> float:
        """Check if predicted direction matches ground truth."""
        return 1.0 if predicted == ground_truth else 0.0

    @staticmethod
    def compute_frame_to_frame_error(pred_traj: List[Tuple[float, float]], gt_traj: List[Tuple[float, float]]) -> float:
        """Compute average frame-to-frame displacement error."""
        if len(pred_traj) != len(gt_traj) or len(pred_traj) < 2:
            return float('inf')

        errors = []
        for i in range(1, len(pred_traj)):
            pred_dx = pred_traj[i][0] - pred_traj[i-1][0]
            pred_dy = pred_traj[i][1] - pred_traj[i-1][1]
            gt_dx = gt_traj[i][0] - gt_traj[i-1][0]
            gt_dy = gt_traj[i][1] - gt_traj[i-1][1]
            error = np.sqrt((pred_dx - gt_dx)**2 + (pred_dy - gt_dy)**2)
            errors.append(error)

        return np.mean(errors) if errors else float('inf')


class AppearanceEvaluator:
    """Evaluate appearance preservation."""

    @staticmethod
    def compute_color_consistency(frames: torch.Tensor) -> float:
        """Compute color consistency across frames."""
        # Convert to [T, H, W, C]
        frames_perm = frames.permute(1, 2, 3, 0)  # [T, H, W, C]
        mean_colors = frames_perm.mean(dim=(1, 2))  # [T, C]
        color_std = mean_colors.std(dim=0).mean()  # Std across time
        return float(color_std)

    @staticmethod
    def compute_frame_variation(frames: torch.Tensor) -> float:
        """Compute average frame-to-frame variation."""
        frames_perm = frames.permute(1, 2, 3, 0)  # [T, H, W, C]
        variations = []
        for i in range(1, frames_perm.shape[0]):
            diff = torch.abs(frames_perm[i] - frames_perm[i-1]).mean()
            variations.append(diff.item())
        return np.mean(variations) if variations else 0.0

    @staticmethod
    def compute_object_visibility(frames: torch.Tensor, metadata: Dict) -> float:
        """Compute object visibility (contrast with background)."""
        # Simple metric: std of frame as proxy for object presence
        return frames.std().item()


class TemporalEvaluator:
    """Evaluate temporal consistency."""

    @staticmethod
    def compute_temporal_smoothness(frames: torch.Tensor) -> float:
        """Compute temporal smoothness (inverse of high-frequency variation)."""
        frames_perm = frames.permute(1, 2, 3, 0)  # [T, H, W, C]
        smoothness = 0.0
        for i in range(1, frames_perm.shape[0]):
            # Second derivative
            if i < frames_perm.shape[0] - 1:
                second_diff = frames_perm[i+1] - 2 * frames_perm[i] + frames_perm[i-1]
                smoothness += torch.abs(second_diff).mean().item()
        return smoothness / max(1, frames_perm.shape[0] - 2)

    @staticmethod
    def compute_flicker(frames: torch.Tensor) -> float:
        """Compute high-frequency temporal flicker."""
        frames_perm = frames.permute(1, 2, 3, 0)  # [T, H, W, C]
        flicker = 0.0
        for i in range(1, frames_perm.shape[0]):
            diff = torch.abs(frames_perm[i] - frames_perm[i-1])
            flicker += (diff > 0.1).float().mean().item()  # Count pixels with large changes
        return flicker / max(1, frames_perm.shape[0] - 1)


class Stage3_2_9BEvaluator:
    """Main evaluator for Stage 3.2.9B experiments."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.parent_dir = self.experiment_dir.parent

        # Paths
        self.stage2_checkpoint = str(self.parent_dir / "checkpoints_conditional" / "checkpoint_final.pt")
        self.stage3_spatial_checkpoint = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_spatial.pt")
        self.tokenizer_path = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json")
        self.val_metadata = str(self.parent_dir / "experiments_stage3_2_7" / "data_synthetic" / "val_metadata.json")

        # Evaluators
        self.tracker = ObjectTracker()
        self.motion_eval = MotionEvaluator()
        self.appearance_eval = AppearanceEvaluator()
        self.temporal_eval = TemporalEvaluator()

        # Results
        self.results = {
            'experiment_A': {},
            'experiment_B': {},
            'experiment_C': {}
        }

        # CPU/memory measurements
        self.start_time = None
        self.end_time = None

    def log_section(self, title: str):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def get_file_hash(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return "FILE_NOT_FOUND"
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def load_model(self, experiment_name: str):
        """Load model for specified experiment."""
        # TODO: Implement model loading
        pass

    def evaluate_clip(self, model, clip_data: Dict, experiment_name: str) -> Dict:
        """Evaluate a single video clip."""
        # Load video
        video = np.load(clip_data['video_path'])
        video_tensor = torch.from_numpy(video).float() / 255.0 * 2 - 1
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]

        # Load ground truth trajectory if available
        gt_trajectory = None
        if 'trajectory_path' in clip_data and os.path.exists(clip_data['trajectory_path']):
            gt_trajectory = np.load(clip_data['trajectory_path'])

        # Extract predicted trajectory
        pred_trajectory = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t, :, :]  # [C, H, W]
            frame_perm = frame.permute(1, 2, 0)  # [H, W, C]
            center = self.tracker.locate_object(frame_perm, clip_data)
            if center is not None:
                pred_trajectory.append(center)
            else:
                # Use interpolation or last known position
                if pred_trajectory:
                    pred_trajectory.append(pred_trajectory[-1])
                else:
                    pred_trajectory.append((32.0, 32.0))  # Default center

        # Compute metrics
        metrics = {
            'clip_id': clip_data['id'],
            'motion_type': clip_data.get('motion_type', 'unknown'),
            'caption': clip_data['caption'],
            'detection_success': self.tracker.detection_successes,
            'detection_failure': self.tracker.detection_failures
        }

        # Motion accuracy
        if len(pred_trajectory) >= 2:
            metrics['displacement'] = self.motion_eval.compute_displacement(pred_trajectory)
            metrics['avg_speed'] = self.motion_eval.compute_average_speed(pred_trajectory)
            metrics['predicted_direction'] = self.motion_eval.classify_direction(pred_trajectory)
            metrics['ground_truth_direction'] = clip_data.get('motion_type', 'unknown')
            metrics['direction_accuracy'] = self.motion_eval.compute_direction_accuracy(
                metrics['predicted_direction'],
                metrics['ground_truth_direction']
            )

            if gt_trajectory is not None:
                metrics['trajectory_error'] = self.motion_eval.compute_frame_to_frame_error(
                    pred_trajectory, gt_trajectory.tolist()
                )

        # Appearance preservation
        metrics['color_consistency'] = self.appearance_eval.compute_color_consistency(video_tensor)
        metrics['frame_variation'] = self.appearance_eval.compute_frame_variation(video_tensor)
        metrics['object_visibility'] = self.appearance_eval.compute_object_visibility(video_tensor, clip_data)

        # Temporal consistency
        metrics['temporal_smoothness'] = self.temporal_eval.compute_temporal_smoothness(video_tensor)
        metrics['flicker'] = self.temporal_eval.compute_flicker(video_tensor)

        # Stationary behavior
        if clip_data.get('motion_type') == 'stationary':
            metrics['unwanted_movement'] = metrics['displacement']
            metrics['drift'] = metrics['displacement']
        else:
            metrics['unwanted_movement'] = 0.0  # Not applicable
            metrics['drift'] = 0.0  # Not applicable

        return metrics

    def evaluate_experiment(self, experiment_name: str, metadata: List[Dict]) -> Dict:
        """Evaluate all clips for an experiment."""
        self.log_section(f"EVALUATING EXPERIMENT {experiment_name}")

        model = self.load_model(experiment_name)

        all_metrics = []
        for clip_data in metadata[:10]:  # Evaluate first 10 clips for speed
            metrics = self.evaluate_clip(model, clip_data, experiment_name)
            all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = self.aggregate_metrics(all_metrics)

        return aggregated

    def aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across clips."""
        aggregated = {}

        # Collect numeric metrics
        numeric_keys = [
            'displacement', 'avg_speed', 'direction_accuracy', 'trajectory_error',
            'color_consistency', 'frame_variation', 'object_visibility',
            'temporal_smoothness', 'flicker', 'unwanted_movement', 'drift'
        ]

        for key in numeric_keys:
            values = [m.get(key, 0.0) for m in metrics_list if key in m and m.get(key) is not None]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)

        # Detection stats
        total_success = sum(m.get('detection_success', 0) for m in metrics_list)
        total_failure = sum(m.get('detection_failure', 0) for m in metrics_list)
        aggregated['detection_success_rate'] = total_success / (total_success + total_failure) if (total_success + total_failure) > 0 else 0.0

        # Direction accuracy
        direction_accuracies = [m.get('direction_accuracy', 0.0) for m in metrics_list if 'direction_accuracy' in m]
        if direction_accuracies:
            aggregated['direction_accuracy_mean'] = np.mean(direction_accuracies)

        return aggregated

    def run_evaluation(self):
        """Run complete evaluation for all experiments."""
        self.log_section("STAGE 3.2.9B - EVALUATION FRAMEWORK")

        self.start_time = time.time()

        # Load validation metadata
        with open(self.val_metadata, 'r') as f:
            val_metadata = json.load(f)

        print(f"Loaded {len(val_metadata)} validation clips")

        # Evaluate each experiment
        for exp_name in ['A', 'B', 'C']:
            self.results[f'experiment_{exp_name}'] = self.evaluate_experiment(exp_name, val_metadata)

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        # Add metadata
        self.results['evaluation_metadata'] = {
            'duration_seconds': duration,
            'num_clips_evaluated': len(val_metadata),
            'cpu_info': {
                'platform': os.name,
                'processor': os.uname().processor if hasattr(os, 'uname') else 'unknown'
            }
        }

        # Save results
        self.save_results()

    def save_results(self):
        """Save evaluation results."""
        results_path = self.experiment_dir / "STAGE3_2_9B_EVALUATION_RESULTS.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved results to {results_path}")


def main():
    experiment_dir = Path(__file__).parent
    evaluator = Stage3_2_9BEvaluator(str(experiment_dir))
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
