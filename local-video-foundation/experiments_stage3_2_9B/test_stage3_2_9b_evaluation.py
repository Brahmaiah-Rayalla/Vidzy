"""
Stage 3.2.9B - Evaluation Regression Tests

Test the evaluation framework before pilot training.

Author: Devin
Date: 2026-11-01
"""

import sys
import os
import torch
import numpy as np
import unittest
from pathlib import Path

# Add paths
parent_dir = Path(__file__).parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))

from stage3_2_9b_evaluation import ObjectTracker, MotionEvaluator, AppearanceEvaluator, TemporalEvaluator


class EvaluationFrameworkTests(unittest.TestCase):
    """Test evaluation framework components."""

    def setUp(self):
        self.tracker = ObjectTracker()
        self.motion_eval = MotionEvaluator()
        self.appearance_eval = AppearanceEvaluator()
        self.temporal_eval = TemporalEvaluator()

    def test_1_stationary_trajectory_zero_displacement(self):
        """Stationary trajectory produces approximately zero displacement."""
        trajectory = [(32.0, 32.0) for _ in range(16)]
        displacement = self.motion_eval.compute_displacement(trajectory)
        self.assertLess(displacement, 0.1, f"Stationary displacement: {displacement}")

    def test_2_leftward_classification(self):
        """Leftward movement is classified correctly."""
        trajectory = [(50.0, 32.0), (40.0, 32.0), (30.0, 32.0), (20.0, 32.0)]
        direction = self.motion_eval.classify_direction(trajectory)
        self.assertEqual(direction, "left", f"Expected 'left', got '{direction}'")

    def test_3_rightward_classification(self):
        """Rightward movement is classified correctly."""
        trajectory = [(10.0, 32.0), (20.0, 32.0), (30.0, 32.0), (40.0, 32.0)]
        direction = self.motion_eval.classify_direction(trajectory)
        self.assertEqual(direction, "right", f"Expected 'right', got '{direction}'")

    def test_4_upward_classification(self):
        """Upward movement is classified correctly."""
        trajectory = [(32.0, 50.0), (32.0, 40.0), (32.0, 30.0), (32.0, 20.0)]
        direction = self.motion_eval.classify_direction(trajectory)
        self.assertEqual(direction, "up", f"Expected 'up', got '{direction}'")

    def test_5_downward_classification(self):
        """Downward movement is classified correctly."""
        trajectory = [(32.0, 10.0), (32.0, 20.0), (32.0, 30.0), (32.0, 40.0)]
        direction = self.motion_eval.classify_direction(trajectory)
        self.assertEqual(direction, "down", f"Expected 'down', got '{direction}'")

    def test_6_speed_calculation(self):
        """Speed calculations match known synthetic values."""
        # Moving 10 pixels over 10 frames = 1 pixel/frame
        trajectory = [(10.0, 32.0), (11.0, 32.0), (12.0, 32.0), (13.0, 32.0), (14.0, 32.0)]
        speed = self.motion_eval.compute_average_speed(trajectory)
        self.assertAlmostEqual(speed, 1.0, places=1, msg=f"Speed: {speed}")

    def test_7_direction_accuracy(self):
        """Direction reversal is detected."""
        accuracy = self.motion_eval.compute_direction_accuracy("left", "right")
        self.assertEqual(accuracy, 0.0, "Opposite directions should have 0 accuracy")

        accuracy = self.motion_eval.compute_direction_accuracy("left", "left")
        self.assertEqual(accuracy, 1.0, "Same directions should have 1 accuracy")

    def test_8_frame_to_frame_error(self):
        """Frame-to-frame error calculation works."""
        pred = [(10.0, 32.0), (11.0, 32.0), (12.0, 32.0)]
        gt = [(10.0, 32.0), (12.0, 32.0), (14.0, 32.0)]
        error = self.motion_eval.compute_frame_to_frame_error(pred, gt)
        self.assertAlmostEqual(error, 1.0, places=1, msg=f"Error: {error}")

    def test_9_color_consistency(self):
        """Color consistency calculation works."""
        # Constant frames should have zero variation
        frames = torch.ones(3, 16, 64, 64) * 0.5
        consistency = self.appearance_eval.compute_color_consistency(frames)
        self.assertEqual(consistency, 0.0, f"Constant frames should have zero consistency variation: {consistency}")

    def test_10_temporal_smoothness(self):
        """Temporal smoothness calculation works."""
        # Constant frames should be perfectly smooth
        frames = torch.ones(3, 16, 64, 64) * 0.5
        smoothness = self.temporal_eval.compute_temporal_smoothness(frames)
        self.assertEqual(smoothness, 0.0, f"Constant frames should be perfectly smooth: {smoothness}")

    def test_11_flicker_detection(self):
        """Flicker detection works."""
        # Alternating frames should have high flicker
        frames = torch.zeros(3, 16, 64, 64)
        frames[:, ::2, :, :] = 1.0
        flicker = self.temporal_eval.compute_flicker(frames)
        self.assertGreater(flicker, 0.0, f"Alternating frames should have flicker: {flicker}")

    def test_12_detection_stats(self):
        """Detection stats are tracked correctly."""
        self.tracker.detection_successes = 8
        self.tracker.detection_failures = 2
        stats = self.tracker.get_detection_stats()
        self.assertEqual(stats['successes'], 8)
        self.assertEqual(stats['failures'], 2)
        self.assertAlmostEqual(stats['success_rate'], 0.8, places=2)

    def test_13_smoothness_not_motion_accuracy(self):
        """Temporal smoothness does not automatically imply motion accuracy."""
        # Stationary video is smooth but has no motion
        frames = torch.ones(3, 16, 64, 64) * 0.5
        smoothness = self.temporal_eval.compute_temporal_smoothness(frames)
        displacement = self.motion_eval.compute_displacement([(32.0, 32.0) for _ in range(16)])
        
        self.assertEqual(smoothness, 0.0, "Stationary video is smooth")
        self.assertEqual(displacement, 0.0, "Stationary video has no displacement")
        # Smoothness != motion accuracy

    def test_14_evaluation_deterministic(self):
        """Evaluation results are deterministic for fixed inputs."""
        frames = torch.randn(3, 16, 64, 64)
        torch.manual_seed(42)
        result1 = self.appearance_eval.compute_color_consistency(frames)
        torch.manual_seed(42)
        result2 = self.appearance_eval.compute_color_consistency(frames)
        self.assertEqual(result1, result2, "Results should be deterministic")

    def test_15_short_trajectory_handling(self):
        """Short trajectories are handled gracefully."""
        # Single point
        trajectory = [(32.0, 32.0)]
        displacement = self.motion_eval.compute_displacement(trajectory)
        self.assertEqual(displacement, 0.0, "Single point should have zero displacement")

        speed = self.motion_eval.compute_average_speed(trajectory)
        self.assertEqual(speed, 0.0, "Single point should have zero speed")


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(EvaluationFrameworkTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
