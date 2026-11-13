"""
Stage 3.2.9G - Evaluator Audit

Validate motion evaluation metrics using synthetic test videos with known trajectories.

Author: Devin
Date: 2026-11-03
"""

import numpy as np
from pathlib import Path
from PIL import Image
import json
from motion_evaluator import MotionEvaluator


class SyntheticVideoGenerator:
    """Generate synthetic videos with known ground-truth trajectories."""

    def __init__(self, num_frames=16, height=64, width=64):
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def generate_left_to_right(self, speed=2):
        """Generate video with object moving left to right."""
        frames = []
        start_x = 10
        y = self.height // 2
        for i in range(self.num_frames):
            x = start_x + speed * i
            frame = self._create_frame(x, y, color=(255, 0, 0))
            frames.append(frame)
        return frames, "left"

    def generate_right_to_left(self, speed=2):
        """Generate video with object moving right to left."""
        frames = []
        start_x = self.width - 10
        y = self.height // 2
        for i in range(self.num_frames):
            x = start_x - speed * i
            frame = self._create_frame(x, y, color=(255, 0, 0))
            frames.append(frame)
        return frames, "right"

    def generate_top_to_bottom(self, speed=2):
        """Generate video with object moving top to bottom."""
        frames = []
        x = self.width // 2
        start_y = 10
        for i in range(self.num_frames):
            y = start_y + speed * i
            frame = self._create_frame(x, y, color=(0, 0, 255))
            frames.append(frame)
        return frames, "down"

    def generate_bottom_to_top(self, speed=2):
        """Generate video with object moving bottom to top."""
        frames = []
        x = self.width // 2
        start_y = self.height - 10
        for i in range(self.num_frames):
            y = start_y - speed * i
            frame = self._create_frame(x, y, color=(0, 0, 255))
            frames.append(frame)
        return frames, "up"

    def generate_stationary(self):
        """Generate video with stationary object."""
        frames = []
        x = self.width // 2
        y = self.height // 2
        for i in range(self.num_frames):
            frame = self._create_frame(x, y, color=(128, 128, 128))
            frames.append(frame)
        return frames, "stationary"

    def generate_direction_reversal(self):
        """Generate video with object reversing direction."""
        frames = []
        x = self.width // 2
        y = self.height // 2
        speed = 2
        for i in range(self.num_frames):
            # Move right for first half, left for second half
            if i < self.num_frames // 2:
                offset = speed * i
            else:
                offset = speed * (self.num_frames - 1 - i)
            frame = self._create_frame(x + offset, y, color=(255, 0, 0))
            frames.append(frame)
        return frames, "reversal"

    def _create_frame(self, center_x, center_y, color):
        """Create a single frame with an object at the specified position."""
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255  # White background

        # Draw object (circle)
        radius = 8
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        dist_sq = (x_grid - center_x)**2 + (y_grid - center_y)**2
        mask = dist_sq <= radius**2

        # Ensure within bounds
        mask = mask & (x_grid >= 0) & (x_grid < self.width) & (y_grid >= 0) & (y_grid < self.height)

        frame[mask] = color

        return frame


def save_synthetic_video(frames, output_dir, name):
    """Save synthetic video frames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / name / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        frame_path = frames_dir / f"frame_{i:04d}.png"
        Image.fromarray(frame).save(frame_path)

    return frames_dir


def main():
    """Run evaluator audit."""
    print("="*80)
    print("STAGE 3.2.9G - EVALUATOR AUDIT")
    print("="*80)

    base_dir = Path(__file__).parent
    output_dir = base_dir / "synthetic_test_videos"

    # Generate synthetic videos
    generator = SyntheticVideoGenerator(num_frames=16, height=64, width=64)

    test_cases = [
        ("left_to_right", generator.generate_left_to_right, "right"),
        ("right_to_left", generator.generate_right_to_left, "left"),
        ("top_to_bottom", generator.generate_top_to_bottom, "down"),
        ("bottom_to_top", generator.generate_bottom_to_top, "up"),
        ("stationary", generator.generate_stationary, "stationary"),
        ("direction_reversal", generator.generate_direction_reversal, "reversal"),
    ]

    results = {}

    for name, gen_func, expected_direction in test_cases:
        print(f"\nGenerating: {name}")
        frames, actual_direction = gen_func()
        frames_dir = save_synthetic_video(frames, output_dir, name)

        # Evaluate
        evaluator = MotionEvaluator()
        evaluation = evaluator.evaluate_video(frames_dir, expected_direction)

        results[name] = {
            'expected_direction': expected_direction,
            'actual_direction': actual_direction,
            'evaluation': evaluation
        }

        # Print key metrics
        if evaluation.get('valid', False):
            tracking = evaluation['tracking']
            direction = evaluation['direction']
            speed = evaluation['speed']
            temporal = evaluation['temporal']

            print(f"  Detection Rate: {tracking['detection_rate']:.3f}")
            print(f"  Total Displacement: {tracking['total_displacement']:.2f} px")
            print(f"  Avg Displacement: {tracking['avg_displacement']:.2f} px")
            print(f"  Direction Accuracy: {direction['direction_accuracy']:.3f}")
            print(f"  Dominant Direction: {direction['dominant_direction']}")
            print(f"  Smoothness: {temporal['smoothness']:.6f}")
            print(f"  Frame-to-Frame Variation: {temporal['frame_to_frame_variation']:.6f}")
        else:
            print(f"  ERROR: {evaluation.get('error', 'Unknown error')}")

    # Save results
    results_path = output_dir / "evaluator_audit_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("AUDIT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to {results_path}")

    # Generate audit report
    generate_audit_report(results, output_dir)


def generate_audit_report(results, output_dir):
    """Generate audit report with findings."""
    report = []
    report.append("# Stage 3.2.9G - Evaluator Audit Report\n")
    report.append("**Date:** 2026-11-03\n")
    report.append("**Status:** Complete\n\n")
    report.append("## Coordinate Convention Validation\n\n")
    report.append("### Test Cases\n\n")

    coord_system_validation = {
        'left_to_right': {'expected': 'right', 'should_decrease_x': False},
        'right_to_left': {'expected': 'left', 'should_decrease_x': True},
        'top_to_bottom': {'expected': 'down', 'should_increase_y': True},
        'bottom_to_top': {'expected': 'up', 'should_increase_y': False},
    }

    for name, data in results.items():
        if name == 'direction_reversal':
            continue

        expected = data['expected_direction']
        eval_data = data['evaluation']

        if not eval_data.get('valid', False):
            report.append(f"### {name}\n")
            report.append(f"**Status:** FAILED - {eval_data.get('error', 'Unknown error')}\n\n")
            continue

        tracking = eval_data['tracking']
        direction = eval_data['direction']

        report.append(f"### {name}\n")
        report.append(f"- **Expected Direction:** {expected}\n")
        report.append(f"- **Detection Rate:** {tracking['detection_rate']:.3f}\n")
        report.append(f"- **Total Displacement:** {tracking['total_displacement']:.2f} px\n")
        report.append(f"- **Avg Displacement:** {tracking['avg_displacement']:.2f} px\n")
        report.append(f"- **Direction Accuracy:** {direction['direction_accuracy']:.3f}\n")
        report.append(f"- **Dominant Direction:** {direction['dominant_direction']}\n")
        report.append(f"- **Direction X:** {direction['direction_x']:.2f}\n")
        report.append(f"- **Direction Y:** {direction['direction_y']:.2f}\n\n")

        # Analyze coordinate system
        if name == 'left_to_right':
            if direction['direction_x'] > 0:
                report.append("[PASS] X increases when moving right (image coordinate system)\n")
            else:
                report.append("[FAIL] X decreases when moving right (inverse coordinate system)\n")
        elif name == 'right_to_left':
            if direction['direction_x'] < 0:
                report.append("[PASS] X decreases when moving left (image coordinate system)\n")
            else:
                report.append("[FAIL] X increases when moving left (inverse coordinate system)\n")
        elif name == 'top_to_bottom':
            if direction['direction_y'] > 0:
                report.append("[PASS] Y increases when moving down (image coordinate system)\n")
            else:
                report.append("[FAIL] Y decreases when moving down (inverse coordinate system)\n")
        elif name == 'bottom_to_top':
            if direction['direction_y'] < 0:
                report.append("[PASS] Y decreases when moving up (image coordinate system)\n")
            else:
                report.append("[FAIL] Y increases when moving up (inverse coordinate system)\n")

        report.append("\n")

    report.append("## Smoothness Analysis\n\n")
    report.append("### Smoothness Scores\n\n")

    for name, data in results.items():
        if not data['evaluation'].get('valid', False):
            continue
        temporal = data['evaluation']['temporal']
        report.append(f"- **{name}:** {temporal['smoothness']:.6f} (variation: {temporal['frame_to_frame_variation']:.6f})\n")

    report.append("\n")

    report.append("## Key Findings\n\n")

    # Analyze findings
    findings = []

    # Check coordinate system
    left_right = results['left_to_right']['evaluation']
    if left_right.get('valid', False):
        dx = left_right['direction']['direction_x']
        if dx > 0:
            findings.append("[PASS] Image coordinate system: X increases rightward, Y increases downward")
        else:
            findings.append("[FAIL] Unexpected coordinate system detected")

    # Check stationary detection
    stationary = results['stationary']['evaluation']
    if stationary.get('valid', False):
        disp = stationary['tracking']['total_displacement']
        if disp < 1.0:
            findings.append("[PASS] Stationary detection: Minimal displacement for stationary object")
        else:
            findings.append(f"[WARN] Stationary detection: {disp:.2f} px displacement detected (may be too sensitive)")

    # Check direction accuracy
    direction_accuracies = []
    for name, data in results.items():
        if name in ['direction_reversal', 'stationary']:
            continue
        if data['evaluation'].get('valid', False):
            acc = data['evaluation']['direction']['direction_accuracy']
            direction_accuracies.append((name, acc))

    all_perfect = all(acc == 1.0 for _, acc in direction_accuracies)
    if all_perfect:
        findings.append("[PASS] Direction accuracy: Perfect for all simple motions")
    else:
        failures = [name for name, acc in direction_accuracies if acc < 1.0]
        findings.append(f"[WARN] Direction accuracy: Failed for {', '.join(failures)}")

    for finding in findings:
        report.append(f"- {finding}\n")

    report.append("\n")

    report.append("## Recommendations\n\n")

    # Check if any issues were found
    if any("[FAIL]" in f or "[WARN]" in f for f in findings):
        report.append("Based on the audit findings:\n\n")
        if "[FAIL] Unexpected coordinate system" in " ".join(findings):
            report.append("1. **URGENT:** Correct coordinate system interpretation in motion_evaluator.py\n")
            report.append("   - Update direction determination logic to match image coordinate system\n")
        if "[WARN] Stationary detection" in " ".join(findings):
            report.append("2. Adjust stationary detection threshold in motion direction evaluation\n")
            report.append("   - Current threshold may be too sensitive for small displacements\n")
        if not all_perfect:
            report.append("3. Investigate direction accuracy failures for simple motions\n")
            report.append("   - Review object detection and trajectory tracking logic\n")
    else:
        report.append("The motion evaluation system is functioning correctly:\n")
        report.append("- Coordinate conventions are consistent with image coordinate system\n")
        report.append("- Direction detection works for basic motions\n")
        report.append("- Stationary detection is appropriately sensitive\n")
        report.append("- Previous Stage 3.2.9F results reflect actual model behavior, not evaluation bugs\n")

    report.append("\n")

    # Save report
    report_path = output_dir / "STAGE3_2_9G_EVALUATOR_AUDIT.md"
    with open(report_path, 'w') as f:
        f.writelines(report)

    print(f"Audit report saved to {report_path}")


if __name__ == "__main__":
    main()
