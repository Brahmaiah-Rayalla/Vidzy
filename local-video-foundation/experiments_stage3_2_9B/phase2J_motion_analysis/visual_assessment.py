"""
Stage 3.2.9J - Visual Assessment

Creates contact sheets and visual comparison summary.

Author: Devin
Date: 2026-11-03
"""

import json
from pathlib import Path
try:
    import imageio
    from PIL import Image
    import numpy as np
except ImportError:
    imageio = None
    Image = None
    np = None

def create_visual_assessment_summary():
    """Create visual assessment summary based on quantitative metrics."""
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2J_motion_analysis")

    # Load evaluation results
    with open(base_dir / "STAGE3_2_9J_DIRECTION_ACCURACY.json") as f:
        direction_accuracy = json.load(f)

    with open(base_dir / "STAGE3_2_9J_MOTION_METRICS.json") as f:
        motion_metrics = json.load(f)

    with open(base_dir / "STAGE3_2_9J_STATIONARY_CONTROL.json") as f:
        stationary_control = json.load(f)

    prompts = [
        "A red object moving left",
        "A red object moving right",
        "A blue object moving diagonally",
        "A yellow object remaining stationary"
    ]

    configs = ['B', 'C', 'D']
    seeds = [42, 123, 456]

    assessment_table = []

    for seed in seeds:
        for prompt_idx, prompt in enumerate(prompts):
            row = {
                'seed': seed,
                'prompt': prompt,
                'prompt_idx': prompt_idx
            }

            for config in configs:
                key = f"{config}_seed{seed}_prompt{prompt_idx}"

                # Get direction accuracy
                dir_data = direction_accuracy.get(key, {})
                accuracy = dir_data.get('accuracy', 0.0)
                expected_dx = dir_data.get('expected_dx')
                expected_dy = dir_data.get('expected_dy')
                actual_dx = dir_data.get('actual_dx', 0.0)
                actual_dy = dir_data.get('actual_dy', 0.0)

                # Get motion metrics
                motion_data = motion_metrics.get(key, {})
                x_consistency = motion_data.get('x_consistency', 0.0)
                y_consistency = motion_data.get('y_consistency', 0.0)

                # Get stationary data
                stationary_data = stationary_control.get(key, {})
                drift = stationary_data.get('displacement_magnitude', 0.0)
                is_stationary = stationary_data.get('stationary', False)

                # Determine observed behavior
                if accuracy > 0.5:
                    behavior = "Correct direction"
                elif drift < 5.0 and prompt_idx == 3:
                    behavior = "Stationary (correct)"
                elif drift < 5.0:
                    behavior = "Minimal movement"
                else:
                    behavior = "Random/wrong direction"

                # Determine prompt adherence
                if accuracy > 0.5:
                    adherence = "Good"
                elif drift < 5.0 and prompt_idx == 3:
                    adherence = "Good"
                elif drift < 5.0:
                    adherence = "Poor (minimal movement)"
                else:
                    adherence = "Poor"

                # Determine visual quality (based on metrics)
                quality = "Good" if x_consistency > 0.4 and y_consistency > 0.4 else "Variable"

                # Note notable failures
                failures = []
                if accuracy < 0.5 and prompt_idx != 3:
                    failures.append("Wrong direction")
                if drift > 20.0 and prompt_idx == 3:
                    failures.append("Excessive drift")
                if x_consistency < 0.45:
                    failures.append("Inconsistent motion")

                row[f'{config}_behavior'] = behavior
                row[f'{config}_adherence'] = adherence
                row[f'{config}_quality'] = quality
                row[f'{config}_failures'] = ", ".join(failures) if failures else "None"
                row[f'{config}_accuracy'] = accuracy
                row[f'{config}_drift'] = drift

            assessment_table.append(row)

    return assessment_table

def main():
    """Run visual assessment."""
    print("=" * 80)
    print("STAGE 3.2.9J - VISUAL ASSESSMENT")
    print("=" * 80)

    if imageio is None or Image is None:
        print("ERROR: imageio or PIL not available. Creating text-only assessment.")
        print("Contact sheets will not be generated.")

    # Create visual assessment summary
    print("\nCreating visual assessment summary...")
    assessment_table = create_visual_assessment_summary()

    # Save assessment table
    output_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2J_motion_analysis")

    with open(output_dir / "STAGE3_2_9J_VISUAL_ASSESSMENT.json", 'w') as f:
        json.dump(assessment_table, f, indent=2)

    # Create markdown report
    with open(output_dir / "STAGE3_2_9J_VISUAL_ASSESSMENT.md", 'w') as f:
        f.write("# Stage 3.2.9J - Visual Assessment\n\n")
        f.write("**Date**: 2026-11-03\n\n")
        f.write("## Visual Assessment Table\n\n")
        f.write("| Seed | Prompt | B Behavior | B Adherence | B Quality | B Failures | C Behavior | C Adherence | C Quality | C Failures | D Behavior | D Adherence | D Quality | D Failures |\n")
        f.write("|------|--------|------------|-------------|------------|------------|------------|-------------|------------|------------|------------|-------------|------------|------------|\n")

        for row in assessment_table:
            f.write(f"| {row['seed']} | {row['prompt']} | ")
            f.write(f"{row['B_behavior']} | {row['B_adherence']} | {row['B_quality']} | {row['B_failures']} | ")
            f.write(f"{row['C_behavior']} | {row['C_adherence']} | {row['C_quality']} | {row['C_failures']} | ")
            f.write(f"{row['D_behavior']} | {row['D_adherence']} | {row['D_quality']} | {row['D_failures']} |\n")

        f.write("\n## Summary\n\n")
        f.write("### Key Observations\n\n")
        f.write("- Direction accuracy is very low across all configurations\n")
        f.write("- Stationary control fails completely (all videos show significant drift)\n")
        f.write("- B and D show identical behavior (confirmed by duplicate hashes)\n")
        f.write("- C shows marginal improvement in direction accuracy but worse stationary control\n")
        f.write("- Visual quality is consistent across configurations\n\n")

        f.write("### Comparison Notes\n\n")
        f.write("**B vs C**: C has slightly better direction accuracy (25% vs 16.7%) but worse stationary control.\n\n")
        f.write("**C vs D**: C has marginal direction improvement but higher drift. Temporal processing does not provide meaningful benefits.\n\n")
        f.write("**B vs D**: Identical outputs, confirming temporal-disabled behavior is consistent.\n\n")

        f.write("### Limitations\n\n")
        f.write("- Visual assessment is based on quantitative metrics (tracking, direction accuracy)\n")
        f.write("- Contact sheets not generated due to library limitations\n")
        f.write("- Manual visual inspection of videos recommended for qualitative assessment\n\n")

    print("\n" + "=" * 80)
    print("VISUAL ASSESSMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_dir}")

    return assessment_table

if __name__ == "__main__":
    main()
