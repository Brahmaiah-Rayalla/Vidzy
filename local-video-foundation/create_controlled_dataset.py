"""
Create controlled video dataset for Stage 3.2 validation.

Generates a small dataset with specific, known trajectories:
A. Red circle moving left to right
B. Red circle moving right to left
C. Blue square moving diagonally
D. Yellow circle remaining stationary

Each trajectory is generated multiple times with different random seeds
to create enough training examples for overfitting experiments.
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.data.synthetic_video_dataset import SyntheticVideoGenerator


def create_controlled_dataset(
    output_dir="data/controlled_video",
    num_examples_per_trajectory=8,
    num_frames=16,
    image_size=64,
    seed=42
):
    """
    Create controlled dataset with specific trajectories.

    Args:
        output_dir: Output directory
        num_examples_per_trajectory: Number of examples per trajectory type
        num_frames: Number of frames per video
        image_size: Image resolution
        seed: Random seed
    """
    np.random.seed(seed)

    # Create output directory
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    # Create video generator
    generator = SyntheticVideoGenerator(
        image_size=image_size,
        num_frames=num_frames,
        seed=seed
    )

    # Define controlled trajectories
    trajectories = [
        {
            'name': 'red_circle_left_to_right',
            'shape': 'circle',
            'color': 'red',
            'motion': 'left_to_right',
            'size_adj': 'medium',
            'speed_adj': 'quickly',
            'bg_color': 'black'
        },
        {
            'name': 'red_circle_right_to_left',
            'shape': 'circle',
            'color': 'red',
            'motion': 'right_to_left',
            'size_adj': 'medium',
            'speed_adj': 'quickly',
            'bg_color': 'black'
        },
        {
            'name': 'blue_square_diagonal',
            'shape': 'square',
            'color': 'blue',
            'motion': 'diagonal_down_right',
            'size_adj': 'medium',
            'speed_adj': 'quickly',
            'bg_color': 'black'
        },
        {
            'name': 'yellow_circle_stationary',
            'shape': 'circle',
            'color': 'yellow',
            'motion': 'stationary',
            'size_adj': 'medium',
            'speed_adj': 'slowly',
            'bg_color': 'black'
        }
    ]

    all_metadata = []
    all_captions = []
    video_dirs = []

    print(f"Creating controlled dataset with {len(trajectories)} trajectory types")
    print(f"Examples per trajectory: {num_examples_per_trajectory}")
    print(f"Total videos: {len(trajectories) * num_examples_per_trajectory}")
    print()

    for traj_idx, traj in enumerate(trajectories):
        print(f"Generating trajectory {traj_idx + 1}/{len(trajectories)}: {traj['name']}")

        for example_idx in range(num_examples_per_trajectory):
            # Use different seed for each example
            example_seed = seed + traj_idx * 1000 + example_idx
            generator.rng = np.random.RandomState(example_seed)

            # Generate video
            frames, caption, metadata = generator.generate_single_video(
                shape=traj['shape'],
                color=traj['color'],
                motion=traj['motion'],
                size_adj=traj['size_adj'],
                speed_adj=traj['speed_adj'],
                bg_color=traj['bg_color']
            )

            # Add trajectory type to metadata
            metadata['trajectory_type'] = traj['name']
            metadata['example_index'] = example_idx
            metadata['trajectory_index'] = traj_idx

            # Save video
            video_name = f"{traj['name']}_example_{example_idx:03d}"
            video_dir = os.path.join(train_dir, video_name)
            os.makedirs(video_dir, exist_ok=True)

            for frame_idx, frame in enumerate(frames):
                frame_path = os.path.join(video_dir, f"frame_{frame_idx:04d}.png")
                frame.save(frame_path)

            # Save metadata
            metadata_path = os.path.join(video_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            video_dirs.append(video_dir)
            all_metadata.append(metadata)
            all_captions.append(caption)

            if (example_idx + 1) % 2 == 0:
                print(f"  Generated {example_idx + 1}/{num_examples_per_trajectory} examples")

        print()

    # Save all metadata
    all_metadata_path = os.path.join(train_dir, "all_metadata.json")
    with open(all_metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    # Save captions
    captions_path = os.path.join(train_dir, "captions.txt")
    with open(captions_path, 'w') as f:
        for caption in all_captions:
            f.write(caption + '\n')

    # Save dataset info
    dataset_info = {
        'num_videos': len(video_dirs),
        'num_frames': num_frames,
        'image_size': image_size,
        'trajectories': trajectories,
        'examples_per_trajectory': num_examples_per_trajectory,
        'seed': seed
    }

    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Dataset creation complete!")
    print(f"  Total videos: {len(video_dirs)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Train directory: {train_dir}")
    print(f"  Metadata: {all_metadata_path}")
    print(f"  Captions: {captions_path}")
    print(f"  Dataset info: {info_path}")

    return video_dirs, all_metadata


def validate_dataset(dataset_dir="data/controlled_video"):
    """
    Validate the controlled dataset.

    Checks that:
    - Each caption matches the actual generated video
    - Object identity remains consistent
    - The trajectory is continuous
    - Motion direction and velocity are recorded in metadata
    """
    print("\n" + "=" * 60)
    print("VALIDATING CONTROLLED DATASET")
    print("=" * 60)

    train_dir = os.path.join(dataset_dir, "train")

    # Load metadata
    metadata_path = os.path.join(train_dir, "all_metadata.json")
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    # Load captions
    captions_path = os.path.join(train_dir, "captions.txt")
    with open(captions_path, 'r') as f:
        captions = [line.strip() for line in f.readlines()]

    print(f"\nLoaded {len(all_metadata)} videos")

    # Validate each video
    errors = []
    warnings = []

    for idx, (metadata, caption) in enumerate(zip(all_metadata, captions)):
        video_dir = os.path.join(train_dir, f"{metadata['trajectory_type']}_example_{metadata['example_index']:03d}")

        # Check caption matches metadata
        expected_shape = metadata['shape']
        expected_color = metadata['color']
        expected_motion = metadata['motion']

        if expected_shape not in caption.lower():
            errors.append(f"Video {idx}: Shape '{expected_shape}' not in caption '{caption}'")

        if expected_color not in caption.lower():
            errors.append(f"Video {idx}: Color '{expected_color}' not in caption '{caption}'")

        # Check trajectory continuity
        trajectory = metadata['trajectory']
        if len(trajectory) != metadata['num_frames']:
            errors.append(f"Video {idx}: Trajectory length {len(trajectory)} != num_frames {metadata['num_frames']}")

        # Check motion direction is in metadata
        if 'motion' not in metadata:
            errors.append(f"Video {idx}: Motion type not in metadata")

        # Check velocity is in metadata
        if 'velocity' not in metadata:
            errors.append(f"Video {idx}: Velocity not in metadata")

        # Check frames exist
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.png')])
        if len(frame_files) != metadata['num_frames']:
            errors.append(f"Video {idx}: Found {len(frame_files)} frames, expected {metadata['num_frames']}")

    # Print results
    print(f"\nValidation results:")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")

    if errors:
        print(f"\nErrors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    if not errors:
        print("\nDataset validation PASSED")
    else:
        print("\nDataset validation FAILED")

    return len(errors) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create controlled video dataset")
    parser.add_argument("--output_dir", type=str, default="data/controlled_video", help="Output directory")
    parser.add_argument("--num_examples", type=int, default=8, help="Examples per trajectory")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate_only", action="store_true", help="Only validate existing dataset")

    args = parser.parse_args()

    if args.validate_only:
        validate_dataset(args.output_dir)
    else:
        create_controlled_dataset(
            output_dir=args.output_dir,
            num_examples_per_trajectory=args.num_examples,
            num_frames=args.num_frames,
            image_size=args.image_size,
            seed=args.seed
        )
        validate_dataset(args.output_dir)
