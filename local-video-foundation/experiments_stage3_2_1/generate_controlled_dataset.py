"""
Generate controlled dataset for Stage 3.2.1 overfitting experiment.

Creates a small dataset with 4 distinct trajectories to test temporal learning:
- Red circle left-to-right
- Red circle right-to-left
- Blue square diagonal
- Yellow circle stationary

Splits into train (80%) and validation (20%) for proper evaluation.
"""

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.data.synthetic_video_dataset import SyntheticVideoGenerator


def generate_controlled_dataset(
    output_dir: str,
    num_videos_per_trajectory: int = 8,
    num_frames: int = 16,
    image_size: int = 64,
    seed: int = 42
):
    """
    Generate controlled dataset with 4 distinct trajectories.

    Args:
        output_dir: Output directory
        num_videos_per_trajectory: Number of videos per trajectory type
        num_frames: Number of frames per video
        image_size: Image size
        seed: Random seed
    """
    random.seed(seed)

    # Define 4 distinct trajectories
    # Use gray background (128, 128, 128) for balanced normalization
    trajectories = [
        {
            'shape': 'circle',
            'color': 'red',
            'motion': 'left_to_right',
            'size_adj': 'medium',
            'speed_adj': 'slowly',
            'bg_color': 'gray'
        },
        {
            'shape': 'circle',
            'color': 'red',
            'motion': 'right_to_left',
            'size_adj': 'medium',
            'speed_adj': 'slowly',
            'bg_color': 'gray'
        },
        {
            'shape': 'square',
            'color': 'blue',
            'motion': 'diagonal_down_right',
            'size_adj': 'medium',
            'speed_adj': 'slowly',
            'bg_color': 'gray'
        },
        {
            'shape': 'circle',
            'color': 'yellow',
            'motion': 'stationary',
            'size_adj': 'medium',
            'speed_adj': 'slowly',
            'bg_color': 'gray'
        }
    ]

    # Create generator
    generator = SyntheticVideoGenerator(
        image_size=image_size,
        num_frames=num_frames,
        seed=seed
    )

    # Generate videos for each trajectory
    all_videos = []
    for traj_idx, traj in enumerate(trajectories):
        print(f"Generating trajectory {traj_idx + 1}/4: {traj['motion']} {traj['color']} {traj['shape']}")

        for i in range(num_videos_per_trajectory):
            frames, caption, metadata = generator.generate_single_video(
                shape=traj['shape'],
                color=traj['color'],
                motion=traj['motion'],
                size_adj=traj['size_adj'],
                speed_adj=traj['speed_adj'],
                bg_color=traj['bg_color']
            )

            metadata['trajectory_id'] = traj_idx
            metadata['trajectory_name'] = f"{traj['color']}_{traj['shape']}_{traj['motion']}"

            all_videos.append({
                'frames': frames,
                'caption': caption,
                'metadata': metadata
            })

    # Shuffle and split into train/val
    random.shuffle(all_videos)

    split_idx = int(len(all_videos) * 0.8)
    train_videos = all_videos[:split_idx]
    val_videos = all_videos[split_idx:]

    print(f"\nDataset split:")
    print(f"  Total videos: {len(all_videos)}")
    print(f"  Train: {len(train_videos)}")
    print(f"  Val: {len(val_videos)}")

    # Save splits
    for split, videos in [('train', train_videos), ('val', val_videos)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        video_dirs = []
        metadata_list = []
        captions = []

        for i, video_data in enumerate(videos):
            frames = video_data['frames']
            caption = video_data['caption']
            metadata = video_data['metadata']

            # Save video frames
            video_name = f"video_{i:05d}"
            video_dir = os.path.join(split_dir, video_name)
            os.makedirs(video_dir, exist_ok=True)

            for frame_idx, frame in enumerate(frames):
                frame_path = os.path.join(video_dir, f"frame_{frame_idx:04d}.png")
                frame.save(frame_path)

            # Save metadata
            metadata_path = os.path.join(video_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            video_dirs.append(video_dir)
            metadata_list.append(metadata)
            captions.append(caption)

        # Save captions
        captions_path = os.path.join(split_dir, "captions.txt")
        with open(captions_path, 'w') as f:
            for caption in captions:
                f.write(caption + '\n')

        # Save all metadata
        all_metadata_path = os.path.join(split_dir, "all_metadata.json")
        with open(all_metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)

        # Save preview grid
        if len(videos) > 0:
            generator._save_preview_grid(videos[0]['frames'], split_dir, f"preview_{split}.png")

        print(f"  {split.capitalize()} split saved to {split_dir}")

    print(f"\nControlled dataset generation complete!")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate controlled dataset for Stage 3.2.1")
    parser.add_argument("--output_dir", type=str, default="data/controlled_video", help="Output directory")
    parser.add_argument("--num_videos_per_trajectory", type=int, default=8, help="Videos per trajectory")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per video")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_controlled_dataset(
        output_dir=args.output_dir,
        num_videos_per_trajectory=args.num_videos_per_trajectory,
        num_frames=args.num_frames,
        image_size=args.image_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
