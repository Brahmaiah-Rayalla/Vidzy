"""
Stage 3.2.7B: Synthetic Video Dataset Generator

Creates approximately 256 synthetic video clips with:
- 16 frames, 64x64 RGB
- Balanced motion categories
- Varied attributes (position, size, velocity, shape, color, background)
- Deterministic generation with seed
- Train/val/test splits
- Ground truth trajectories for evaluation
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch


class SyntheticVideoGenerator:
    """Generate synthetic video clips with controlled motion."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Motion categories
        self.motion_categories = [
            'left_to_right',
            'right_to_left',
            'top_to_bottom',
            'bottom_to_top',
            'diagonal',
            'stationary'
        ]

        # Shapes
        self.shapes = ['circle', 'square', 'triangle']

        # Colors
        self.colors = [
            'red', 'blue', 'green', 'yellow', 'orange',
            'purple', 'pink', 'cyan', 'white', 'black'
        ]

        # Backgrounds
        self.backgrounds = [
            'white', 'light_gray', 'light_blue', 'light_yellow',
            'cream', 'gray', 'black'
        ]

        # Sizes
        self.sizes = ['small', 'medium', 'large']

        # Velocities
        self.velocities = ['slow', 'medium', 'fast']

    def generate_trajectory(
        self,
        motion_type: str,
        velocity: str,
        num_frames: int = 16,
        image_size: int = 64
    ) -> np.ndarray:
        """Generate trajectory coordinates for an object."""
        start_x = random.randint(10, image_size - 10)
        start_y = random.randint(10, image_size - 10)

        # Velocity to pixels per frame
        vel_map = {'slow': 1, 'medium': 2, 'fast': 3}
        vel = vel_map[velocity]

        trajectory = np.zeros((num_frames, 2), dtype=np.float32)

        for f in range(num_frames):
            if motion_type == 'stationary':
                trajectory[f] = [start_x, start_y]
            elif motion_type == 'left_to_right':
                trajectory[f] = [start_x + f * vel, start_y]
            elif motion_type == 'right_to_left':
                trajectory[f] = [start_x - f * vel, start_y]
            elif motion_type == 'top_to_bottom':
                trajectory[f] = [start_x, start_y + f * vel]
            elif motion_type == 'bottom_to_top':
                trajectory[f] = [start_x, start_y - f * vel]
            elif motion_type == 'diagonal':
                trajectory[f] = [start_x + f * vel, start_y + f * vel]

        return trajectory

    def draw_shape(
        self,
        frame: np.ndarray,
        shape: str,
        color: str,
        background: str,
        position: Tuple[float, float],
        size: str
    ) -> np.ndarray:
        """Draw a shape on a frame."""
        color_map = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'cyan': (0, 255, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }

        bg_map = {
            'white': (255, 255, 255),
            'light_gray': (211, 211, 211),
            'light_blue': (173, 216, 230),
            'light_yellow': (255, 255, 224),
            'cream': (255, 253, 208),
            'gray': (128, 128, 128),
            'black': (0, 0, 0)
        }

        size_map = {'small': 4, 'medium': 6, 'large': 8}

        # Fill background
        frame[:, :] = bg_map[background]

        # Draw shape
        rgb = color_map[color]
        radius = size_map[size]
        x, y = int(position[0]), int(position[1])

        # Simple shape drawing
        if shape == 'circle':
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        if 0 <= y+dy < 64 and 0 <= x+dx < 64:
                            frame[y+dy, x+dx] = rgb
        elif shape == 'square':
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if 0 <= y+dy < 64 and 0 <= x+dx < 64:
                        frame[y+dy, x+dx] = rgb
        elif shape == 'triangle':
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) <= radius - abs(dy) and abs(dy) <= radius:
                        if 0 <= y+dy < 64 and 0 <= x+dx < 64:
                            frame[y+dy, x+dx] = rgb

        return frame

    def generate_video(
        self,
        motion_type: str,
        shape: str,
        color: str,
        background: str,
        size: str,
        velocity: str,
        num_frames: int = 16,
        image_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Generate a single video clip."""
        trajectory = self.generate_trajectory(motion_type, velocity, num_frames, image_size)

        video = np.zeros((num_frames, image_size, image_size, 3), dtype=np.uint8)

        for f in range(num_frames):
            video[f] = self.draw_shape(
                video[f].copy(),
                shape,
                color,
                background,
                trajectory[f],
                size
            )

        # Generate caption
        caption = self.generate_caption(motion_type, shape, color, background, size, velocity)

        return video, trajectory, caption

    def generate_caption(
        self,
        motion_type: str,
        shape: str,
        color: str,
        background: str,
        size: str,
        velocity: str
    ) -> str:
        """Generate caption from parameters."""
        motion_phrases = {
            'left_to_right': 'moving from left to right',
            'right_to_left': 'moving from right to left',
            'top_to_bottom': 'moving from top to bottom',
            'bottom_to_top': 'moving from bottom to top',
            'diagonal': 'moving diagonally',
            'stationary': 'remaining stationary'
        }

        vel_phrases = {
            'slow': 'slowly',
            'medium': '',
            'fast': 'quickly'
        }

        motion_phrase = motion_phrases[motion_type]
        vel_phrase = vel_phrases[velocity]

        if motion_type == 'stationary':
            caption = f"A {size} {color} {shape} on a {background} background, {motion_phrase}."
        else:
            if vel_phrase:
                caption = f"A {size} {color} {shape} {vel_phrase} {motion_phrase} on a {background} background."
            else:
                caption = f"A {size} {color} {shape} {motion_phrase} on a {background} background."

        return caption

    def generate_dataset(
        self,
        total_clips: int = 256,
        num_frames: int = 16,
        image_size: int = 64,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict:
        """Generate complete dataset."""
        print(f"Generating {total_clips} synthetic video clips...")

        # Calculate clips per motion category
        clips_per_category = total_clips // len(self.motion_categories)

        dataset = []
        for motion_type in self.motion_categories:
            for i in range(clips_per_category):
                # Randomly sample attributes
                shape = random.choice(self.shapes)
                color = random.choice(self.colors)
                background = random.choice(self.backgrounds)
                size = random.choice(self.sizes)
                velocity = random.choice(self.velocities)

                # Generate video
                video, trajectory, caption = self.generate_video(
                    motion_type, shape, color, background, size, velocity,
                    num_frames, image_size
                )

                dataset.append({
                    'id': f"{motion_type}_{i}",
                    'motion_type': motion_type,
                    'shape': shape,
                    'color': color,
                    'background': background,
                    'size': size,
                    'velocity': velocity,
                    'video': video,
                    'trajectory': trajectory,
                    'caption': caption
                })

        # Shuffle dataset
        random.shuffle(dataset)

        # Split into train/val/test
        n_train = int(len(dataset) * train_ratio)
        n_val = int(len(dataset) * val_ratio)

        train_set = dataset[:n_train]
        val_set = dataset[n_train:n_train + n_val]
        test_set = dataset[n_train + n_val:]

        print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        return {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }

    def save_dataset(
        self,
        dataset: Dict,
        output_dir: str
    ):
        """Save dataset to disk."""
        script_dir = Path(__file__).parent
        output_path = script_dir / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        for split in ['train', 'val', 'test']:
            split_dir = output_path / split
            split_dir.mkdir(exist_ok=True)

            metadata = []
            for item in dataset[split]:
                # Save video as numpy array
                video_path = split_dir / f"{item['id']}.npy"
                np.save(video_path, item['video'])

                # Save trajectory
                traj_path = split_dir / f"{item['id']}_traj.npy"
                np.save(traj_path, item['trajectory'])

                # Metadata
                metadata.append({
                    'id': item['id'],
                    'motion_type': item['motion_type'],
                    'shape': item['shape'],
                    'color': item['color'],
                    'background': item['background'],
                    'size': item['size'],
                    'velocity': item['velocity'],
                    'caption': item['caption'],
                    'video_path': str(video_path),
                    'trajectory_path': str(traj_path)
                })

            # Save metadata
            metadata_path = output_path / f"{split}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"Dataset saved to {output_dir}")


def main():
    print("=" * 80)
    print("STAGE 3.2.7B: SYNTHETIC VIDEO DATASET GENERATION")
    print("=" * 80)

    generator = SyntheticVideoGenerator(seed=42)

    # Generate dataset
    dataset = generator.generate_dataset(
        total_clips=256,
        num_frames=16,
        image_size=64,
        train_ratio=0.7,
        val_ratio=0.15
    )

    # Save dataset
    output_dir = "data_synthetic"
    generator.save_dataset(dataset, output_dir)

    # Print statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    for split in ['train', 'val', 'test']:
        motion_counts = {}
        for item in dataset[split]:
            motion = item['motion_type']
            motion_counts[motion] = motion_counts.get(motion, 0) + 1

        print(f"\n{split.upper()} split ({len(dataset[split])} clips):")
        for motion, count in motion_counts.items():
            print(f"  {motion}: {count}")

    print("\n[SUCCESS] Dataset generation complete")


if __name__ == "__main__":
    main()
