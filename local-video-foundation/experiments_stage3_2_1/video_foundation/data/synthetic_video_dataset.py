"""
Synthetic video dataset generator for Stage 3 video generation.

Generates paired video-caption samples with moving geometric shapes.
Videos are generated through consistent mathematical trajectories to ensure
temporal coherence between frames.
"""

import numpy as np
from PIL import Image, ImageDraw
import os
from typing import Tuple, List, Dict, Optional
import random
import json


class SyntheticVideoGenerator:
    """Generates synthetic video clips with moving geometric shapes."""

    # Predefined colors
    COLORS = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'gray': (128, 128, 128),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'cyan': (0, 255, 255),
        'pink': (255, 192, 203),
    }

    # Shape types
    SHAPES = ['circle', 'square', 'triangle']

    # Motion types
    MOTIONS = [
        'left_to_right',
        'right_to_left',
        'top_to_bottom',
        'bottom_to_top',
        'diagonal_down_right',
        'diagonal_down_left',
        'stationary',
    ]

    # Size adjectives
    SIZE_ADJECTIVES = {
        'small': (10, 20),
        'medium': (20, 35),
        'large': (35, 50),
    }

    # Speed adjectives
    SPEED_ADJECTIVES = {
        'slowly': (0.5, 1.5),
        'quickly': (2.0, 4.0),
    }

    def __init__(
        self,
        image_size: int = 64,
        num_frames: int = 16,
        seed: int = 42
    ):
        """
        Initialize the video generator.

        Args:
            image_size: Size of the square images (default: 64)
            num_frames: Number of frames per video clip (default: 16)
            seed: Random seed for reproducibility
        """
        self.image_size = image_size
        self.num_frames = num_frames
        self.rng = np.random.RandomState(seed)

    def _get_color(self, color_name: str) -> Tuple[int, int, int]:
        """Get RGB tuple for a color name."""
        return self.COLORS.get(color_name, (128, 128, 128))

    def _get_random_color(self) -> Tuple[str, Tuple[int, int, int]]:
        """Get a random color name and RGB tuple."""
        color_name = self.rng.choice(list(self.COLORS.keys()))
        return color_name, self.COLORS[color_name]

    def _draw_circle(
        self,
        draw: ImageDraw.Draw,
        center: Tuple[int, int],
        size: int,
        color: Tuple[int, int, int]
    ):
        """Draw a circle at the given center with given size."""
        x, y = center
        bbox = (x - size//2, y - size//2, x + size//2, y + size//2)
        draw.ellipse(bbox, fill=color)

    def _draw_square(
        self,
        draw: ImageDraw.Draw,
        center: Tuple[int, int],
        size: int,
        color: Tuple[int, int, int]
    ):
        """Draw a square at the given center with given size."""
        x, y = center
        bbox = (x - size//2, y - size//2, x + size//2, y + size//2)
        draw.rectangle(bbox, fill=color)

    def _draw_triangle(
        self,
        draw: ImageDraw.Draw,
        center: Tuple[int, int],
        size: int,
        color: Tuple[int, int, int]
    ):
        """Draw a triangle at the given center with given size."""
        x, y = center
        # Triangle points: top, bottom-left, bottom-right
        points = [
            (x, y - size//2),  # Top center
            (x - size//2, y + size//2),  # Bottom left
            (x + size//2, y + size//2)   # Bottom right
        ]
        draw.polygon(points, fill=color)

    def _compute_trajectory(
        self,
        start_pos: Tuple[float, float],
        motion_type: str,
        velocity: float,
        num_frames: int
    ) -> List[Tuple[float, float]]:
        """
        Compute object trajectory across frames.

        Args:
            start_pos: Starting (x, y) position
            motion_type: Type of motion
            velocity: Speed of movement (pixels per frame)
            num_frames: Number of frames

        Returns:
            List of (x, y) positions for each frame
        """
        x, y = start_pos
        positions = []

        for i in range(num_frames):
            positions.append((x, y))

            # Update position based on motion type
            if motion_type == 'left_to_right':
                x += velocity
            elif motion_type == 'right_to_left':
                x -= velocity
            elif motion_type == 'top_to_bottom':
                y += velocity
            elif motion_type == 'bottom_to_top':
                y -= velocity
            elif motion_type == 'diagonal_down_right':
                x += velocity
                y += velocity
            elif motion_type == 'diagonal_down_left':
                x -= velocity
                y += velocity
            elif motion_type == 'stationary':
                pass  # No movement

        return positions

    def _generate_caption(
        self,
        shape: str,
        color: str,
        size_adj: str,
        speed_adj: str,
        motion: str,
        bg_color: str
    ) -> str:
        """
        Generate a caption describing the video.

        Args:
            shape: Shape type
            color: Color name
            size_adj: Size adjective
            speed_adj: Speed adjective
            motion: Motion type
            bg_color: Background color name

        Returns:
            Caption string
        """
        # Motion descriptions
        motion_descriptions = {
            'left_to_right': 'moves from left to right',
            'right_to_left': 'moves from right to left',
            'top_to_bottom': 'moves from top to bottom',
            'bottom_to_top': 'moves from bottom to top',
            'diagonal_down_right': 'moves diagonally down and right',
            'diagonal_down_left': 'moves diagonally down and left',
            'stationary': 'remains stationary',
        }

        # Build caption
        parts = [
            f"A {size_adj}",
            color,
            shape,
            f"{speed_adj}",
            motion_descriptions[motion],
            f"on a {bg_color} background."
        ]

        return ' '.join(parts)

    def generate_single_video(
        self,
        shape: Optional[str] = None,
        color: Optional[str] = None,
        motion: Optional[str] = None,
        size_adj: Optional[str] = None,
        speed_adj: Optional[str] = None,
        bg_color: Optional[str] = None,
        start_pos: Optional[Tuple[float, float]] = None
    ) -> Tuple[List[Image.Image], str, Dict]:
        """
        Generate a single video clip with caption.

        Args:
            shape: Shape type (random if None)
            color: Color name (random if None)
            motion: Motion type (random if None)
            size_adj: Size adjective (random if None)
            speed_adj: Speed adjective (random if None)
            bg_color: Background color name (random if None)
            start_pos: Starting position (random if None)

        Returns:
            Tuple of (frames, caption, metadata)
        """
        # Random selections if not specified
        if shape is None:
            shape = self.rng.choice(self.SHAPES)
        if color is None:
            color, _ = self._get_random_color()
        if motion is None:
            motion = self.rng.choice(self.MOTIONS)
        if size_adj is None:
            size_adj = self.rng.choice(list(self.SIZE_ADJECTIVES.keys()))
        if speed_adj is None:
            speed_adj = self.rng.choice(list(self.SPEED_ADJECTIVES.keys()))
        if bg_color is None:
            bg_color, _ = self._get_random_color()
            # Ensure background is different from object
            while bg_color == color:
                bg_color, _ = self._get_random_color()

        # Get RGB values
        obj_rgb = self._get_color(color)
        bg_rgb = self._get_color(bg_color)

        # Get size range
        size_range = self.SIZE_ADJECTIVES[size_adj]
        size = self.rng.randint(size_range[0], size_range[1])

        # Get velocity range
        vel_range = self.SPEED_ADJECTIVES[speed_adj]
        velocity = self.rng.uniform(vel_range[0], vel_range[1])

        # Random starting position (ensure object stays in frame)
        margin = size
        max_start = self.image_size - margin
        if start_pos is None:
            start_x = self.rng.uniform(margin, max_start)
            start_y = self.rng.uniform(margin, max_start)
            start_pos = (start_x, start_y)

        # Compute trajectory
        positions = self._compute_trajectory(start_pos, motion, velocity, self.num_frames)

        # Generate frames
        frames = []
        for pos in positions:
            # Create image with background color
            image = Image.new('RGB', (self.image_size, self.image_size), bg_rgb)
            draw = ImageDraw.Draw(image)

            # Draw shape at current position
            if shape == 'circle':
                self._draw_circle(draw, pos, size, obj_rgb)
            elif shape == 'square':
                self._draw_square(draw, pos, size, obj_rgb)
            elif shape == 'triangle':
                self._draw_triangle(draw, pos, size, obj_rgb)

            frames.append(image)

        # Generate caption
        caption = self._generate_caption(
            shape, color, size_adj, speed_adj, motion, bg_color
        )

        # Metadata
        metadata = {
            'shape': shape,
            'color': color,
            'motion': motion,
            'size_adjective': size_adj,
            'speed_adjective': speed_adj,
            'background_color': bg_color,
            'object_size': size,
            'velocity': velocity,
            'start_position': start_pos,
            'trajectory': positions,
            'num_frames': self.num_frames,
            'image_size': self.image_size,
        }

        return frames, caption, metadata

    def generate_dataset(
        self,
        num_videos: int,
        output_dir: str,
        split: str = 'train',
        save_preview: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Generate a full dataset of synthetic videos.

        Args:
            num_videos: Number of videos to generate
            output_dir: Directory to save videos
            split: Dataset split ('train', 'val', 'test')
            save_preview: Whether to save a preview grid

        Returns:
            Tuple of (video_dirs, metadata_paths)
        """
        output_dir = os.path.join(output_dir, split)
        os.makedirs(output_dir, exist_ok=True)

        video_dirs = []
        metadata_list = []
        captions = []

        for i in range(num_videos):
            # Generate video
            frames, caption, metadata = self.generate_single_video()

            # Save video frames
            video_name = f"video_{i:05d}"
            video_dir = os.path.join(output_dir, video_name)
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

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_videos} videos")

        # Save captions
        captions_path = os.path.join(output_dir, "captions.txt")
        with open(captions_path, 'w') as f:
            for caption in captions:
                f.write(caption + '\n')

        # Save all metadata
        all_metadata_path = os.path.join(output_dir, "all_metadata.json")
        with open(all_metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)

        # Save preview grid
        if save_preview and num_videos > 0:
            self._save_preview_grid(frames, output_dir, f"preview_{split}.png")

        print(f"Dataset generation complete for {split} split.")
        print(f"  Videos: {len(video_dirs)}")
        print(f"  Output directory: {output_dir}")

        return video_dirs, metadata_list

    def _save_preview_grid(
        self,
        frames: List[Image.Image],
        output_dir: str,
        filename: str
    ):
        """Save a grid of frames from a video."""
        # Show first 16 frames in a 4x4 grid
        num_frames_to_show = min(16, len(frames))
        grid_size = 4

        grid_width = grid_size * self.image_size
        grid_height = grid_size * self.image_size
        grid = Image.new('RGB', (grid_width, grid_height))

        for idx in range(num_frames_to_show):
            img = frames[idx]
            row = idx // grid_size
            col = idx % grid_size
            x = col * self.image_size
            y = row * self.image_size
            grid.paste(img, (x, y))

        preview_path = os.path.join(output_dir, filename)
        grid.save(preview_path)
        print(f"Preview grid saved to {preview_path}")


def main():
    """Main function to generate the video dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic video dataset")
    parser.add_argument("--num_videos", type=int, default=5000, help="Number of videos to generate")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--image_size", type=int, default=64, help="Image resolution")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_video", help="Output directory")
    parser.add_argument("--split", type=str, default="train", choices=['train', 'val', 'test'], help="Dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Generating {args.num_videos} synthetic videos...")
    print(f"  Frames per video: {args.num_frames}")
    print(f"  Resolution: {args.image_size}x{args.image_size}")
    print(f"  Split: {args.split}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed: {args.seed}")

    generator = SyntheticVideoGenerator(
        image_size=args.image_size,
        num_frames=args.num_frames,
        seed=args.seed
    )

    video_dirs, metadata_list = generator.generate_dataset(
        num_videos=args.num_videos,
        output_dir=args.output_dir,
        split=args.split,
        save_preview=True
    )

    print(f"Dataset generation complete.")


if __name__ == "__main__":
    main()
