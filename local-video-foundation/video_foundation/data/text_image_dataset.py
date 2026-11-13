"""
Paired text-image dataset generator for synthetic geometric shapes.

Generates images with corresponding captions that truthfully describe
the visual content based on scene parameters.
"""

import os
import json
import random
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw
import numpy as np
import torch


class TextImageShapeGenerator:
    """
    Generates paired synthetic images and captions.

    Supports:
    - Multiple shapes (circle, square, triangle)
    - Multiple colors (red, green, blue, yellow, etc.)
    - Multiple sizes (small, medium, large)
    - Multiple positions (center, left, right, top, bottom)
    - Multiple objects per image
    - Multiple caption templates
    """

    # Available colors with RGB values
    COLORS = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'cyan': (0, 255, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'gray': (128, 128, 128),
        'light_gray': (220, 220, 220),
        'cream': (255, 253, 208),
        'light_blue': (173, 216, 230),
        'light_yellow': (255, 255, 224),
    }

    # Background colors (lighter colors for better contrast)
    BG_COLORS = ['white', 'light_gray', 'cream', 'light_blue', 'light_yellow']

    # Sizes as fractions of image size
    SIZES = {
        'small': 0.15,
        'medium': 0.25,
        'large': 0.35,
    }

    # Position ranges as fractions of image size (x_range, y_range)
    POSITIONS = {
        'center': ((0.35, 0.65), (0.35, 0.65)),
        'left': ((0.1, 0.3), (0.2, 0.8)),
        'right': ((0.7, 0.9), (0.2, 0.8)),
        'top': ((0.2, 0.8), (0.1, 0.3)),
        'bottom': ((0.2, 0.8), (0.7, 0.9)),
    }

    # Caption templates
    SINGLE_OBJECT_TEMPLATES = [
        "A {size} {color} {shape} on a {bg_color} background.",
        "A {color} {shape} of {size} size on {bg_color}.",
        "On a {bg_color} background, there is a {size} {color} {shape}.",
        "A {size} {color} {shape} positioned in the {position} on {bg_color}.",
        "A {color} {shape} in the {position}, {size} in size, on {bg_color}.",
    ]

    MULTI_OBJECT_TEMPLATES = [
        "A {size1} {color1} {shape1} and a {size2} {color2} {shape2} on {bg_color}.",
        "On {bg_color}, there is a {size1} {color1} {shape1} and a {size2} {color2} {shape2}.",
        "A {color1} {shape1} in the {position1} and a {color2} {shape2} in the {position2} on {bg_color}.",
    ]

    def __init__(
        self,
        image_size: int = 64,
        seed: int = 42,
        max_objects: int = 2,
        bg_colors: Optional[List[str]] = None
    ):
        """
        Initialize the generator.

        Args:
            image_size: Size of generated images (square)
            seed: Random seed for reproducibility
            max_objects: Maximum number of objects per image
            bg_colors: List of background color names
        """
        self.image_size = image_size
        self.seed = seed
        self.max_objects = max_objects
        self.bg_colors = bg_colors or self.BG_COLORS

        # Set seed
        random.seed(seed)
        np.random.seed(seed)

    def _sample_color(self, exclude: Optional[List[str]] = None) -> str:
        """Sample a random color, excluding specified colors."""
        available = [c for c in self.COLORS.keys() if c not in (exclude or [])]
        return random.choice(available)

    def _sample_shape(self) -> str:
        """Sample a random shape."""
        return random.choice(['circle', 'square', 'triangle'])

    def _sample_size(self) -> str:
        """Sample a random size."""
        return random.choice(list(self.SIZES.keys()))

    def _sample_position(self) -> str:
        """Sample a random position."""
        return random.choice(list(self.POSITIONS.keys()))

    def _get_position_coords(self, position: str, size_fraction: float) -> Tuple[int, int]:
        """
        Get (x, y) coordinates for a given position.

        Args:
            position: Position name (center, left, right, top, bottom)
            size_fraction: Size of object as fraction of image

        Returns:
            (x, y) center coordinates
        """
        x_range, y_range = self.POSITIONS[position]

        # Randomize within the position range
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)

        return int(x * self.image_size), int(y * self.image_size)

    def _draw_shape(
        self,
        draw: ImageDraw.ImageDraw,
        shape: str,
        color: Tuple[int, int, int],
        x: int,
        y: int,
        size: int
    ) -> None:
        """
        Draw a shape on the image.

        Args:
            draw: PIL ImageDraw object
            shape: Shape name (circle, square, triangle)
            color: RGB color tuple
            x: Center x coordinate
            y: Center y coordinate
            size: Size in pixels
        """
        if shape == 'circle':
            draw.ellipse([x - size, y - size, x + size, y + size], fill=color)
        elif shape == 'square':
            draw.rectangle([x - size, y - size, x + size, y + size], fill=color)
        elif shape == 'triangle':
            # Draw triangle pointing up
            points = [
                (x, y - size),  # Top
                (x - size, y + size),  # Bottom left
                (x + size, y + size),  # Bottom right
            ]
            draw.polygon(points, fill=color)

    def generate_scene(self) -> Tuple[Image.Image, Dict]:
        """
        Generate a single scene with one or more objects.

        Returns:
            Tuple of (PIL Image, scene parameters dict)
        """
        # Choose background color
        bg_color_name = random.choice(self.bg_colors)
        bg_color_rgb = self.COLORS.get(bg_color_name, (255, 255, 255))

        # Create image
        image = Image.new('RGB', (self.image_size, self.image_size), bg_color_rgb)
        draw = ImageDraw.Draw(image)

        # Determine number of objects
        num_objects = random.randint(1, self.max_objects)

        objects = []
        used_colors = []

        for i in range(num_objects):
            # Sample attributes
            shape = self._sample_shape()
            color = self._sample_color(exclude=used_colors)
            used_colors.append(color)
            size_name = self._sample_size()
            position = self._sample_position()

            # Get size in pixels
            size_fraction = self.SIZES[size_name]
            size_pixels = int(size_fraction * self.image_size)

            # Get position coordinates
            x, y = self._get_position_coords(position, size_fraction)

            # Draw shape
            color_rgb = self.COLORS[color]
            self._draw_shape(draw, shape, color_rgb, x, y, size_pixels)

            # Store object parameters
            objects.append({
                'shape': shape,
                'color': color,
                'size': size_name,
                'position': position,
                'x': x,
                'y': y,
                'size_pixels': size_pixels
            })

        scene_params = {
            'bg_color': bg_color_name,
            'objects': objects
        }

        return image, scene_params

    def generate_caption(self, scene_params: Dict) -> str:
        """
        Generate a caption for a scene.

        Args:
            scene_params: Scene parameters dict

        Returns:
            Caption string
        """
        objects = scene_params['objects']
        bg_color = scene_params['bg_color']

        # Single object
        if len(objects) == 1:
            obj = objects[0]
            template = random.choice(self.SINGLE_OBJECT_TEMPLATES)
            caption = template.format(
                size=obj['size'],
                color=obj['color'],
                shape=obj['shape'],
                position=obj['position'],
                bg_color=bg_color
            )
        # Multiple objects
        else:
            obj1, obj2 = objects[0], objects[1]
            template = random.choice(self.MULTI_OBJECT_TEMPLATES)
            caption = template.format(
                size1=obj1['size'],
                color1=obj1['color'],
                shape1=obj1['shape'],
                position1=obj1['position'],
                size2=obj2['size'],
                color2=obj2['color'],
                shape2=obj2['shape'],
                position2=obj2['position'],
                bg_color=bg_color
            )

        return caption

    def generate_dataset(
        self,
        num_images: int,
        output_dir: str,
        save_preview: bool = True,
        preview_grid_size: int = 16
    ) -> List[Dict]:
        """
        Generate a complete dataset of paired images and captions.

        Args:
            num_images: Number of images to generate
            output_dir: Directory to save outputs
            save_preview: Whether to save a preview grid
            preview_grid_size: Number of images in preview grid

        Returns:
            List of metadata dicts for each image
        """
        os.makedirs(output_dir, exist_ok=True)

        metadata = []

        print(f"Generating {num_images} paired image-caption examples...")

        for i in range(num_images):
            # Generate scene
            image, scene_params = self.generate_scene()

            # Generate caption
            caption = self.generate_caption(scene_params)

            # Save image
            image_path = os.path.join(output_dir, f"image_{i:05d}.png")
            image.save(image_path)

            # Store metadata
            metadata.append({
                'image_path': image_path,
                'caption': caption,
                'scene_params': scene_params
            })

            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_images} examples")

        # Save metadata
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {metadata_path}")

        # Save captions separately for easy access
        captions_path = os.path.join(output_dir, 'captions.txt')
        with open(captions_path, 'w') as f:
            for m in metadata:
                f.write(m['caption'] + '\n')

        print(f"Saved captions to {captions_path}")

        # Save preview grid
        if save_preview:
            self._save_preview_grid(output_dir, preview_grid_size)

        print(f"Dataset generation complete: {output_dir}")
        return metadata

    def _save_preview_grid(self, output_dir: str, grid_size: int) -> None:
        """Save a preview grid of sample images."""
        # Sample a subset of images
        metadata_files = [f for f in os.listdir(output_dir) if f.endswith('.png') and f != 'preview_grid.png']
        sample_files = random.sample(metadata_files, min(grid_size, len(metadata_files)))
        sample_files.sort()

        # Load images
        images = []
        for f in sample_files:
            img = Image.open(os.path.join(output_dir, f))
            images.append(img)

        # Create grid
        cols = int(np.sqrt(grid_size))
        rows = (grid_size + cols - 1) // cols

        img_width, img_height = images[0].size
        grid_width = cols * img_width
        grid_height = rows * img_height

        grid = Image.new('RGB', (grid_width, grid_height))
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            grid.paste(img, (col * img_width, row * img_height))

        preview_path = os.path.join(output_dir, 'preview_grid.png')
        grid.save(preview_path)
        print(f"Saved preview grid to {preview_path}")


def create_train_val_test_splits(
    metadata: List[Dict],
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    holdout_combinations: Optional[Dict] = None
) -> Dict[str, List[Dict]]:
    """
    Create train/val/test splits from metadata.

    Args:
        metadata: List of metadata dicts
        output_dir: Directory to save split files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        holdout_combinations: Dict of attribute combinations to hold out for test

    Returns:
        Dict with 'train', 'val', 'test' keys containing metadata lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Shuffle metadata
    random.shuffle(metadata)

    # Simple split
    n = len(metadata)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        'train': metadata[:n_train],
        'val': metadata[n_train:n_train + n_val],
        'test': metadata[n_train + n_val:]
    }

    # Save splits
    for split_name, split_data in splits.items():
        split_path = os.path.join(output_dir, f'{split_name}_metadata.json')
        with open(split_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name} split ({len(split_data)} examples) to {split_path}")

    return splits


if __name__ == "__main__":
    # Test the generator
    generator = TextImageShapeGenerator(image_size=64, seed=42, max_objects=2)

    # Generate a small test dataset
    metadata = generator.generate_dataset(
        num_images=100,
        output_dir='data/text_image_test',
        save_preview=True
    )

    # Create splits
    create_train_val_test_splits(metadata, 'data/text_image_test')

    # Print some examples
    print("\nSample captions:")
    for i in range(5):
        print(f"  {i+1}. {metadata[i]['caption']}")
