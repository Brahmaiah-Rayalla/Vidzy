"""
Synthetic dataset generator for Stage 1.

Generates random geometric images (circles, squares, triangles) at 64x64 resolution.
No pretrained models or external APIs used.
"""

import numpy as np
from PIL import Image, ImageDraw
import os
from typing import Tuple, List
import random


class SyntheticShapeGenerator:
    """Generates synthetic geometric shapes for training."""

    def __init__(self, image_size: int = 64, seed: int = 42):
        """
        Initialize the shape generator.

        Args:
            image_size: Size of the square images (default: 64)
            seed: Random seed for reproducibility
        """
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)

    def generate_random_color(self) -> Tuple[int, int, int]:
        """Generate a random RGB color."""
        return tuple(self.rng.randint(0, 256, size=3).tolist())

    def draw_circle(self, draw: ImageDraw.Draw, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]):
        """Draw a circle in the given bounding box."""
        draw.ellipse(bbox, fill=color)

    def draw_square(self, draw: ImageDraw.Draw, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]):
        """Draw a square/rectangle in the given bounding box."""
        draw.rectangle(bbox, fill=color)

    def draw_triangle(self, draw: ImageDraw.Draw, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]):
        """Draw a triangle in the given bounding box."""
        x1, y1, x2, y2 = bbox
        # Triangle points: top, bottom-left, bottom-right
        points = [
            (x1 + (x2 - x1) // 2, y1),  # Top center
            (x1, y2),  # Bottom left
            (x2, y2)   # Bottom right
        ]
        draw.polygon(points, fill=color)

    def generate_single_image(self, num_shapes: int = None) -> Image.Image:
        """
        Generate a single synthetic image with random shapes.

        Args:
            num_shapes: Number of shapes to draw (random if None)

        Returns:
            PIL Image object
        """
        # Create image with random background color
        bg_color = self.generate_random_color()
        image = Image.new('RGB', (self.image_size, self.image_size), bg_color)
        draw = ImageDraw.Draw(image)

        # Random number of shapes (1-3)
        if num_shapes is None:
            num_shapes = self.rng.randint(1, 4)

        # Shape drawing functions
        shape_drawers = [self.draw_circle, self.draw_square, self.draw_triangle]

        for _ in range(num_shapes):
            # Random shape type
            shape_drawer = self.rng.choice(shape_drawers)

            # Random position and size
            max_size = self.image_size // 2
            size = self.rng.randint(10, max_size)
            x = self.rng.randint(0, self.image_size - size)
            y = self.rng.randint(0, self.image_size - size)
            bbox = (x, y, x + size, y + size)

            # Random color (ensure it's different from background for visibility)
            color = self.generate_random_color()

            # Draw the shape
            shape_drawer(draw, bbox, color)

        return image

    def generate_dataset(
        self,
        num_images: int,
        output_dir: str,
        save_preview: bool = True
    ) -> List[str]:
        """
        Generate a full dataset of synthetic images.

        Args:
            num_images: Number of images to generate
            output_dir: Directory to save images
            save_preview: Whether to save a preview grid

        Returns:
            List of generated image file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        image_paths = []
        preview_images = []

        for i in range(num_images):
            image = self.generate_single_image()
            filename = f"image_{i:05d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            image_paths.append(filepath)

            # Collect some images for preview
            if i < 16:
                preview_images.append(image)

        # Save preview grid
        if save_preview and preview_images:
            self._save_preview_grid(preview_images, output_dir)

        return image_paths

    def _save_preview_grid(self, images: List[Image.Image], output_dir: str):
        """Save a 4x4 grid of sample images."""
        grid_size = 4
        grid_width = grid_size * self.image_size
        grid_height = grid_size * self.image_size
        grid = Image.new('RGB', (grid_width, grid_height))

        for idx, img in enumerate(images):
            row = idx // grid_size
            col = idx % grid_size
            x = col * self.image_size
            y = row * self.image_size
            grid.paste(img, (x, y))

        preview_path = os.path.join(output_dir, "preview_grid.png")
        grid.save(preview_path)
        print(f"Preview grid saved to {preview_path}")


def main():
    """Main function to generate the dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic geometric dataset")
    parser.add_argument("--num_images", type=int, default=5000, help="Number of images to generate")
    parser.add_argument("--image_size", type=int, default=64, help="Image resolution")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Generating {args.num_images} synthetic images at {args.image_size}x{args.image_size}...")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")

    generator = SyntheticShapeGenerator(image_size=args.image_size, seed=args.seed)
    image_paths = generator.generate_dataset(
        num_images=args.num_images,
        output_dir=args.output_dir,
        save_preview=True
    )

    print(f"Dataset generation complete. {len(image_paths)} images saved.")


if __name__ == "__main__":
    main()
