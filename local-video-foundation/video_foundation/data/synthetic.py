"""
Synthetic dataset generator for Stage 1.

Generates random geometric images (circles, squares, triangles) at 64x64 resolution.
No pretrained models or external APIs used.
"""

import numpy as np
from PIL import Image, ImageDraw
import os
import json
from typing import Tuple, List
import random


class SyntheticShapeGenerator:
    """Generates synthetic geometric shapes for training."""

    def __init__(self, image_size: int = 64, seed: int = 42, curriculum: bool = False, curriculum_stage: int = 0):
        """
        Initialize the shape generator.

        Args:
            image_size: Size of the square images (default: 64)
            seed: Random seed for reproducibility
        """
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)
        self.generated_count = 0
        self.last_class_label = 0
        self.last_color_label = 0
        self.curriculum = curriculum or curriculum_stage > 0
        self.curriculum_stage = curriculum_stage if self.curriculum else 0
        if self.curriculum_stage not in range(6):
            raise ValueError("curriculum_stage must be 0, 1, 2, 3, 4, or 5")
        self.palette = [
            (0, 0, 0), (255, 255, 255), (220, 40, 40),
            (40, 180, 60), (40, 80, 220), (245, 210, 40)
        ]

    def generate_random_color(self) -> Tuple[int, int, int]:
        """Generate a random RGB color."""
        if self.curriculum:
            return self.palette[int(self.rng.randint(0, len(self.palette)))]
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
        # Curriculum starts with a solid high-contrast background and one
        # centered shape; random mode preserves the original data distribution.
        bg_color = self.palette[0] if self.curriculum else self.generate_random_color()
        image = Image.new('RGB', (self.image_size, self.image_size), bg_color)
        draw = ImageDraw.Draw(image)

        if self.curriculum:
            num_shapes = 1 if self.curriculum_stage < 4 or self.curriculum_stage == 5 else self.rng.randint(1, 3)
        elif num_shapes is None:
            num_shapes = self.rng.randint(1, 4)

        # Shape drawing functions
        shape_drawers = [self.draw_circle, self.draw_square, self.draw_triangle]
        if self.curriculum_stage == 0:
            shape_drawers = [self.draw_circle]
        elif self.curriculum_stage in (1, 2):
            shape_drawers = [self.draw_circle, self.draw_square]

        for _ in range(num_shapes):
            # Random shape type
            if self.curriculum and self.curriculum_stage in (1, 2, 3, 5):
                shape_drawer = shape_drawers[self.generated_count % len(shape_drawers)]
            else:
                shape_drawer = self.rng.choice(shape_drawers)
            self.last_class_label = {
                self.draw_circle: 0,
                self.draw_square: 1,
                self.draw_triangle: 2,
            }[shape_drawer]

            # Random position and size
            max_size = self.image_size // 2
            if not self.curriculum or self.curriculum_stage >= 3 and self.curriculum_stage != 5:
                size = self.rng.randint(max(8, self.image_size // 4), max_size + 1)
            else:
                size = max(10, self.image_size // 2)
            if self.curriculum and (self.curriculum_stage < 3 or self.curriculum_stage == 5):
                x = y = (self.image_size - size) // 2
            else:
                x = self.rng.randint(0, self.image_size - size)
                y = self.rng.randint(0, self.image_size - size)
            bbox = (x, y, x + size, y + size)

            # Random color (ensure it's different from background for visibility)
            if self.curriculum and self.curriculum_stage < 2:
                color = self.palette[2]
            elif self.curriculum and self.curriculum_stage in (2, 3, 5):
                color_id = (self.generated_count // len(shape_drawers)) % 4
                color = self.palette[2 + color_id]
            else:
                color = self.generate_random_color()
            if self.curriculum and color == bg_color:
                color = self.palette[2]
            # Label foreground colors independently from shape so downstream
            # class conditioning can control both attributes without requiring
            # the denoiser to guess an unobserved palette mode.
            self.last_color_label = self.palette.index(color) - 2 if color in self.palette[2:] else 0

            # Draw the shape
            shape_drawer(draw, bbox, color)

        self.generated_count += 1
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
        class_labels = []
        color_labels = []

        for i in range(num_images):
            image = self.generate_single_image()
            class_labels.append(self.last_class_label)
            color_labels.append(self.last_color_label)
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

        with open(os.path.join(output_dir, "class_labels.json"), "w", encoding="utf-8") as f:
            json.dump(class_labels, f)
        with open(os.path.join(output_dir, "color_labels.json"), "w", encoding="utf-8") as f:
            json.dump(color_labels, f)

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
    parser.add_argument("--curriculum", action="store_true", help="Generate centered single shapes from a fixed high-contrast palette")
    parser.add_argument("--curriculum_stage", type=int, choices=range(6), default=None, help="Enable curriculum: 0 circle, 1 circle/square, 2 add colors, 3 vary size/position, 4 multiple shapes, 5 add triangles")
    args = parser.parse_args()

    print(f"Generating {args.num_images} synthetic images at {args.image_size}x{args.image_size}...")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")

    curriculum_stage = 0 if args.curriculum_stage is None else args.curriculum_stage
    generator = SyntheticShapeGenerator(
        image_size=args.image_size,
        seed=args.seed,
        curriculum=args.curriculum or args.curriculum_stage is not None,
        curriculum_stage=curriculum_stage,
    )
    image_paths = generator.generate_dataset(
        num_images=args.num_images,
        output_dir=args.output_dir,
        save_preview=True
    )

    print(f"Dataset generation complete. {len(image_paths)} images saved.")


if __name__ == "__main__":
    main()
