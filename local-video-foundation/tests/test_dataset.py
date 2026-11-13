"""
Tests for synthetic dataset generation.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.data.synthetic import SyntheticShapeGenerator
from video_foundation.data.dataset import SyntheticImageDataset


def test_synthetic_generation():
    """Test that synthetic images are generated correctly."""
    print("Testing synthetic dataset generation...")

    temp_dir = tempfile.mkdtemp()
    try:
        generator = SyntheticShapeGenerator(image_size=64, seed=42)

        # Generate a small dataset
        image_paths = generator.generate_dataset(
            num_images=10,
            output_dir=temp_dir,
            save_preview=False
        )

        # Check that images were created
        assert len(image_paths) == 10, f"Expected 10 images, got {len(image_paths)}"

        # Check image properties
        for path in image_paths:
            img = Image.open(path)
            assert img.size == (64, 64), f"Image size should be 64x64, got {img.size}"
            assert img.mode == 'RGB', f"Image mode should be RGB, got {img.mode}"
            img.close()

        print("[PASS] Synthetic generation test passed")
    finally:
        # Clean up
        import shutil
        import time
        time.sleep(0.1)  # Give Windows time to release file handles
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_loading():
    """Test that the PyTorch dataset loads images correctly."""
    print("Testing PyTorch dataset loading...")

    temp_dir = tempfile.mkdtemp()
    try:
        # Generate dataset
        generator = SyntheticShapeGenerator(image_size=64, seed=42)
        generator.generate_dataset(num_images=10, output_dir=temp_dir, save_preview=False)

        # Create PyTorch dataset
        dataset = SyntheticImageDataset(
            data_dir=temp_dir,
            image_size=64,
            normalize=True
        )

        # Check dataset size
        assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"

        # Check sample properties
        sample = dataset[0]
        assert sample.shape == (3, 64, 64), f"Expected shape (3, 64, 64), got {sample.shape}"
        assert sample.dtype == torch.float32, f"Expected float32, got {sample.dtype}"

        # Check normalization range
        assert sample.min() >= -1.0 and sample.max() <= 1.0, "Sample should be in [-1, 1] range"

        print("[PASS] Dataset loading test passed")
    finally:
        # Clean up
        import shutil
        import time
        time.sleep(0.1)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_reproducibility():
    """Test that the same seed produces the same images."""
    print("Testing reproducibility...")

    temp_dir1 = tempfile.mkdtemp()
    temp_dir2 = tempfile.mkdtemp()
    try:
        generator1 = SyntheticShapeGenerator(image_size=64, seed=42)
        generator2 = SyntheticShapeGenerator(image_size=64, seed=42)

        # Generate datasets with same seed
        generator1.generate_dataset(num_images=5, output_dir=temp_dir1, save_preview=False)
        generator2.generate_dataset(num_images=5, output_dir=temp_dir2, save_preview=False)

        # Load images and compare
        for i in range(5):
            img1 = np.array(Image.open(os.path.join(temp_dir1, f"image_{i:05d}.png")))
            img2 = np.array(Image.open(os.path.join(temp_dir2, f"image_{i:05d}.png")))

            assert np.array_equal(img1, img2), f"Image {i} should be identical with same seed"

        print("[PASS] Reproducibility test passed")
    finally:
        # Clean up
        import shutil
        import time
        time.sleep(0.1)
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)


if __name__ == "__main__":
    test_synthetic_generation()
    test_dataset_loading()
    test_reproducibility()

    print("\n" + "="*60)
    print("All dataset tests passed!")
    print("="*60)
