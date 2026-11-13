"""
PyTorch Dataset wrapper for synthetic geometric images.
"""

import os
from typing import Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class SyntheticImageDataset(Dataset):
    """PyTorch Dataset for synthetic geometric images."""

    def __init__(
        self,
        data_dir: str,
        image_size: int = 64,
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the synthetic images
            image_size: Size to resize images to (if needed)
            transform: Optional transform to apply to images
            normalize: Whether to normalize images to [-1, 1] range
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize

        # Find all image files
        self.image_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.png') and f != 'preview_grid.png'
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Found {len(self.image_files)} images in {data_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single image from the dataset.

        Returns:
            Image tensor of shape [3, image_size, image_size]
        """
        # Load image
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')

        # Resize if needed
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Normalize to [-1, 1] range
        if self.normalize:
            image_tensor = image_tensor / 255.0 * 2.0 - 1.0

        return image_tensor
