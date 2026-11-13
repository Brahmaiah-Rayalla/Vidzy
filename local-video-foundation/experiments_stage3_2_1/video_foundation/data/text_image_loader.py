"""
PyTorch Dataset for paired text-image data.
"""

import os
import json
from typing import Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class TextImageDataset(Dataset):
    """PyTorch Dataset for paired text-image data."""

    def __init__(
        self,
        metadata_path: str,
        image_dir: str,
        image_size: int = 64,
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            metadata_path: Path to metadata JSON file
            image_dir: Directory containing images
            image_size: Size to resize images to
            transform: Optional transform to apply to images
            normalize: Whether to normalize images to [-1, 1] range
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"Loaded {len(self.metadata)} examples from {metadata_path}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single example from the dataset.

        Returns:
            Dict with 'image' and 'caption' keys
        """
        entry = self.metadata[idx]

        # Load image
        image_path = os.path.join(self.image_dir, os.path.basename(entry['image_path']))
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

        return {
            'image': image_tensor,
            'caption': entry['caption'],
            'scene_params': entry.get('scene_params', {})
        }
