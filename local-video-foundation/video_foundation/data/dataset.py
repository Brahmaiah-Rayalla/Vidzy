"""
PyTorch Dataset wrapper for synthetic geometric images.
"""

import os
import json
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
        normalize: bool = True,
        cache_in_memory: bool = False,
        return_label: bool = False,
        return_color_label: bool = False
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
        self.cache_in_memory = cache_in_memory
        self.return_label = return_label
        self.return_color_label = return_color_label
        self._cache = {}

        # Find all image files
        self.image_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.png') and f != 'preview_grid.png'
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")

        if self.return_label:
            labels_path = os.path.join(data_dir, "class_labels.json")
            if not os.path.isfile(labels_path):
                raise ValueError(f"Class-conditioned training requires {labels_path}")
            with open(labels_path, "r", encoding="utf-8") as f:
                self.class_labels = json.load(f)
            if len(self.class_labels) != len(self.image_files):
                raise ValueError(
                    f"Found {len(self.class_labels)} class labels for "
                    f"{len(self.image_files)} images in {data_dir}"
                )
        if self.return_color_label:
            labels_path = os.path.join(data_dir, "color_labels.json")
            if not os.path.isfile(labels_path):
                raise ValueError(f"Color-conditioned training requires {labels_path}")
            with open(labels_path, "r", encoding="utf-8") as f:
                self.color_labels = json.load(f)
            if len(self.color_labels) != len(self.image_files):
                raise ValueError(f"Found {len(self.color_labels)} color labels for {len(self.image_files)} images")

        print(f"Found {len(self.image_files)} images in {data_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single image from the dataset.

        Returns:
            Image tensor of shape [3, image_size, image_size]
        """
        if self.cache_in_memory and idx in self._cache:
            image_tensor = self._cache[idx]
            return self._format_sample(image_tensor, idx)

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

        if self.cache_in_memory:
            self._cache[idx] = image_tensor
        return self._format_sample(image_tensor, idx)

    def _format_sample(self, image_tensor, idx):
        labels = []
        if self.return_label:
            labels.append(self.class_labels[idx])
        if self.return_color_label:
            labels.append(self.color_labels[idx])
        if not labels:
            return image_tensor
        return (image_tensor, *labels)
