"""
Utilities for reproducibility.

Sets random seeds for Python, NumPy, and PyTorch.
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Automatically select the best available device.

    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    return device
