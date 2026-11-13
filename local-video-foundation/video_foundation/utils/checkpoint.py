"""
Checkpoint management for saving and loading model states.
"""

import torch
import os
from typing import Dict, Any, Optional
import json


class CheckpointManager:
    """Manages model checkpoint saving and loading."""

    @staticmethod
    def save_checkpoint(
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        config: Dict[str, Any],
        seed: int,
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """
        Save a training checkpoint.

        Args:
            checkpoint_path: Path to save the checkpoint
            model: The model to save
            optimizer: The optimizer to save
            step: Current training step
            config: Configuration dictionary
            seed: Random seed used
            extra_data: Optional additional data to save
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'config': config,
            'seed': seed,
        }

        if extra_data:
            checkpoint.update(extra_data)

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            model: The model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load the checkpoint on

        Returns:
            Dictionary containing checkpoint metadata (step, config, seed, etc.)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")

        # Return metadata
        metadata = {
            'step': checkpoint.get('step', 0),
            'config': checkpoint.get('config', {}),
            'seed': checkpoint.get('seed', 42),
        }

        # Return any extra data
        for key in ['model_state_dict', 'optimizer_state_dict', 'step', 'config', 'seed']:
            checkpoint.pop(key, None)

        if checkpoint:
            metadata['extra_data'] = checkpoint

        return metadata

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """
        Save configuration to a JSON file.

        Args:
            config: Configuration dictionary
            config_path: Path to save the config
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_path}")

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.

        Args:
            config_path: Path to the config file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Config loaded from {config_path}")
        return config
