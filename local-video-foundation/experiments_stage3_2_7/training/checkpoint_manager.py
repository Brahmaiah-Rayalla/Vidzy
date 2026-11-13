"""
Stage 3.2.7B: Checkpoint Manager

Handles checkpoint saving, loading, and validation.
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


class CheckpointManager:
    """Manage checkpoints for training experiments."""

    def __init__(self, experiment_name: str, checkpoint_dir: str):
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: float,
        additional_info: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """Save a training checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step_{step:06d}.pt"
        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'step': step,
            'loss': loss,
            'timestamp': timestamp,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'experiment_name': self.experiment_name
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, filepath)

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)

        return str(filepath)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """Load a training checkpoint."""
        if checkpoint_path is None:
            # Load latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def load_spatial_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        freeze_spatial: bool = True
    ):
        """Load spatial checkpoint and optionally freeze it."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load spatial U-Net state
        if 'spatial_unet_state_dict' in checkpoint:
            model.spatial_unet.load_state_dict(checkpoint['spatial_unet_state_dict'], strict=True)
        elif 'model_state_dict' in checkpoint:
            # Handle different checkpoint format
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if freeze_spatial:
            for param in model.spatial_unet.parameters():
                param.requires_grad = False

    def load_text_encoder_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        freeze_original: bool = True
    ):
        """Load text encoder checkpoint and optionally freeze original embeddings."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load text encoder state
        if 'text_encoder_state_dict' in checkpoint:
            model.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'], strict=True)

        if freeze_original:
            # Freeze original embeddings (indices 0-41)
            token_emb = model.text_encoder.token_emb
            for i in range(42):
                token_emb.weight[i].requires_grad = False

    def save_metadata(self, metadata: Dict[str, Any]):
        """Save experiment metadata."""
        metadata_path = self.checkpoint_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self) -> Dict[str, Any]:
        """Load experiment metadata."""
        metadata_path = self.checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """Get information about a checkpoint without loading the full model."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return {
            'step': checkpoint.get('step', 'N/A'),
            'loss': checkpoint.get('loss', 'N/A'),
            'timestamp': checkpoint.get('timestamp', 'N/A'),
            'experiment_name': checkpoint.get('experiment_name', 'N/A')
        }

    def list_checkpoints(self) -> list:
        """List all checkpoints in the directory."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
