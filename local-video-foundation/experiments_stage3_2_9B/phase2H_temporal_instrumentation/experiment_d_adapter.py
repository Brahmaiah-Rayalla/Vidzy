"""
Stage 3.2.9F - Experiment D Adapter

Experiment D: Load Experiment C checkpoint with temporal processing explicitly disabled.
This is the temporal-disabled ablation for comparing against Experiment C.

Author: Devin
Date: 2026-11-02
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import sys

from conditional_unet_stage2 import ConditionalUNet
from video_unet_feature_fusion import VideoUNetFixed
from text_encoder_split import TextEncoderSplit


class ExperimentDAdapter:
    """Adapter for Experiment D: Experiment C checkpoint with temporal processing disabled."""
    
    def __init__(self, checkpoint_path: str, config: Dict[str, Any]):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.device = config.get('device', 'cpu')
        
    def load(self) -> nn.Module:
        """Load Experiment C checkpoint with temporal processing disabled."""
        print(f"Loading Experiment D (C checkpoint, temporal disabled) from {self.checkpoint_path}")
        
        # Create spatial UNet component
        spatial_unet = ConditionalUNet(
            in_channels=self.config.get('in_channels', 3),
            model_channels=self.config.get('model_channels', 64),
            channel_mult=self.config.get('channel_mult', [1, 2, 2, 2]),
            num_res_blocks=self.config.get('num_res_blocks', 2),
            attention_resolutions=self.config.get('attention_resolutions', []),
            dropout=self.config.get('dropout', 0.0),
            num_timesteps=self.config.get('num_timesteps', 1000),
            text_dim=self.config.get('text_dim', 256),
            n_heads=self.config.get('n_heads', 4),
            use_cross_attn=self.config.get('use_cross_attn', True)
        )
        
        # Create text encoder component
        text_encoder = TextEncoderSplit(
            original_vocab_size=self.config.get('vocab_size', 42),
            motion_vocab_size=self.config.get('motion_vocab_size', 40),
            d_model=self.config.get('text_dim', 256),
            n_heads=self.config.get('n_heads', 4),
            n_layers=self.config.get('n_layers', 4),
            d_ff=self.config.get('d_ff', 512),
            max_seq_len=self.config.get('max_seq_len', 64),
            dropout=self.config.get('dropout', 0.1),
            freeze_transformer=True
        )
        
        # Create VideoUNetFixed model with use_temporal=False (EXPERIMENT D KEY DIFFERENCE)
        model = VideoUNetFixed(
            spatial_unet=spatial_unet,
            text_encoder=text_encoder,
            num_frames=self.config.get('num_frames', 16),
            image_size=self.config.get('image_size', 64),
            temporal_kernel=self.config.get('temporal_kernel', 3),
            temporal_layers=self.config.get('temporal_layers', 2),
            freeze_spatial=self.config.get('freeze_spatial', True),
            use_temporal=False  # EXPERIMENT D: Explicitly disable temporal processing
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load state dict from nested structure
        # Use strict=False to ignore temporal modules in checkpoint
        if 'video_unet' in checkpoint:
            model.load_state_dict(checkpoint['video_unet'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        model = model.to(self.device)
        model.eval()
        self.model = model
        
        print(f"  [OK] Experiment D model loaded successfully")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  use_temporal: False (explicitly disabled)")
        print(f"  Checkpoint: Experiment C (temporal-enabled training)")
        
        return model
        
    def denoise(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor,
                motion_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Denoise using temporal-disabled forward signature."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        with torch.no_grad():
            output = self.model(x, t, text_emb)
            
        return output


def create_experiment_d_adapter(checkpoint_path: str) -> ExperimentDAdapter:
    """
    Factory function to create Experiment D adapter.
    
    Args:
        checkpoint_path: Path to Experiment C checkpoint
        
    Returns:
        Experiment D adapter instance
    """
    config = {
        'in_channels': 3,
        'model_channels': 64,
        'channel_mult': [1, 2, 2, 2],
        'num_res_blocks': 2,
        'attention_resolutions': [],
        'dropout': 0.0,
        'num_timesteps': 1000,
        'text_dim': 256,
        'n_heads': 4,
        'use_cross_attn': True,
        'vocab_size': 42,
        'motion_vocab_size': 40,
        'n_layers': 4,
        'd_ff': 512,
        'max_seq_len': 64,
        'num_frames': 16,
        'image_size': 64,
        'temporal_kernel': 3,
        'temporal_layers': 2,
        'freeze_spatial': True,
        'use_temporal': False,  # Key difference: temporal disabled
        'device': 'cpu'
    }
    return ExperimentDAdapter(checkpoint_path, config)
