"""
Stage 3.2.9D - Architecture-Specific Inference Adapters

Provides adapter classes for loading and using different model architectures
with their respective checkpoints.

Author: Devin
Date: 2026-11-02
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import sys


class InferenceAdapter:
    """Common interface for all model types"""
    
    def __init__(self, checkpoint_path: str, config: Dict[str, Any]):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def load(self) -> nn.Module:
        """Load appropriate model based on checkpoint type"""
        raise NotImplementedError
        
    def denoise(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor, 
                motion_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Common denoising interface"""
        raise NotImplementedError


class Stage2Adapter(InferenceAdapter):
    """Adapter for Stage 2.1 ConditionalUNet checkpoints (image generation)"""
    
    def __init__(self, checkpoint_path: str, config: Dict[str, Any]):
        super().__init__(checkpoint_path, config)
        # Import Stage 2.1 implementation from local copy
        from conditional_unet_stage2 import ConditionalUNet
        self.ConditionalUNet = ConditionalUNet
        
    def load(self) -> nn.Module:
        """Load Stage 2.1 checkpoint with ConditionalUNet architecture"""
        print(f"Loading Stage 2.1 checkpoint from {self.checkpoint_path}")
        
        # Create model with Stage 2.1 configuration
        model = self.ConditionalUNet(
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
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        model.eval()
        self.model = model
        
        print(f"  [OK] Stage 2.1 model loaded successfully")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
        
    def denoise(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor,
                motion_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Denoise using Stage 2.1 forward signature"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Stage 2.1 doesn't support motion_tokens
        # text_emb should be [batch, seq_len, text_dim]
        with torch.no_grad():
            output = self.model(x, t, text_emb)
            
        return output


class Stage3Adapter(InferenceAdapter):
    """Adapter for Stage 3 VideoUNetFixed checkpoints (video generation)"""
    
    def __init__(self, checkpoint_path: str, config: Dict[str, Any]):
        super().__init__(checkpoint_path, config)
        # Import Stage 3 implementation from local copy
        from video_unet_feature_fusion import VideoUNetFixed
        from conditional_unet_stage2 import ConditionalUNet
        from text_encoder_split import TextEncoderSplit
        self.VideoUNetFixed = VideoUNetFixed
        self.ConditionalUNet = ConditionalUNet
        self.TextEncoder = TextEncoderSplit
        
    def load(self) -> nn.Module:
        """Load Stage 3 checkpoint with VideoUNetFixed architecture"""
        print(f"Loading Stage 3 checkpoint from {self.checkpoint_path}")
        
        # Create spatial UNet component
        spatial_unet = self.ConditionalUNet(
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
        
        # Create text encoder component (TextEncoderSplit)
        text_encoder = self.TextEncoder(
            original_vocab_size=self.config.get('vocab_size', 1000),
            motion_vocab_size=self.config.get('motion_vocab_size', 32),
            d_model=self.config.get('text_dim', 256),
            n_heads=self.config.get('n_heads', 4),
            n_layers=self.config.get('n_layers', 4),
            d_ff=self.config.get('d_ff', 512),
            max_seq_len=self.config.get('max_seq_len', 64),
            dropout=self.config.get('dropout', 0.1),
            freeze_transformer=True
        )
        
        # Create VideoUNetFixed model
        model = self.VideoUNetFixed(
            spatial_unet=spatial_unet,
            text_encoder=text_encoder,
            num_frames=self.config.get('num_frames', 16),
            image_size=self.config.get('image_size', 64),
            temporal_kernel=self.config.get('temporal_kernel', 3),
            temporal_layers=self.config.get('temporal_layers', 2),
            freeze_spatial=self.config.get('freeze_spatial', True),
            use_temporal=self.config.get('use_temporal', False)
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load state dict from nested structure
        # Use strict=False to handle temporal modules in B checkpoint with use_temporal=False
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
        
        print(f"  [OK] Stage 3 model loaded successfully")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  use_temporal: {self.config.get('use_temporal', False)}")
        
        return model
        
    def denoise(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor,
                motion_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Denoise using Stage 3 forward signature"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        with torch.no_grad():
            output = self.model(x, t, text_emb)
            
        return output


def create_adapter(checkpoint_path: str, experiment_type: str) -> InferenceAdapter:
    """
    Factory function to create appropriate adapter based on experiment type.
    
    Args:
        checkpoint_path: Path to checkpoint file
        experiment_type: 'A', 'B', or 'C'
        
    Returns:
        Appropriate adapter instance
    """
    base_dir = Path(__file__).parent.parent.parent
    
    if experiment_type == 'A':
        # Stage 2.1 checkpoint (ConditionalUNet)
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
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        return Stage2Adapter(checkpoint_path, config)
        
    elif experiment_type in ['B', 'C']:
        # Stage 3 checkpoint (VideoUNetFixed)
        use_temporal = (experiment_type == 'C')
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
            'vocab_size': 42,  # From checkpoint inspection
            'motion_vocab_size': 40,  # From checkpoint inspection
            'n_layers': 4,
            'd_ff': 512,
            'max_seq_len': 64,
            'num_frames': 16,
            'image_size': 64,
            'temporal_kernel': 3,
            'temporal_layers': 2,
            'freeze_spatial': True,
            'use_temporal': use_temporal,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        return Stage3Adapter(checkpoint_path, config)
        
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
