"""
Stage 3.2.9H.2 - Corrected Inference Adapters with Actual Text Conditioning

Integrates the corrected text-to-embedding adapter into the main inference pipeline.
This replaces the buggy random embedding implementation with actual text encoding.

Author: Devin
Date: 2026-11-03
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import sys

# Add parent directories to path
base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "experiments_stage3_2_7" / "video_foundation"))
sys.path.insert(0, str(Path(__file__).parent))


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
    """Adapter for Stage 3 VideoUNetFixed checkpoints (video generation) with CORRECTED text conditioning"""
    
    def __init__(self, checkpoint_path: str, config: Dict[str, Any]):
        super().__init__(checkpoint_path, config)
        # Import Stage 3 implementation from local copy
        from video_unet_feature_fusion import VideoUNetFixed
        from conditional_unet_stage2 import ConditionalUNet
        from text_encoder_split import TextEncoderSplit
        self.VideoUNetFixed = VideoUNetFixed
        self.ConditionalUNet = ConditionalUNet
        self.TextEncoder = TextEncoderSplit
        
        # Text encoder components for actual encoding
        self.tokenizer = None
        self.text_encoder = None
        
    def load(self) -> nn.Module:
        """Load Stage 3 checkpoint with VideoUNetFixed architecture and actual text encoder"""
        print(f"Loading Stage 3 checkpoint from {self.checkpoint_path}")
        
        # Load tokenizer and text encoder for actual text encoding
        self._load_text_encoder_components()
        
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
        print(f"  [OK] Text encoder loaded for actual text encoding")
        
        return model
    
    def _load_text_encoder_components(self):
        """Load tokenizer and text encoder for actual text encoding"""
        from text_encoder import SimpleTokenizer, TextEncoder
        
        # Determine paths
        base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
        tokenizer_path = base_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json"
        text_encoder_path = base_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_motion_expanded.pt"
        
        print(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = SimpleTokenizer.load(str(tokenizer_path))
        
        print(f"Loading text encoder from {text_encoder_path}")
        te_checkpoint = torch.load(text_encoder_path, map_location='cpu')
        
        # Extract config from nested structure
        if 'config' in te_checkpoint and 'text_encoder' in te_checkpoint['config']:
            te_config = te_checkpoint['config']['text_encoder']
        else:
            te_config = te_checkpoint
        
        # Get actual vocab size from checkpoint state dict
        if 'text_encoder_state_dict' in te_checkpoint:
            actual_vocab_size = te_checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]
        elif 'model_state_dict' in te_checkpoint:
            actual_vocab_size = te_checkpoint['model_state_dict']['token_emb.weight'].shape[0]
        else:
            actual_vocab_size = te_config.get('vocab_size', 1000)
        
        print(f"  Actual vocab size from checkpoint: {actual_vocab_size}")
        
        # Create text encoder with config and actual vocab size
        self.text_encoder = TextEncoder(
            vocab_size=actual_vocab_size,
            d_model=te_config.get('d_model', 256),
            n_heads=te_config.get('n_heads', 4),
            n_layers=te_config.get('n_layers', 4),
            d_ff=te_config.get('d_ff', 512),
            max_seq_len=te_config.get('max_seq_len', 64),
            dropout=te_config.get('dropout', 0.0)
        )
        
        # Load state dict
        if 'text_encoder_state_dict' in te_checkpoint:
            self.text_encoder.load_state_dict(te_checkpoint['text_encoder_state_dict'])
        elif 'model_state_dict' in te_checkpoint:
            self.text_encoder.load_state_dict(te_checkpoint['model_state_dict'])
        
        self.text_encoder.eval()
        
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        Encode text prompt using ACTUAL text encoder (CORRECTED).
        
        This replaces the buggy random embedding implementation.
        """
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError("Text encoder components not loaded. Call load() first.")
        
        # Tokenize prompt
        token_ids, mask = self.tokenizer.encode(prompt, max_length=64)
        
        # Convert to tensors
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        
        # Encode with text encoder
        with torch.no_grad():
            text_emb, text_mask = self.text_encoder(token_ids_tensor, mask_tensor)
        
        return text_emb, text_mask
        
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
        experiment_type: 'A', 'B', 'C', or 'D'
        
    Returns:
        Appropriate adapter instance
    """
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
    
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
        
    elif experiment_type in ['B', 'C', 'D']:
        # Stage 3 checkpoint (VideoUNetFixed) with corrected text conditioning
        # D is same as C checkpoint but with temporal disabled
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
