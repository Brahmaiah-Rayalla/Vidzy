"""
Stage 3.2.9H - Corrected Inference Adapter

Fixes the text conditioning bug by using actual text encoding instead of random embeddings.

Author: Devin
Date: 2026-11-03
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import json
from text_encoder import SimpleTokenizer, TextEncoder


class CorrectedInferenceAdapter:
    """Adapter for Stage 3 inference with CORRECTED text conditioning."""

    def __init__(self, checkpoint_path: str, exp_type: str, tokenizer_path: str, text_encoder_path: str):
        self.checkpoint_path = checkpoint_path
        self.exp_type = exp_type
        self.tokenizer_path = tokenizer_path
        self.text_encoder_path = text_encoder_path
        self.model = None
        self.tokenizer = None
        self.text_encoder = None

    def load(self) -> nn.Module:
        """Load model with proper text conditioning."""
        print(f"\nLoading Stage 3 checkpoint from {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        # Load tokenizer
        print(f"Loading tokenizer from {self.tokenizer_path}")
        self.tokenizer = SimpleTokenizer.load(self.tokenizer_path)

        # Load text encoder
        print(f"Loading text encoder from {self.text_encoder_path}")
        te_checkpoint = torch.load(self.text_encoder_path, map_location='cpu')
        
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

        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent.parent
        stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
        sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))

        from models.conditional_unet import ConditionalUNet
        from models.video_unet_fixed import VideoUNetFixed
        from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder

        # Load spatial UNet
        spatial_unet = ConditionalUNet(
            in_channels=3,
            model_channels=64,
            channel_mult=[1, 2, 2, 2],
            num_res_blocks=2,
            attention_resolutions=[],
            dropout=0.0,
            num_timesteps=1000,
            text_dim=256,
            n_heads=4,
            use_cross_attn=True
        )

        # Load spatial parameters from checkpoint
        if 'spatial_unet_state_dict' in checkpoint:
            spatial_unet.load_state_dict(checkpoint['spatial_unet_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # Try loading from model_state_dict
            spatial_state = {k: v for k, v in checkpoint['model_state_dict'].items() 
                          if 'spatial' in k or 'unet' in k}
            if spatial_state:
                spatial_unet.load_state_dict(spatial_state, strict=False)

        # Create split encoder
        split_encoder = load_from_standard_encoder(
            self.text_encoder,
            motion_vocab_size=40,
            freeze_transformer=True
        )

        # Load split encoder state if available
        if 'text_encoder_state_dict' in checkpoint:
            # This should handle the split encoder state
            pass  # load_from_standard_encoder already initialized it

        # Determine temporal processing
        use_temporal = self.exp_type in ['C', 'D']
        enable_temporal = self.exp_type == 'C'

        # Create video model
        self.model = VideoUNetFixed(
            spatial_unet=spatial_unet,
            text_encoder=split_encoder,
            num_frames=16,
            image_size=64,
            temporal_kernel=3,
            temporal_layers=2,
            freeze_spatial=True,
            use_temporal=use_temporal,
            enable_temporal_processing=enable_temporal
        )

        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Load motion embeddings if available
        if 'motion_embeddings' in checkpoint:
            self.model.text_encoder.token_emb.motion_emb.weight.data = checkpoint['motion_embeddings']

        # Load fusion gates if available
        if 'fusion_gates' in checkpoint:
            for i, gate in enumerate(self.model.fusion_modules):
                gate.weight.data = torch.tensor([checkpoint['fusion_gates'][i]])

        self.model.eval()

        # Print parameter counts
        spatial_params = sum(p.numel() for p in self.model.spatial_unet.parameters())
        temporal_params = sum(p.numel() for p in self.model.temporal_modules.parameters())
        fusion_params = sum(p.numel() for p in self.model.fusion_modules.parameters())
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"Parameter counts:")
        print(f"  Spatial: {spatial_params:,}")
        print(f"  Temporal: {temporal_params:,}")
        print(f"  Fusion: {fusion_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"  use_temporal: {self.model.use_temporal}")
        print(f"  enable_temporal_processing: {self.model.enable_temporal_processing}")

        return self.model

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        Encode text prompt using ACTUAL text encoder (FIXED).
        
        This replaces the buggy random embedding implementation.
        """
        # Tokenize prompt
        token_ids, mask = self.tokenizer.encode(prompt, max_length=64)

        # Convert to tensors
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)

        # Encode with text encoder
        with torch.no_grad():
            text_emb, text_mask = self.text_encoder(token_ids_tensor, mask_tensor)

        return text_emb, text_mask


def create_corrected_adapter(checkpoint_path: str, exp_type: str) -> CorrectedInferenceAdapter:
    """Create a corrected inference adapter."""
    
    # Determine paths
    base_dir = Path(checkpoint_path).parent.parent.parent.parent
    tokenizer_path = str(base_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json")
    text_encoder_path = str(base_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_motion_expanded.pt")

    return CorrectedInferenceAdapter(
        checkpoint_path=checkpoint_path,
        exp_type=exp_type,
        tokenizer_path=tokenizer_path,
        text_encoder_path=text_encoder_path
    )
