"""
Conditional U-Net with text conditioning for diffusion models.

Extends the unconditional U-Net with cross-attention layers that incorporate
text embeddings from the text encoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from unet import (
    SinusoidalPositionEmbedding,
    ResidualBlock,
    Downsample,
    Upsample
)
from text_encoder import MultiHeadAttention
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "experiments_stage3_2_7" / "video_foundation"))


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for conditioning on text embeddings.

    Takes image features as query and text embeddings as key/value.
    """

    def __init__(
        self,
        channels: int,
        text_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0
    ):
        """
        Initialize cross-attention block.

        Args:
            channels: Number of channels in image features
            text_dim: Dimension of text embeddings
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.norm = nn.GroupNorm(8, channels)

        # Cross-attention: image features (query) attend to text (key, value)
        self.cross_attn = MultiHeadAttention(
            d_model=channels,
            n_heads=n_heads,
            dropout=dropout
        )

        # Project text embeddings to match image feature dimension
        self.text_proj = nn.Linear(text_dim, channels)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Image features [batch, channels, height, width]
            text_emb: Text embeddings [batch, seq_len, text_dim]
            text_mask: Text attention mask [batch, seq_len] (1 for real tokens, 0 for padding)

        Returns:
            Conditioned image features [batch, channels, height, width]
        """
        # Normalize
        h = self.norm(x)

        # Reshape for attention: [batch, channels, h, w] -> [batch, h*w, channels]
        batch_size, channels, height, width = h.shape
        spatial_size = height * width
        h = h.reshape(batch_size, channels, spatial_size).transpose(1, 2)
        # Shape: [batch, h*w, channels]

        # Project text embeddings
        text_proj = self.text_proj(text_emb)
        # Shape: [batch, seq_len, channels]

        # Cross-attention: image features (query) attend to text (key, value)
        attn_out = self.cross_attn(h, text_proj, text_proj, text_mask)
        # Shape: [batch, h*w, channels]

        # Reshape back: [batch, h*w, channels] -> [batch, channels, h, w]
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, channels, height, width)

        # Residual connection
        x = x + self.dropout(attn_out)

        return x


class ConditionalResidualBlock(nn.Module):
    """
    Residual block with both timestep and text conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        timestep_emb_dim: int,
        text_dim: int,
        use_cross_attn: bool = True,
        n_heads: int = 4,
        dropout: float = 0.0
    ):
        """
        Initialize conditional residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            timestep_emb_dim: Dimension of timestep embeddings
            text_dim: Dimension of text embeddings
            use_cross_attn: Whether to use cross-attention
            n_heads: Number of attention heads for cross-attention
            dropout: Dropout probability
        """
        super().__init__()

        # Timestep embedding projection
        self.timestep_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_emb_dim, out_channels)
        )

        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Skip connection projection if needed
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Cross-attention for text conditioning
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttentionBlock(
                channels=out_channels,
                text_dim=text_dim,
                n_heads=n_heads,
                dropout=dropout
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, in_channels, height, width]
            timestep_emb: Timestep embedding [batch, timestep_emb_dim]
            text_emb: Text embeddings [batch, seq_len, text_dim]
            text_mask: Text attention mask [batch, seq_len]

        Returns:
            Output tensor [batch, out_channels, height, width]
        """
        # Project timestep embedding
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add timestep embedding
        timestep_emb = self.timestep_proj(timestep_emb)[:, :, None, None]
        h = h + timestep_emb

        # Cross-attention for text conditioning
        if self.use_cross_attn:
            h = self.cross_attn(h, text_emb, text_mask)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = F.silu(h)

        # Skip connection
        return h + self.skip_connection(x)


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for text-conditioned diffusion models.

    Architecture:
    - Encoder: downsampling path with conditional residual blocks
    - Middle: processing at bottleneck resolution with cross-attention
    - Decoder: upsampling path with skip connections and cross-attention
    - Conditioning: Timestep embeddings + text embeddings via cross-attention
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 64,
        channel_mult: List[int] = [1, 2, 2, 2],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [],
        dropout: float = 0.0,
        num_timesteps: int = 1000,
        text_dim: int = 256,
        n_heads: int = 4,
        use_cross_attn: bool = True
    ):
        """
        Initialize conditional U-Net.

        Args:
            in_channels: Number of input channels (3 for RGB)
            model_channels: Base number of channels in the model
            channel_mult: Channel multipliers for each resolution level
            num_res_blocks: Number of residual blocks per level
            attention_resolutions: Resolutions at which to use self-attention (not used in this version)
            dropout: Dropout probability
            num_timesteps: Number of diffusion timesteps for embedding dimension
            text_dim: Dimension of text embeddings
            n_heads: Number of attention heads for cross-attention
            use_cross_attn: Whether to use cross-attention for text conditioning
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.text_dim = text_dim
        self.use_cross_attn = use_cross_attn

        # Timestep embedding
        self.timestep_embed = SinusoidalPositionEmbedding(model_channels)

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        now_channels = model_channels

        for level, mult in enumerate(channel_mult):
            out_channels = model_channels * mult

            for _ in range(num_res_blocks):
                layers = [ConditionalResidualBlock(
                    now_channels,
                    out_channels,
                    model_channels,
                    text_dim,
                    use_cross_attn=use_cross_attn,
                    n_heads=n_heads,
                    dropout=dropout
                )]
                now_channels = out_channels
                self.down_blocks.append(nn.ModuleList(layers))
                channels.append(now_channels)

            # Downsample (except at the last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(now_channels)]))
                channels.append(now_channels)

        # Middle block
        self.middle_block = ConditionalResidualBlock(
            now_channels,
            now_channels,
            model_channels,
            text_dim,
            use_cross_attn=use_cross_attn,
            n_heads=n_heads,
            dropout=dropout
        )

        # Decoder
        self.up_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult

            for i in range(num_res_blocks + 1):
                layers = [
                    ConditionalResidualBlock(
                        now_channels + channels.pop(),
                        out_channels,
                        model_channels,
                        text_dim,
                        use_cross_attn=use_cross_attn,
                        n_heads=n_heads,
                        dropout=dropout
                    )
                ]
                now_channels = out_channels
                self.up_blocks.append(nn.ModuleList(layers))

            # Upsample (except at the last level)
            if level != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(now_channels)]))

        # Output convolution
        self.conv_out = nn.Conv2d(now_channels, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the conditional U-Net.

        Args:
            x: Noisy image tensor of shape [batch, in_channels, height, width]
            timesteps: Diffusion timesteps of shape [batch]
            text_emb: Text embeddings of shape [batch, seq_len, text_dim]
            text_mask: Text attention mask of shape [batch, seq_len]
            return_features: If True, return intermediate features for temporal processing

        Returns:
            If return_features=False: Predicted noise of shape [batch, in_channels, height, width]
            If return_features=True: Tuple of (output, features_dict) where features_dict contains
                                       'bottleneck' and 'low_res' features
        """
        # Timestep embedding
        timestep_emb = self.timestep_embed(timesteps)

        # Initial convolution
        h = self.conv_in(x)

        # Encoder with skip connections
        hs = [h]
        for modules in self.down_blocks:
            for module in modules:
                if isinstance(module, ConditionalResidualBlock):
                    h = module(h, timestep_emb, text_emb, text_mask)
                else:
                    h = module(h)
                hs.append(h)

        # Middle block (bottleneck)
        h = self.middle_block(h, timestep_emb, text_emb, text_mask)
        bottleneck_features = h.clone() if return_features else None

        # Decoder with skip connections
        for modules in self.up_blocks:
            for module in modules:
                if isinstance(module, ConditionalResidualBlock):
                    h = torch.cat([h, hs.pop()], dim=1)
                    h = module(h, timestep_emb, text_emb, text_mask)
                else:
                    h = module(h)

        # Capture low-res features (after first upsample in decoder)
        low_res_features = h.clone() if return_features else None

        # Output
        output = self.conv_out(h)

        if return_features:
            features_dict = {
                'bottleneck': bottleneck_features,
                'low_res': low_res_features
            }
            return output, features_dict
        else:
            return output

    def load_unconditional_weights(
        self,
        state_dict: dict,
        strict: bool = False
    ) -> None:
        """
        Load weights from an unconditional U-Net checkpoint.

        Reuses compatible layers and initializes new layers randomly.

        Args:
            state_dict: State dict from unconditional U-Net
            strict: Whether to enforce strict loading (False allows partial loading)
        """
        # Map unconditional parameter names to conditional ones
        # Most parameters are compatible (conv layers, norm layers, etc.)
        # Only timestep projection and new cross-attention layers are different

        adapted_dict = {}
        adapted_keys = set()

        for key, value in state_dict.items():
            # Timestep projection: different in conditional version
            if 'timestep_proj' in key:
                # Skip timestep projection (conditional version has different architecture)
                continue

            # All other parameters should be compatible
            adapted_dict[key] = value
            adapted_keys.add(key)

        # Load adapted state dict
        missing, unexpected = self.load_state_dict(adapted_dict, strict=False)

        print(f"Loaded unconditional weights:")
        print(f"  Loaded {len(adapted_keys)} parameters")
        print(f"  Missing keys (newly initialized): {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

        if len(missing) > 0:
            print(f"  Sample missing keys: {missing[:5]}")
