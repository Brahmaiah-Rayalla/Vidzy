"""
U-Net architecture for diffusion models.

Implements a compact convolutional U-Net with:
- Sinusoidal timestep embeddings
- Residual convolution blocks
- Group normalization
- Skip connections
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        """
        Args:
            dim: Dimension of the embedding
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for timesteps.

        Args:
            t: Tensor of timesteps of shape [batch_size]

        Returns:
            Embeddings of shape [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2

        # Create frequencies
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Compute embeddings
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb


class ResidualBlock(nn.Module):
    """Residual convolution block with group normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        timestep_emb_dim: int,
        dropout: float = 0.0,
        dilation: int = 1,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            timestep_emb_dim: Dimension of timestep embeddings
            dropout: Dropout probability
        """
        super().__init__()

        if dilation < 1:
            raise ValueError("ResidualBlock dilation must be positive")

        # Timestep embedding projection
        self.timestep_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_emb_dim, out_channels)
        )

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(8, out_channels)

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Skip connection projection if needed
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, timestep_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x: Input tensor of shape [batch, in_channels, height, width]
            timestep_emb: Timestep embedding of shape [batch, timestep_emb_dim]

        Returns:
            Output tensor of shape [batch, out_channels, height, width]
        """
        # Project timestep embedding
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add timestep embedding
        timestep_emb = self.timestep_proj(timestep_emb)[:, :, None, None]
        h = h + timestep_emb

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = F.silu(h)

        # Skip connection
        return h + self.skip_connection(x)


class Downsample(nn.Module):
    """Downsampling block using strided convolution."""

    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input/output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by factor of 2."""
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling block using nearest neighbor + convolution."""

    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input/output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by factor of 2."""
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """
    Compact U-Net for diffusion models.

    Architecture:
    - Encoder: downsampling path with residual blocks
    - Middle: processing at bottleneck resolution
    - Decoder: upsampling path with skip connections
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
        num_classes: int = 0,
        num_color_classes: int = 0,
        num_size_classes: int = 0,
        spatial_size_conditioning: bool = False,
        high_resolution_refinement: bool = False,
        size_embedding_scale: float = 1.0,
        middle_dilation: int = 1,
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            model_channels: Base number of channels in the model
            channel_mult: Channel multipliers for each resolution level
            num_res_blocks: Number of residual blocks per level
            attention_resolutions: Resolutions at which to use attention (not used in this version)
            dropout: Dropout probability
            num_timesteps: Number of diffusion timesteps for embedding dimension
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.num_classes = num_classes
        self.num_color_classes = num_color_classes
        self.num_size_classes = num_size_classes
        self.spatial_size_conditioning = spatial_size_conditioning
        self.high_resolution_refinement = high_resolution_refinement
        if middle_dilation < 1:
            raise ValueError("middle_dilation must be positive")
        self.middle_dilation = int(middle_dilation)
        if size_embedding_scale <= 0:
            raise ValueError("size_embedding_scale must be positive")
        self.size_embedding_scale = float(size_embedding_scale)

        # Timestep embedding
        self.timestep_embed = SinusoidalPositionEmbedding(model_channels)
        self.class_embed = nn.Embedding(num_classes, model_channels) if num_classes > 0 else None
        self.color_embed = nn.Embedding(num_color_classes, model_channels) if num_color_classes > 0 else None
        self.size_embed = nn.Embedding(num_size_classes, model_channels) if num_size_classes > 0 else None
        if self.size_embed is not None:
            nn.init.zeros_(self.size_embed.weight)

        # Initial convolution
        extra_size_channel = int(spatial_size_conditioning)
        self.conv_in = nn.Conv2d(in_channels + extra_size_channel, model_channels, 3, padding=1)
        if extra_size_channel:
            nn.init.zeros_(self.conv_in.weight[:, in_channels:])

        # Encoder
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        now_channels = model_channels

        for level, mult in enumerate(channel_mult):
            out_channels = model_channels * mult

            for _ in range(num_res_blocks):
                layers = [ResidualBlock(now_channels, out_channels, model_channels, dropout)]
                now_channels = out_channels
                self.down_blocks.append(nn.ModuleList(layers))
                channels.append(now_channels)

            # Downsample (except at the last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(now_channels)]))
                channels.append(now_channels)

        # Middle block
        self.middle_block = ResidualBlock(
            now_channels, now_channels, model_channels, dropout,
            dilation=self.middle_dilation,
        )

        # Decoder
        self.up_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult

            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        now_channels + channels.pop(),
                        out_channels,
                        model_channels,
                        dropout
                    )
                ]
                now_channels = out_channels
                self.up_blocks.append(nn.ModuleList(layers))

            # Upsample (except at the last level)
            if level != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(now_channels)]))

        # Output convolution
        self.high_res_refine = (
            ResidualBlock(now_channels, now_channels, model_channels, dropout)
            if high_resolution_refinement else None
        )
        if self.high_res_refine is not None:
            # Preserve the pretrained denoiser at initialization; training can
            # then learn a small full-resolution correction for sharp edges.
            nn.init.zeros_(self.high_res_refine.conv2.weight)
            nn.init.zeros_(self.high_res_refine.conv2.bias)
        self.conv_out = nn.Conv2d(now_channels, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        color_labels: Optional[torch.Tensor] = None,
        size_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the U-Net.

        Args:
            x: Noisy image tensor of shape [batch, in_channels, height, width]
            timesteps: Diffusion timesteps of shape [batch]

        Returns:
            Predicted noise of shape [batch, in_channels, height, width]
        """
        # Timestep embedding
        timestep_emb = self.timestep_embed(timesteps)
        if self.class_embed is not None:
            if class_labels is None:
                raise ValueError("class_labels are required for this class-conditional U-Net")
            timestep_emb = timestep_emb + self.class_embed(class_labels.long())
        elif class_labels is not None:
            raise ValueError("class_labels were provided to an unconditional U-Net")
        if self.color_embed is not None:
            if color_labels is None:
                raise ValueError("color_labels are required for this color-conditional U-Net")
            timestep_emb = timestep_emb + self.color_embed(color_labels.long())
        elif color_labels is not None:
            raise ValueError("color_labels were provided to a model without color conditioning")
        if self.size_embed is not None:
            if size_labels is None:
                raise ValueError("size_labels are required for this size-conditional U-Net")
            timestep_emb = timestep_emb + self.size_embedding_scale * self.size_embed(size_labels.long())
        elif size_labels is not None:
            raise ValueError("size_labels were provided to a model without size conditioning")

        # Initial convolution
        if self.spatial_size_conditioning:
            if size_labels is None:
                raise ValueError("size_labels are required for spatial size conditioning")
            size_plane = size_labels.to(dtype=x.dtype).view(-1, 1, 1, 1)
            size_plane = (2.0 * size_plane / max(self.num_size_classes - 1, 1) - 1.0)
            size_plane = size_plane.expand(-1, 1, x.shape[2], x.shape[3])
            x = torch.cat((x, size_plane), dim=1)
        h = self.conv_in(x)

        # Encoder with skip connections
        hs = [h]
        for modules in self.down_blocks:
            for module in modules:
                if isinstance(module, ResidualBlock):
                    h = module(h, timestep_emb)
                else:
                    h = module(h)
                hs.append(h)

        # Middle block
        h = self.middle_block(h, timestep_emb)

        # Decoder with skip connections
        for modules in self.up_blocks:
            for module in modules:
                if isinstance(module, ResidualBlock):
                    h = torch.cat([h, hs.pop()], dim=1)
                    h = module(h, timestep_emb)
                else:
                    h = module(h)

        # Optional inexpensive full-resolution detail refinement.
        if self.high_res_refine is not None:
            h = self.high_res_refine(h, timestep_emb)

        # Output
        return self.conv_out(h)
