"""
Stage 3.2.7: Fixed Video U-Net with Feature-Level Temporal Fusion

Key fixes:
1. Temporal processing integrated inside spatial U-Net decoder path
2. Feature-level fusion instead of noise-level fusion
3. Identity-preserving initialization
4. Proper tensor transformations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

from .conditional_unet import ConditionalUNet
from .text_encoder import TextEncoder


class TemporalConv1DBlock(nn.Module):
    """1D temporal convolution block for video generation."""

    def __init__(self, channels: int, temporal_kernel: int = 3, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = channels if i == 0 else channels
            out_ch = channels
            self.layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=temporal_kernel, padding=temporal_kernel // 2))
            self.layers.append(nn.GroupNorm(8, channels))
            self.layers.append(nn.SiLU())
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, frames, channels, h, w]
        Returns:
            [batch, frames, channels, h, w]
        """
        residual = x
        batch, frames, channels, height, width = x.shape
        x = x.permute(0, 3, 4, 1, 2)  # [batch, h, w, frames, channels]
        x = x.reshape(batch * height * width, channels, frames)  # [batch*h*w, channels, frames]
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(batch, height, width, frames, channels)
        x = x.permute(0, 3, 4, 1, 2)  # [batch, frames, channels, h, w]
        x = x + residual
        x = self.dropout(x)
        return x


class FeatureLevelTemporalFusion(nn.Module):
    """
    Feature-level temporal fusion for Stage 3.2.7.

    Fuses temporal features with spatial features at the feature level,
    NOT at the noise level.
    """

    def __init__(self, channels: int, initial_gate_value: float = 0.1):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(initial_gate_value))
        self.gate_activation = nn.Sigmoid()

    def forward(self, temporal_features: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: [batch, frames, channels, h, w]
            spatial_features: [batch, frames, channels, h, w]
        Returns:
            Fused features [batch, frames, channels, h, w]
        """
        gate = self.gate_activation(self.gate)
        gate = gate.view(1, 1, 1, 1, 1)  # Broadcast to [1,1,1,1,1]
        fused = spatial_features + gate * temporal_features
        return fused


class VideoUNetFixed(nn.Module):
    """
    Fixed Video U-Net for Stage 3.2.7.

    Architecture:
    - Spatial encoder processes each frame
    - Temporal convolutions at bottleneck and low-res levels
    - Feature-level fusion before spatial decoder
    - Spatial decoder completes processing
    """

    def __init__(
        self,
        spatial_unet: ConditionalUNet,
        text_encoder: TextEncoder,
        num_frames: int = 16,
        image_size: int = 64,
        temporal_kernel: int = 3,
        temporal_layers: int = 2,
        freeze_spatial: bool = True,
        use_temporal: bool = True,
        enable_temporal_processing: bool = True
    ):
        super().__init__()
        self.spatial_unet = spatial_unet
        self.text_encoder = text_encoder
        self.num_frames = num_frames
        self.image_size = image_size
        self.use_temporal = use_temporal
        self.enable_temporal_processing = enable_temporal_processing

        if freeze_spatial:
            for param in self.spatial_unet.parameters():
                param.requires_grad = False

        # Create temporal modules
        self.temporal_modules = nn.ModuleDict()
        self.fusion_modules = nn.ModuleDict()

        if use_temporal:
            model_channels = spatial_unet.model_channels
            channel_mult = spatial_unet.channel_mult

            # Bottleneck level
            bottleneck_channels = model_channels * channel_mult[-1]  # 128
            self.temporal_modules['bottleneck'] = TemporalConv1DBlock(
                channels=bottleneck_channels,
                temporal_kernel=temporal_kernel,
                num_layers=temporal_layers,
                dropout=0.0
            )
            self.fusion_modules['bottleneck'] = FeatureLevelTemporalFusion(
                channels=bottleneck_channels,
                initial_gate_value=0.1
            )

            # Low-res level
            low_res_channels = model_channels  # 64
            self.temporal_modules['low_res'] = TemporalConv1DBlock(
                channels=low_res_channels,
                temporal_kernel=temporal_kernel,
                num_layers=temporal_layers,
                dropout=0.0
            )
            self.fusion_modules['low_res'] = FeatureLevelTemporalFusion(
                channels=low_res_channels,
                initial_gate_value=0.1
            )

            print(f"Fixed temporal modules created at channels: [bottleneck={bottleneck_channels}, low_res={low_res_channels}]")
            print(f"Feature-level fusion modules added")

        self.spatial_params = sum(p.numel() for p in self.spatial_unet.parameters())
        self.temporal_params = sum(p.numel() for p in self.temporal_modules.parameters()) if use_temporal else 0
        self.fusion_params = sum(p.numel() for p in self.fusion_modules.parameters()) if use_temporal else 0

        print(f"Parameter counts:")
        print(f"  Spatial: {self.spatial_params:,}")
        print(f"  Temporal: {self.temporal_params:,}")
        print(f"  Fusion: {self.fusion_params:,}")

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with feature-level temporal fusion.

        Args:
            x: [batch, channels, frames, height, width]
            timesteps: [batch]
            text_emb: [batch, seq_len, text_dim]
            text_mask: [batch, seq_len]
        Returns:
            Predicted noise [batch, channels, frames, height, width]
        """
        batch, channels, num_frames, height, width = x.shape

        if not self.use_temporal or not self.enable_temporal_processing:
            return self._forward_frame_independent(x, timesteps, text_emb, text_mask)
        else:
            return self._forward_with_temporal(x, timesteps, text_emb, text_mask)

    def _forward_frame_independent(self, x, timesteps, text_emb, text_mask):
        """Frame-independent baseline."""
        batch, channels, num_frames, height, width = x.shape
        x_spatial = x.permute(0, 2, 1, 3, 4).reshape(batch * num_frames, channels, height, width)
        timesteps_spatial = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch * num_frames)
        text_emb_spatial = text_emb.unsqueeze(1).expand(-1, num_frames, -1, -1).reshape(batch * num_frames, text_emb.shape[1], text_emb.shape[2])
        if text_mask is not None:
            text_mask_spatial = text_mask.unsqueeze(1).expand(-1, num_frames, -1).reshape(batch * num_frames, text_mask.shape[1])
        else:
            text_mask_spatial = None

        predicted_noise = self.spatial_unet(x_spatial, timesteps_spatial, text_emb_spatial, text_mask_spatial)
        predicted_noise = predicted_noise.reshape(batch, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return predicted_noise

    def _forward_with_temporal(self, x, timesteps, text_emb, text_mask):
        """
        Forward pass with feature-level temporal fusion.

        CRITICAL FIX: Temporal features are fused at feature level and
        fed back into spatial processing, NOT projected to noise space.
        """
        batch, channels, num_frames, height, width = x.shape

        # Reshape for spatial processing
        x_spatial = x.permute(0, 2, 1, 3, 4).reshape(batch * num_frames, channels, height, width)
        timesteps_spatial = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch * num_frames)
        text_emb_spatial = text_emb.unsqueeze(1).expand(-1, num_frames, -1, -1).reshape(batch * num_frames, text_emb.shape[1], text_emb.shape[2])
        if text_mask is not None:
            text_mask_spatial = text_mask.unsqueeze(1).expand(-1, num_frames, -1).reshape(batch * num_frames, text_mask.shape[1])
        else:
            text_mask_spatial = None

        # Get spatial features and initial noise prediction
        predicted_noise_spatial, spatial_features = self.spatial_unet(
            x_spatial,
            timesteps_spatial,
            text_emb_spatial,
            text_mask_spatial,
            return_features=True
        )

        # Extract bottleneck and low-res features
        bottleneck_features = spatial_features['bottleneck']  # [B*T, channels, h_bottleneck, w_bottleneck]
        low_res_features = spatial_features['low_res']  # [B*T, channels, h_low, w_low]

        # Reshape to restore temporal dimension
        # bottleneck: [B*T, C, h, w] -> [B, T, C, h, w]
        bottleneck_temporal = bottleneck_features.reshape(batch, num_frames, -1, bottleneck_features.shape[2], bottleneck_features.shape[3])
        low_res_temporal = low_res_features.reshape(batch, num_frames, -1, low_res_features.shape[2], low_res_features.shape[3])

        # Apply temporal convolutions
        # TemporalConv1DBlock expects [B, T, C, H, W]
        bottleneck_temporal_processed = self.temporal_modules['bottleneck'](bottleneck_temporal)
        low_res_temporal_processed = self.temporal_modules['low_res'](low_res_temporal)

        # Compute temporal residuals (difference between processed and original)
        bottleneck_residual = bottleneck_temporal_processed - bottleneck_temporal
        low_res_residual = low_res_temporal_processed - low_res_temporal

        # Apply fusion gates
        bottleneck_gate = self.fusion_modules['bottleneck'].gate_activation(self.fusion_modules['bottleneck'].gate)
        low_res_gate = self.fusion_modules['low_res'].gate_activation(self.fusion_modules['low_res'].gate)

        # Fused features
        bottleneck_fused = bottleneck_temporal + bottleneck_gate * bottleneck_residual
        low_res_fused = low_res_temporal + low_res_gate * low_res_residual

        # Reshape back to frame-independent format for injection
        bottleneck_injected = bottleneck_fused.reshape(batch * num_frames, -1, bottleneck_features.shape[2], bottleneck_features.shape[3])
        low_res_injected = low_res_fused.reshape(batch * num_frames, -1, low_res_features.shape[2], low_res_features.shape[3])

        # Create injected features dict
        injected_features = {
            'bottleneck': bottleneck_injected,
            'low_res': low_res_injected
        }

        # Run spatial decoder with injected features
        predicted_noise_final = self.spatial_unet(
            x_spatial,
            timesteps_spatial,
            text_emb_spatial,
            text_mask_spatial,
            injected_features=injected_features
        )

        # Reshape to video format
        predicted_noise_final = predicted_noise_final.reshape(batch, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)

        return predicted_noise_final

    def get_gate_values(self):
        """Get current gate values for monitoring."""
        if not self.use_temporal:
            return {}
        return {
            'bottleneck_gate': self.fusion_modules['bottleneck'].gate_activation(self.fusion_modules['bottleneck'].gate).item(),
            'low_res_gate': self.fusion_modules['low_res'].gate_activation(self.fusion_modules['low_res'].gate).item()
        }
