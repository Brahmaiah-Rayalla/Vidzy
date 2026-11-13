"""
Video U-Net for text-conditioned video diffusion.

Extends the Stage 2 conditional U-Net with temporal processing modules.
Architecture:
- Spatial U-Net (from Stage 2) processes individual frames
- Temporal 1D convolution blocks process across frames
- Text conditioning via cross-attention (shared from Stage 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

from .conditional_unet import ConditionalUNet
from .text_encoder import TextEncoder


class TemporalConv1DBlock(nn.Module):
    """
    1D temporal convolution block for video generation (Stage 3.2.1).

    Processes the temporal dimension of spatial features.
    Uses 1D convolutions with residual connections.
    """

    def __init__(
        self,
        channels: int,
        temporal_kernel: int = 3,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        """
        Initialize temporal convolution block.

        Args:
            channels: Number of feature channels
            temporal_kernel: Kernel size for temporal convolution
            num_layers: Number of convolution layers
            dropout: Dropout probability
        """
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = channels if i == 0 else channels
            out_ch = channels
            self.layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=temporal_kernel, padding=temporal_kernel // 2)
            )
            self.layers.append(nn.GroupNorm(8, channels))
            self.layers.append(nn.SiLU())

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, frames, channels, height, width]

        Returns:
            Output tensor of same shape
        """
        # Save input for residual connection
        residual = x

        # Reshape: [batch, frames, channels, h, w] -> [batch * h * w, channels, frames]
        batch, frames, channels, height, width = x.shape
        x = x.permute(0, 3, 4, 1, 2)  # [batch, h, w, frames, channels]
        x = x.reshape(batch * height * width, channels, frames)  # [batch*h*w, channels, frames]

        # Apply temporal convolutions
        for layer in self.layers:
            x = layer(x)

        # Reshape back: [batch * h * w, channels, frames] -> [batch, frames, channels, h, w]
        x = x.reshape(batch, height, width, frames, channels)
        x = x.permute(0, 3, 4, 1, 2)  # [batch, frames, channels, h, w]

        # Residual connection
        x = x + residual
        x = self.dropout(x)

        return x


class LearnableTemporalFusion(nn.Module):
    """
    Learnable temporal fusion for Stage 3.2.1.

    Replaces the fixed scalar multiplier with a learnable gating mechanism.
    Allows the model to learn how much temporal information to incorporate.
    """

    def __init__(
        self,
        channels: int,
        initial_gate_value: float = 0.01,
        use_channel_wise: bool = False
    ):
        """
        Initialize learnable temporal fusion.

        Args:
            channels: Number of feature channels
            initial_gate_value: Initial value for the learnable gate
            use_channel_wise: If True, use per-channel gates; if False, use scalar gate
        """
        super().__init__()

        self.use_channel_wise = use_channel_wise

        if use_channel_wise:
            # Learnable gate per channel
            self.gate = nn.Parameter(torch.ones(channels) * initial_gate_value)
        else:
            # Learnable scalar gate
            self.gate = nn.Parameter(torch.tensor(initial_gate_value))

        # Gate activation to keep values in reasonable range
        self.gate_activation = nn.Sigmoid()

    def forward(self, temporal_features: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse temporal features with spatial features using learnable gate.

        Args:
            temporal_features: Temporal features [batch, frames, channels, h, w]
            spatial_features: Spatial features [batch, frames, channels, h, w]

        Returns:
            Fused features with same shape as inputs
        """
        # Compute learnable gate value
        gate = self.gate_activation(self.gate)

        # Reshape gate for broadcasting
        if self.use_channel_wise:
            # gate: [channels] -> [1, 1, channels, 1, 1]
            gate = gate.view(1, 1, -1, 1, 1)
        else:
            # gate: scalar -> [1, 1, 1, 1, 1]
            gate = gate.view(1, 1, 1, 1, 1)

        # Fuse: spatial + gate * temporal
        fused = spatial_features + gate * temporal_features

        return fused


class VideoUNet(nn.Module):
    """
    Video U-Net for text-conditioned video diffusion.

    Architecture (Stage 3.1):
    - Spatial backbone: Stage 2 conditional U-Net (processes each frame independently)
    - Temporal modules: 1D convolutions at bottleneck and low-resolution feature levels
    - Video tensor layout: [batch, channels, frames, height, width]

    The model predicts noise for the entire video sequence by:
    1. Processing each frame through spatial U-Net
    2. Capturing intermediate features at selected resolutions
    3. Applying temporal convolutions across frames at those feature levels
    4. Continuing spatial processing with temporally-refined features
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
        use_temporal: bool = True
    ):
        """
        Initialize video U-Net.

        Args:
            spatial_unet: Pre-trained Stage 2 conditional U-Net
            text_encoder: Pre-trained text encoder
            num_frames: Number of frames in video
            image_size: Size of each frame
            temporal_kernel: Kernel size for temporal convolutions
            temporal_layers: Number of temporal convolution layers per block
            freeze_spatial: Whether to freeze spatial backbone weights
            use_temporal: Whether to use temporal modules (True for Stage 3.1)
        """
        super().__init__()

        self.spatial_unet = spatial_unet
        self.text_encoder = text_encoder
        self.num_frames = num_frames
        self.image_size = image_size
        self.text_dim = text_encoder.d_model
        self.use_temporal = use_temporal

        # Freeze spatial backbone if requested
        if freeze_spatial:
            for param in self.spatial_unet.parameters():
                param.requires_grad = False
            print("Spatial backbone frozen")

        # Determine temporal insertion points based on spatial U-Net channel multipliers
        # Channel multipliers: [1, 2, 2, 2] -> channels: [64, 128, 128, 128]
        model_channels = spatial_unet.model_channels
        channel_mult = spatial_unet.channel_mult

        # Create temporal modules at selected resolutions
        self.temporal_modules = nn.ModuleDict()

        if use_temporal:
            # Bottleneck level (highest resolution features)
            bottleneck_channels = model_channels * channel_mult[-1]  # 128
            self.temporal_modules['bottleneck'] = TemporalConv1DBlock(
                channels=bottleneck_channels,
                temporal_kernel=temporal_kernel,
                num_layers=temporal_layers,
                dropout=0.0
            )

            # Low-resolution level (base model channels at lower resolution)
            # This is after the first upsample in the decoder
            low_res_channels = model_channels  # 64
            self.temporal_modules['low_res'] = TemporalConv1DBlock(
                channels=low_res_channels,
                temporal_kernel=temporal_kernel,
                num_layers=temporal_layers,
                dropout=0.0
            )

            # Learnable temporal fusion modules (Stage 3.2.1)
            self.bottleneck_fusion = LearnableTemporalFusion(
                channels=bottleneck_channels,
                initial_gate_value=0.01,
                use_channel_wise=False
            )
            self.low_res_fusion = LearnableTemporalFusion(
                channels=low_res_channels,
                initial_gate_value=0.01,
                use_channel_wise=False
            )

            # Projection layers to map temporal features to output channels
            self.bottleneck_proj = nn.Conv2d(bottleneck_channels, 3, 1)
            self.low_res_proj = nn.Conv2d(low_res_channels, 3, 1)

            print(f"Temporal modules created at channels: [bottleneck={bottleneck_channels}, low_res={low_res_channels}]")
            print(f"Learnable temporal fusion modules added")

        # Track parameter counts
        self.spatial_params = sum(p.numel() for p in self.spatial_unet.parameters())
        self.temporal_params = sum(p.numel() for p in self.temporal_modules.parameters()) if use_temporal else 0
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Parameter counts:")
        print(f"  Spatial: {self.spatial_params:,}")
        print(f"  Temporal: {self.temporal_params:,}")
        print(f"  Trainable: {self.trainable_params:,}")

    def encode_text(self, prompt: str, max_length: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompt to embeddings.

        Args:
            prompt: Text prompt
            max_length: Maximum sequence length

        Returns:
            Tuple of (text_emb, text_mask)
        """
        # Tokenize
        from .text_encoder import SimpleTokenizer
        # Note: This assumes tokenizer is saved with text encoder
        # In practice, tokenizer should be loaded separately
        # For now, we'll use a placeholder
        raise NotImplementedError("Text encoding requires loaded tokenizer")

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of video U-Net.

        Args:
            x: Noisy video tensor of shape [batch, channels, frames, height, width]
            timesteps: Diffusion timesteps of shape [batch]
            text_emb: Text embeddings of shape [batch, seq_len, text_dim]
            text_mask: Text attention mask of shape [batch, seq_len]

        Returns:
            Predicted noise of shape [batch, channels, frames, height, width]
        """
        batch, channels, num_frames, height, width = x.shape

        if not self.use_temporal:
            # Frame-independent baseline (no temporal processing)
            return self._forward_frame_independent(x, timesteps, text_emb, text_mask)
        else:
            # Temporal processing with frame-by-frame spatial + temporal modules
            return self._forward_with_temporal(x, timesteps, text_emb, text_mask)

    def _forward_frame_independent(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Frame-independent forward pass (baseline without temporal modules)."""
        batch, channels, num_frames, height, width = x.shape

        # Reshape video for spatial processing
        x_spatial = x.permute(0, 2, 1, 3, 4)  # [batch, frames, channels, h, w]
        x_spatial = x_spatial.reshape(batch * num_frames, channels, height, width)

        # Broadcast timesteps across frames
        timesteps_spatial = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch * num_frames)

        # Broadcast text embeddings across frames
        text_emb_spatial = text_emb.unsqueeze(1).expand(-1, num_frames, -1, -1).reshape(
            batch * num_frames, text_emb.shape[1], text_emb.shape[2]
        )
        if text_mask is not None:
            text_mask_spatial = text_mask.unsqueeze(1).expand(-1, num_frames, -1).reshape(
                batch * num_frames, text_mask.shape[1]
            )
        else:
            text_mask_spatial = None

        # Spatial processing: apply U-Net to each frame independently
        predicted_noise_spatial = self.spatial_unet(
            x_spatial,
            timesteps_spatial,
            text_emb_spatial,
            text_mask_spatial
        )

        # Reshape back to video format
        predicted_noise = predicted_noise_spatial.reshape(batch, num_frames, channels, height, width)
        predicted_noise = predicted_noise.permute(0, 2, 1, 3, 4)  # [batch, channels, frames, h, w]

        return predicted_noise

    def _forward_with_temporal(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with temporal processing at intermediate feature levels."""
        batch, channels, num_frames, height, width = x.shape

        # Reshape video for spatial processing
        x_spatial = x.permute(0, 2, 1, 3, 4)  # [batch, frames, channels, h, w]
        x_spatial = x_spatial.reshape(batch * num_frames, channels, height, width)

        # Broadcast timesteps across frames
        timesteps_spatial = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch * num_frames)

        # Broadcast text embeddings across frames
        text_emb_spatial = text_emb.unsqueeze(1).expand(-1, num_frames, -1, -1).reshape(
            batch * num_frames, text_emb.shape[1], text_emb.shape[2]
        )
        if text_mask is not None:
            text_mask_spatial = text_mask.unsqueeze(1).expand(-1, num_frames, -1).reshape(
                batch * num_frames, text_mask.shape[1]
            )
        else:
            text_mask_spatial = None

        # Spatial processing with feature extraction
        predicted_noise_spatial, features = self.spatial_unet(
            x_spatial,
            timesteps_spatial,
            text_emb_spatial,
            text_mask_spatial,
            return_features=True
        )

        # Reshape features to video format for temporal processing
        # features['bottleneck']: [batch*frames, channels, h, w]
        # features['low_res']: [batch*frames, channels, h, w]

        # Process bottleneck features
        bottleneck_features = features['bottleneck']
        b_ch, b_h, b_w = bottleneck_features.shape[1], bottleneck_features.shape[2], bottleneck_features.shape[3]
        bottleneck_video = bottleneck_features.reshape(batch, num_frames, b_ch, b_h, b_w)
        # Temporal module expects [batch, frames, channels, h, w]
        bottleneck_video = bottleneck_video.permute(0, 1, 2, 3, 4)  # [batch, frames, channels, h, w]

        # Apply temporal module to bottleneck
        bottleneck_temporal = self.temporal_modules['bottleneck'](bottleneck_video)

        # Process low-res features
        low_res_features = features['low_res']
        lr_ch, lr_h, lr_w = low_res_features.shape[1], low_res_features.shape[2], low_res_features.shape[3]
        low_res_video = low_res_features.reshape(batch, num_frames, lr_ch, lr_h, lr_w)
        # Temporal module expects [batch, frames, channels, h, w]
        low_res_video = low_res_video.permute(0, 1, 2, 3, 4)  # [batch, frames, channels, h, w]

        # Apply temporal module to low-res
        low_res_temporal = self.temporal_modules['low_res'](low_res_video)

        # For Stage 3.2.1, use learnable temporal fusion at feature level
        # Compute temporal residuals and add to spatial features

        # Reshape temporal features back to spatial format
        bottleneck_temporal_flat = bottleneck_temporal.permute(0, 2, 1, 3, 4).reshape(batch * num_frames, b_ch, b_h, b_w)
        low_res_temporal_flat = low_res_temporal.permute(0, 2, 1, 3, 4).reshape(batch * num_frames, lr_ch, lr_h, lr_w)

        # Compute learnable fusion gates
        bottleneck_gate = self.bottleneck_fusion.gate_activation(self.bottleneck_fusion.gate)
        low_res_gate = self.low_res_fusion.gate_activation(self.low_res_fusion.gate)

        # Apply temporal residuals to features at their respective resolutions
        # This modifies the spatial features with temporal information
        # We need to inject this back into the spatial U-Net path
        # For now, we'll project to noise space and add to prediction

        # Project temporal features to noise predictions
        bottleneck_noise = self.bottleneck_proj(bottleneck_temporal_flat)
        low_res_noise = self.low_res_proj(low_res_temporal_flat)

        # Upsample bottleneck noise to full resolution
        bottleneck_noise_upsampled = F.interpolate(
            bottleneck_noise,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )

        # Upsample low-res noise to full resolution
        low_res_noise_upsampled = F.interpolate(
            low_res_noise,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )

        # Combine temporal noise predictions with learnable gates
        temporal_noise = bottleneck_gate * bottleneck_noise_upsampled + low_res_gate * low_res_noise_upsampled

        # Add temporal contribution to spatial prediction
        predicted_noise_spatial = predicted_noise_spatial + temporal_noise

        # Reshape back to video format for output
        predicted_noise = predicted_noise_spatial.reshape(batch, num_frames, channels, height, width)
        predicted_noise = predicted_noise.permute(0, 2, 1, 3, 4)  # [batch, channels, frames, h, w]

        return predicted_noise

    def get_trainable_parameters(self):
        """Get trainable parameters (useful for optimizer)."""
        return filter(lambda p: p.requires_grad, self.parameters())

    def load_spatial_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load spatial U-Net checkpoint.

        Args:
            checkpoint_path: Path to Stage 2 checkpoint
            strict: Whether to enforce strict loading
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load spatial U-Net weights
        # Checkpoint structure depends on how Stage 2 saved weights
        # Assuming checkpoint contains 'model_state_dict' or similar
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'unet_state_dict' in checkpoint:
            state_dict = checkpoint['unet_state_dict']
        else:
            state_dict = checkpoint

        # Load into spatial U-Net
        missing, unexpected = self.spatial_unet.load_state_dict(state_dict, strict=strict)

        print(f"Loaded spatial checkpoint from {checkpoint_path}")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

        if len(missing) > 0:
            print(f"  Sample missing keys: {missing[:5]}")

    def load_text_encoder_checkpoint(self, checkpoint_path: str, tokenizer_path: str):
        """
        Load text encoder checkpoint.

        Args:
            checkpoint_path: Path to text encoder checkpoint
            tokenizer_path: Path to tokenizer JSON
        """
        from .text_encoder import TextEncoder

        self.text_encoder, self.tokenizer = TextEncoder.load(checkpoint_path, tokenizer_path)

        print(f"Loaded text encoder from {checkpoint_path}")
        print(f"Loaded tokenizer from {tokenizer_path}")


def create_video_unet_from_stage2(
    spatial_checkpoint_path: str,
    text_encoder_checkpoint_path: str,
    tokenizer_path: str,
    num_frames: int = 16,
    image_size: int = 64,
    freeze_spatial: bool = False
) -> VideoUNet:
    """
    Create a VideoUNet from Stage 2 checkpoints.

    Args:
        spatial_checkpoint_path: Path to Stage 2 conditional U-Net checkpoint
        text_encoder_checkpoint_path: Path to text encoder checkpoint
        tokenizer_path: Path to tokenizer JSON
        num_frames: Number of frames
        image_size: Image size
        freeze_spatial: Whether to freeze spatial backbone

    Returns:
        Initialized VideoUNet
    """
    # Load spatial U-Net
    checkpoint = torch.load(spatial_checkpoint_path, map_location='cpu')

    # Reconstruct spatial U-Net
    # This requires knowing the architecture used in Stage 2
    # Based on Stage 2.1, the architecture is:
    # - in_channels: 3
    # - model_channels: 64
    # - channel_mult: [1, 2, 2, 2]
    # - num_res_blocks: 2
    # - text_dim: 256
    # - n_heads: 4

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

    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'unet_state_dict' in checkpoint:
        state_dict = checkpoint['unet_state_dict']
    else:
        state_dict = checkpoint

    spatial_unet.load_state_dict(state_dict, strict=False)
    print("Loaded spatial U-Net weights")

    # Load text encoder
    from .text_encoder import TextEncoder, SimpleTokenizer

    text_encoder, tokenizer = TextEncoder.load(text_encoder_checkpoint_path, tokenizer_path)
    print("Loaded text encoder and tokenizer")

    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=freeze_spatial
    )

    # Store tokenizer for text encoding
    video_unet.tokenizer = tokenizer

    return video_unet


if __name__ == "__main__":
    # Test the video U-Net structure
    print("Testing VideoUNet architecture...")

    # Create dummy spatial U-Net
    spatial_unet = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        text_dim=256,
        n_heads=4
    )

    # Create dummy text encoder
    text_encoder = TextEncoder(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512
    )

    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True
    )

    print(f"\nVideo U-Net created successfully")
    print(f"Total parameters: {sum(p.numel() for p in video_unet.parameters()):,}")
    print(f"Trainable parameters: {video_unet.trainable_params:,}")
