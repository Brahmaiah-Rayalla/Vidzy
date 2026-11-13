"""
Diagnose temporal feature contribution for Stage 3.2.2.

This script instruments both temporal fusion blocks to measure:
- Spatial feature RMS
- Temporal residual RMS
- Fused feature RMS
- Temporal-to-spatial contribution ratio
- Gate values
- Gate gradients
- Temporal parameter gradient norms
- Temporal parameter update magnitudes
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess
from video_foundation.training.train_video import SyntheticVideoDataset


class InstrumentedVideoUNet(VideoUNet):
    """VideoUNet with instrumentation hooks for temporal feature analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks = {
            'bottleneck_spatial_rms': [],
            'bottleneck_temporal_rms': [],
            'bottleneck_residual_rms': [],
            'bottleneck_fused_rms': [],
            'bottleneck_gate': [],
            'bottleneck_temporal_grad_norm': [],
            'bottleneck_fusion_grad_norm': [],
            'low_res_spatial_rms': [],
            'low_res_temporal_rms': [],
            'low_res_residual_rms': [],
            'low_res_fused_rms': [],
            'low_res_gate': [],
            'low_res_temporal_grad_norm': [],
            'low_res_fusion_grad_norm': [],
        }

    def _forward_with_temporal(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass with instrumentation."""
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
        bottleneck_features = features['bottleneck']
        b_ch, b_h, b_w = bottleneck_features.shape[1], bottleneck_features.shape[2], bottleneck_features.shape[3]
        bottleneck_video = bottleneck_features.reshape(batch, num_frames, b_ch, b_h, b_w)
        bottleneck_video = bottleneck_video.permute(0, 1, 2, 3, 4)  # [batch, frames, channels, h, w]

        # Record spatial RMS
        self.hooks['bottleneck_spatial_rms'].append(bottleneck_video.norm(p=2).item())

        # Apply temporal module to bottleneck
        bottleneck_temporal = self.temporal_modules['bottleneck'](bottleneck_video)

        # Record temporal RMS
        self.hooks['bottleneck_temporal_rms'].append(bottleneck_temporal.norm(p=2).item())

        # Record gate value
        bottleneck_gate = self.bottleneck_fusion.gate_activation(self.bottleneck_fusion.gate).item()
        self.hooks['bottleneck_gate'].append(bottleneck_gate)

        # Compute temporal residual (temporal - spatial)
        bottleneck_residual = bottleneck_temporal - bottleneck_video
        self.hooks['bottleneck_residual_rms'].append(bottleneck_residual.norm(p=2).item())

        # Process low-res features
        low_res_features = features['low_res']
        lr_ch, lr_h, lr_w = low_res_features.shape[1], low_res_features.shape[2], low_res_features.shape[3]
        low_res_video = low_res_features.reshape(batch, num_frames, lr_ch, lr_h, lr_w)
        low_res_video = low_res_video.permute(0, 1, 2, 3, 4)  # [batch, frames, channels, h, w]

        # Record spatial RMS
        self.hooks['low_res_spatial_rms'].append(low_res_video.norm(p=2).item())

        # Apply temporal module to low-res
        low_res_temporal = self.temporal_modules['low_res'](low_res_video)

        # Record temporal RMS
        self.hooks['low_res_temporal_rms'].append(low_res_temporal.norm(p=2).item())

        # Record gate value
        low_res_gate = self.low_res_fusion.gate_activation(self.low_res_fusion.gate).item()
        self.hooks['low_res_gate'].append(low_res_gate)

        # Compute temporal residual (temporal - spatial)
        low_res_residual = low_res_temporal - low_res_video
        self.hooks['low_res_residual_rms'].append(low_res_residual.norm(p=2).item())

        # Reshape temporal features back to spatial format
        bottleneck_temporal_flat = bottleneck_temporal.permute(0, 2, 1, 3, 4).reshape(batch * num_frames, b_ch, b_h, b_w)
        low_res_temporal_flat = low_res_temporal.permute(0, 2, 1, 3, 4).reshape(batch * num_frames, lr_ch, lr_h, lr_w)

        # Compute learnable fusion gates
        bottleneck_gate = self.bottleneck_fusion.gate_activation(self.bottleneck_fusion.gate)
        low_res_gate = self.low_res_fusion.gate_activation(self.low_res_fusion.gate)

        # Project temporal features to noise predictions
        bottleneck_noise = self.bottleneck_proj(bottleneck_temporal_flat)
        low_res_noise = self.low_res_proj(low_res_temporal_flat)

        # Upsample bottleneck noise to full resolution
        bottleneck_noise_upsampled = nn.functional.interpolate(
            bottleneck_noise,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )

        # Upsample low-res noise to full resolution
        low_res_noise_upsampled = nn.functional.interpolate(
            low_res_noise,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )

        # Combine temporal noise predictions with learnable gates
        temporal_noise = bottleneck_gate * bottleneck_noise_upsampled + low_res_gate * low_res_noise_upsampled

        # Record fused RMS
        self.hooks['bottleneck_fused_rms'].append(bottleneck_noise_upsampled.norm(p=2).item())
        self.hooks['low_res_fused_rms'].append(low_res_noise_upsampled.norm(p=2).item())

        # Add temporal contribution to spatial prediction
        predicted_noise_spatial = predicted_noise_spatial + temporal_noise

        # Reshape back to video format for output
        predicted_noise = predicted_noise_spatial.reshape(batch, num_frames, channels, height, width)
        predicted_noise = predicted_noise.permute(0, 2, 1, 3, 4)  # [batch, channels, frames, h, w]

        return predicted_noise

    def record_gradients(self):
        """Record gradient norms after backward pass."""
        # Bottleneck temporal gradients
        bottleneck_temporal_grads = []
        for name, param in self.temporal_modules['bottleneck'].named_parameters():
            if param.grad is not None:
                bottleneck_temporal_grads.append(param.grad.norm().item())
        if bottleneck_temporal_grads:
            self.hooks['bottleneck_temporal_grad_norm'].append(np.mean(bottleneck_temporal_grads))

        # Bottleneck fusion gradients
        bottleneck_fusion_grads = []
        for name, param in self.bottleneck_fusion.named_parameters():
            if param.grad is not None:
                bottleneck_fusion_grads.append(param.grad.norm().item())
        if bottleneck_fusion_grads:
            self.hooks['bottleneck_fusion_grad_norm'].append(np.mean(bottleneck_fusion_grads))

        # Low-res temporal gradients
        low_res_temporal_grads = []
        for name, param in self.temporal_modules['low_res'].named_parameters():
            if param.grad is not None:
                low_res_temporal_grads.append(param.grad.norm().item())
        if low_res_temporal_grads:
            self.hooks['low_res_temporal_grad_norm'].append(np.mean(low_res_temporal_grads))

        # Low-res fusion gradients
        low_res_fusion_grads = []
        for name, param in self.low_res_fusion.named_parameters():
            if param.grad is not None:
                low_res_fusion_grads.append(param.grad.norm().item())
        if low_res_fusion_grads:
            self.hooks['low_res_fusion_grad_norm'].append(np.mean(low_res_fusion_grads))

    def get_statistics(self):
        """Compute statistics from recorded hooks."""
        stats = {}
        for key, values in self.hooks.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        return stats


def diagnose_temporal_contribution(
    data_dir,
    checkpoint_path,
    num_samples=10,
    device='cpu'
):
    """Diagnose temporal feature contribution."""
    print("=" * 80)
    print("TEMPORAL FEATURE CONTRIBUTION DIAGNOSIS")
    print("=" * 80)

    # Load dataset
    print("\nLoading dataset...")
    dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=16, image_size=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load tokenizer
    tokenizer_path = os.path.join(data_dir, 'train', '..', '..', 'text_image', 'tokenizer', 'tokenizer.json')
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)
        idx_to_word = {}
        word_to_idx = {}
        for key, value in data['idx_to_word'].items():
            if key.startswith('<'):
                idx = int(value)
                token = key
                idx_to_word[idx] = token
                word_to_idx[token] = idx
            else:
                idx = int(key)
                token = value
                idx_to_word[idx] = token
                word_to_idx[token] = idx
        vocab_size = len(word_to_idx)
    else:
        all_words = set()
        for caption in dataset.captions:
            all_words.update(caption.lower().split())
        word_to_idx = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        for i, word in enumerate(sorted(all_words), start=3):
            word_to_idx[word] = i
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        vocab_size = len(word_to_idx)

    # Create models
    print("Creating models...")
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

    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        spatial_unet.load_state_dict(state_dict, strict=False)
        if 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

    # Create instrumented video U-Net
    video_unet = InstrumentedVideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()

    video_unet = video_unet.to(device)

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # Create optimizer for gradient computation
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, video_unet.parameters()), lr=1e-5)

    print(f"\nEvaluating {num_samples} samples...")
    for i, batch in enumerate(tqdm(dataloader, total=num_samples)):
        if i >= num_samples:
            break

        videos = batch['video'].to(device)
        captions = batch['caption']
        batch_size = videos.shape[0]
        num_frames = 16
        image_size = 64

        # Encode captions
        text_embs = []
        text_masks = []
        for caption in captions:
            token_ids, mask = video_unet.tokenizer.encode(caption, max_length=32)
            text_embs.append(token_ids)
            text_masks.append(mask)

        max_len = max(len(emb) for emb in text_embs)
        padded_embs = []
        padded_masks = []
        for emb, mask in zip(text_embs, text_masks):
            padded = emb + [0] * (max_len - len(emb))
            padded_mask = mask + [0] * (max_len - len(mask))
            padded_embs.append(padded)
            padded_masks.append(padded_mask)

        text_emb = torch.tensor(padded_embs).to(device)
        text_mask = torch.tensor(padded_masks).to(device)
        text_emb_encoded, _ = video_unet.text_encoder(text_emb, text_mask)
        text_emb = text_emb_encoded

        # Sample timesteps
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

        # Forward diffusion
        videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)

        noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)
        noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

        # Forward pass
        predicted_noise = video_unet(noisy_videos, timesteps, text_emb, text_mask)
        predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)

        # Compute loss
        loss = nn.functional.mse_loss(predicted_noise_flat, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Record gradients
        video_unet.record_gradients()

    # Compute statistics
    stats = video_unet.get_statistics()

    # Print results
    print("\n" + "=" * 80)
    print("TEMPORAL FEATURE STATISTICS")
    print("=" * 80)

    print("\nBOTTLENECK LEVEL:")
    print(f"  Spatial RMS:          {stats['bottleneck_spatial_rms']['mean']:.4f} +/- {stats['bottleneck_spatial_rms']['std']:.4f}")
    print(f"  Temporal RMS:         {stats['bottleneck_temporal_rms']['mean']:.4f} +/- {stats['bottleneck_temporal_rms']['std']:.4f}")
    print(f"  Residual RMS:         {stats['bottleneck_residual_rms']['mean']:.4f} +/- {stats['bottleneck_residual_rms']['std']:.4f}")
    print(f"  Fused RMS:            {stats['bottleneck_fused_rms']['mean']:.4f} +/- {stats['bottleneck_fused_rms']['std']:.4f}")
    print(f"  Gate value:           {stats['bottleneck_gate']['mean']:.6f} +/- {stats['bottleneck_gate']['std']:.6f}")
    print(f"  Temporal grad norm:   {stats['bottleneck_temporal_grad_norm']['mean']:.6f} +/- {stats['bottleneck_temporal_grad_norm']['std']:.6f}")
    print(f"  Fusion grad norm:     {stats['bottleneck_fusion_grad_norm']['mean']:.6f} +/- {stats['bottleneck_fusion_grad_norm']['std']:.6f}")

    # Compute contribution ratios
    temporal_to_spatial = stats['bottleneck_temporal_rms']['mean'] / stats['bottleneck_spatial_rms']['mean']
    residual_to_spatial = stats['bottleneck_residual_rms']['mean'] / stats['bottleneck_spatial_rms']['mean']
    print(f"  Temporal/Spatial ratio: {temporal_to_spatial:.4f}")
    print(f"  Residual/Spatial ratio: {residual_to_spatial:.4f}")

    print("\nLOW-RES LEVEL:")
    print(f"  Spatial RMS:          {stats['low_res_spatial_rms']['mean']:.4f} +/- {stats['low_res_spatial_rms']['std']:.4f}")
    print(f"  Temporal RMS:         {stats['low_res_temporal_rms']['mean']:.4f} +/- {stats['low_res_temporal_rms']['std']:.4f}")
    print(f"  Residual RMS:         {stats['low_res_residual_rms']['mean']:.4f} +/- {stats['low_res_residual_rms']['std']:.4f}")
    print(f"  Fused RMS:            {stats['low_res_fused_rms']['mean']:.4f} +/- {stats['low_res_fused_rms']['std']:.4f}")
    print(f"  Gate value:           {stats['low_res_gate']['mean']:.6f} +/- {stats['low_res_gate']['std']:.6f}")
    print(f"  Temporal grad norm:   {stats['low_res_temporal_grad_norm']['mean']:.6f} +/- {stats['low_res_temporal_grad_norm']['std']:.6f}")
    print(f"  Fusion grad norm:     {stats['low_res_fusion_grad_norm']['mean']:.6f} +/- {stats['low_res_fusion_grad_norm']['std']:.6f}")

    # Compute contribution ratios
    temporal_to_spatial = stats['low_res_temporal_rms']['mean'] / stats['low_res_spatial_rms']['mean']
    residual_to_spatial = stats['low_res_residual_rms']['mean'] / stats['low_res_spatial_rms']['mean']
    print(f"  Temporal/Spatial ratio: {temporal_to_spatial:.4f}")
    print(f"  Residual/Spatial ratio: {residual_to_spatial:.4f}")

    print("\n" + "=" * 80)

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose temporal feature contribution")
    parser.add_argument("--data_dir", type=str, default="data/controlled_video", help="Data directory")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints_controlled/checkpoint_best.pt", help="Checkpoint path")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    stats = diagnose_temporal_contribution(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        device=args.device
    )

    # Save results
    output_path = "temporal_contribution_diagnosis.json"
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
