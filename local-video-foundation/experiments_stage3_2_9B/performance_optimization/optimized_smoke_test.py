"""
Stage 3.2.9B - Phase 2B: Optimized CPU Training Smoke Test

Copy of original smoke test with CPU optimizations applied.

Author: Devin
Date: 2026-11-02
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import psutil

# Apply CPU optimizations BEFORE importing PyTorch modules
def apply_cpu_optimizations():
    """Apply CPU performance optimizations."""
    logical_cores = psutil.cpu_count(logical=True)
    torch.set_num_threads(logical_cores)
    torch.set_num_interop_threads(logical_cores)
    os.environ['MKL_NUM_THREADS'] = str(logical_cores)
    os.environ['OMP_NUM_THREADS'] = str(logical_cores)
    return logical_cores

# Apply optimizations
thread_count = apply_cpu_optimizations()

# Add paths
parent_dir = Path(__file__).parent.parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))

from video_foundation.models.conditional_unet import ConditionalUNet as OriginalConditionalUNet
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed
from models.text_encoder_split import load_from_standard_encoder


class SyntheticVideoDataset(Dataset):
    """Synthetic video dataset."""

    def __init__(self, metadata_path: str):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"Loaded {len(self.metadata)} samples")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata[idx]
        video_path = Path(sample['video_path'])
        video = np.load(video_path)
        video = torch.from_numpy(video).float() / 255.0
        video = video.permute(3, 0, 1, 2)  # (C, T, H, W)
        return {'video': video, 'caption': sample['caption']}


def load_models():
    """Load Stage 3 models."""
    device = torch.device('cpu')

    stage2_checkpoint = parent_dir / "checkpoints_conditional" / "checkpoint_final.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location=device)
    actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

    text_encoder = TextEncoder(
        vocab_size=actual_vocab_size, d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0
    )
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
    )
    spatial_unet.load_state_dict(checkpoint['model_state_dict'])

    split_encoder = load_from_standard_encoder(
        text_encoder, motion_vocab_size=40, freeze_transformer=True
    )
    split_encoder.eval()
    split_encoder = split_encoder.to(device)

    return spatial_unet, split_encoder, device


def get_diffusion_schedule():
    """Get diffusion beta schedule."""
    num_timesteps = 1000
    betas = torch.linspace(0.0001, 0.02, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def add_noise(x0, t):
    """Add noise to clean video at timestep t."""
    betas, alphas, alpha_bars = get_diffusion_schedule()
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t])

    # Reshape for broadcasting
    sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1)

    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt, noise


def run_experiment(experiment_id: str, enable_temporal: bool, num_steps: int = 10):
    """Run experiment with timing."""
    print("\n" + "=" * 80)
    print(f"  EXPERIMENT {experiment_id}: {'TEMPORAL ENABLED' if enable_temporal else 'TEMPORAL DISABLED'}")
    print("=" * 80)

    spatial_unet, split_encoder, device = load_models()

    video_unet = VideoUNetFixed(
        spatial_unet=spatial_unet,
        text_encoder=split_encoder,
        num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True, enable_temporal_processing=enable_temporal
    )
    video_unet = video_unet.to(device)

    # Configure trainable parameters
    if enable_temporal:
        trainable_params = list(split_encoder.token_emb.motion_emb.parameters())
        trainable_params.extend(list(video_unet.temporal_modules.parameters()))
        trainable_params.extend(list(video_unet.fusion_modules.parameters()))
    else:
        trainable_params = list(split_encoder.token_emb.motion_emb.parameters())

    optimizer = optim.AdamW(trainable_params, lr=0.0002, weight_decay=0.01)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    metadata_path = stage3_2_7_dir / "data_synthetic" / "train_metadata.json"
    dataset = SyntheticVideoDataset(metadata_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    tokenizer_path = stage3_2_7_dir / "checkpoints" / "tokenizer_motion_expanded.json"
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    results = {
        'losses': [],
        'timing': {
            'data_loading': [],
            'forward_pass': [],
            'backward_pass': [],
            'optimizer_step': [],
            'total_step': []
        },
        'optimizer_step_count': 0
    }

    print(f"\nStarting training for {num_steps} steps...")
    print(f"CPU frequency: {psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'} MHz")
    print(f"Thread count: {torch.get_num_threads()}")

    data_iter = iter(dataloader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        step_start = time.time()

        # Data loading
        data_start = time.time()
        videos = batch['video'].to(device)
        captions = batch['caption']
        data_end = time.time()

        # Text encoding
        text_start = time.time()
        token_ids_list = []
        mask_list = []
        for caption in captions:
            token_ids, mask = tokenizer.encode(caption, max_length=64)
            token_ids_list.append(token_ids)
            mask_list.append(mask)
        token_ids = torch.tensor(token_ids_list)
        mask = torch.tensor(mask_list, dtype=torch.float)
        text_end = time.time()

        # Forward pass
        forward_start = time.time()
        text_emb, text_mask = video_unet.text_encoder(token_ids, mask)
        t = torch.randint(0, 1000, (videos.shape[0],))
        xt, noise = add_noise(videos, t)
        output = video_unet(xt, t, text_emb, text_mask)
        forward_end = time.time()

        # Loss
        loss = nn.functional.mse_loss(output, noise)

        # Backward pass
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_end = time.time()

        # Optimizer step
        opt_start = time.time()
        optimizer.step()
        opt_end = time.time()

        step_end = time.time()

        results['timing']['data_loading'].append(data_end - data_start)
        results['timing']['forward_pass'].append(forward_end - forward_start)
        results['timing']['backward_pass'].append(backward_end - backward_start)
        results['timing']['optimizer_step'].append(opt_end - opt_start)
        results['timing']['total_step'].append(step_end - step_start)
        results['losses'].append(loss.item())
        results['optimizer_step_count'] += 1

        print(f"Step {step + 1}/{num_steps}: Loss = {loss.item():.6f}, Time = {step_end - step_start:.2f}s")

    return results


def main():
    print("\n" + "=" * 80)
    print("  OPTIMIZED CPU SMOKE TEST")
    print("=" * 80)
    print(f"\nOptimizations applied:")
    print(f"  Thread count: {thread_count}")
    print(f"  CPU frequency: {psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'} MHz")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")

    results_B = run_experiment("B", enable_temporal=False, num_steps=10)
    results_C = run_experiment("C", enable_temporal=True, num_steps=10)

    results = {
        'experiment_B': results_B,
        'experiment_C': results_C,
        'optimization_config': {
            'thread_count': thread_count,
            'cpu_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'mkl_threads': os.environ.get('MKL_NUM_THREADS'),
            'omp_threads': os.environ.get('OMP_NUM_THREADS')
        }
    }

    output_path = Path(__file__).parent / "optimized_smoke_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
