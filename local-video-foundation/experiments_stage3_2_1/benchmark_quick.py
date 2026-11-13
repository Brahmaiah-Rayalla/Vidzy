"""
Quick CPU benchmark for Stage 3.2.5

Measures only essential timing for training step estimation.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import psutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


def quick_benchmark():
    print("=" * 80)
    print("STAGE 3.2.5 QUICK CPU BENCHMARK")
    print("=" * 80)
    
    device = 'cpu'
    num_frames = 16
    image_size = 64
    batch_size = 1
    
    # Load Stage 2 checkpoint
    print("\n[1] Loading checkpoint...")
    spatial_checkpoint_path = "../checkpoints_conditional/checkpoint_final.pt"
    checkpoint = torch.load(spatial_checkpoint_path, map_location='cpu')
    
    if 'text_encoder_state_dict' in checkpoint and 'token_emb.weight' in checkpoint['text_encoder_state_dict']:
        vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]
    else:
        vocab_size = 42
    
    # Create models
    print("[2] Creating models...")
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
    
    if 'model_state_dict' in checkpoint:
        spatial_unet.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if 'text_encoder_state_dict' in checkpoint:
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=True,
        use_temporal=True
    )
    
    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: ([1] * 5 + [2], [1] * 6),
        'word_to_idx': {'<pad>': 0, '<unk>': 1, '<eos>': 2, 'test': 3, 'caption': 4},
        'idx_to_word': {0: '<pad>', 1: '<unk>', 2: '<eos>', 3: 'test', 4: 'caption'},
        'vocab_size': 5
    })()
    
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)
    
    print("\n[3] System Info")
    print(f"  CPU cores: {psutil.cpu_count(logical=True)}")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    # Parameter counts
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.bottleneck_fusion.parameters()) + list(video_unet.low_res_fusion.parameters())
    temporal_scalar_count = sum(p.numel() for p in temporal_params)
    fusion_scalar_count = sum(p.numel() for p in fusion_params)
    
    print(f"\n[4] Parameter Counts")
    print(f"  Trainable scalar parameters: {temporal_scalar_count + fusion_scalar_count:,}")
    
    # Prepare data
    print("\n[5] Preparing test data...")
    clean_video = torch.randn(batch_size, 3, num_frames, image_size, image_size)
    clean_video = (clean_video - clean_video.min()) / (clean_video.max() - clean_video.min()) * 2 - 1
    
    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)
    t = torch.randint(0, 1000, (batch_size,))
    
    videos_flat = clean_video.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
    t_flat = t.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)
    noisy_videos, noise = diffusion.q_sample(videos_flat, t_flat)
    noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)
    
    # Create optimizer
    import torch.optim as optim
    optimizer = optim.Adam([
        {'params': temporal_params, 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': fusion_params, 'lr': 1e-4, 'weight_decay': 1e-6}
    ])
    
    # Warm-up
    print("[6] Warm-up (3 steps)...")
    for _ in range(3):
        predicted_noise = video_unet(noisy_videos, t, text_emb, text_mask)
        predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        loss = nn.functional.mse_loss(predicted_noise_flat, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Measure training step time
    print("\n[7] Measuring training step time (10 iterations)...")
    step_times = []
    
    for i in range(10):
        start = time.time()
        
        predicted_noise = video_unet(noisy_videos, t, text_emb, text_mask)
        predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        loss = nn.functional.mse_loss(predicted_noise_flat, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end = time.time()
        step_times.append(end - start)
        print(f"  Step {i+1}/10: {end - start:.4f} s")
    
    avg_step_time = np.mean(step_times)
    std_step_time = np.std(step_times)
    
    print(f"\n[8] Results")
    print(f"  Average step time: {avg_step_time:.4f} s")
    print(f"  Std dev: {std_step_time:.4f} s")
    
    # Estimate training time
    num_steps_target = 500
    estimated_total_time = avg_step_time * num_steps_target
    
    print(f"\n[9] Training Time Estimation")
    print(f"  {num_steps_target} steps: {estimated_total_time / 60:.2f} minutes ({estimated_total_time / 3600:.2f} hours)")
    print(f"  Per 50 steps: {avg_step_time * 50 / 60:.2f} minutes")
    print(f"  Per 100 steps: {avg_step_time * 100 / 60:.2f} minutes")
    
    results = {
        'trainable_param_scalars': int(temporal_scalar_count + fusion_scalar_count),
        'cpu_cores': psutil.cpu_count(logical=True),
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'avg_step_time_s': float(avg_step_time),
        'std_step_time_s': float(std_step_time),
        'estimated_500_steps_min': float(estimated_total_time / 60),
        'estimated_500_steps_hours': float(estimated_total_time / 3600),
    }
    
    with open("quick_benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to quick_benchmark_results.json")
    
    return results


if __name__ == "__main__":
    quick_benchmark()
