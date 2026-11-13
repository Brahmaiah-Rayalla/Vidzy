"""
Stage 3.2.5: CPU Benchmark

Estimate training time and measure performance metrics before full training.
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


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def benchmark_cpu():
    print_section("STAGE 3.2.5 CPU BENCHMARK")
    
    device = 'cpu'
    num_frames = 16
    image_size = 64
    batch_size = 1
    
    # Load Stage 2 checkpoint
    print("\n[1] Loading Stage 2 checkpoint...")
    spatial_checkpoint_path = "../checkpoints_conditional/checkpoint_final.pt"
    checkpoint = torch.load(spatial_checkpoint_path, map_location='cpu')
    
    if 'text_encoder_state_dict' in checkpoint and 'token_emb.weight' in checkpoint['text_encoder_state_dict']:
        vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]
    else:
        vocab_size = 42
    
    # Create models
    print("\n[2] Creating models...")
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
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        spatial_unet.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if 'text_encoder_state_dict' in checkpoint:
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    
    # Create video UNet
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=True,
        use_temporal=True
    )
    
    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [1] * min(len("test caption".split()), max_length) + [2],
            [1] * (min(len("test caption".split()), max_length) + 1)
        ),
        'word_to_idx': {'<pad>': 0, '<unk>': 1, '<eos>': 2, 'test': 3, 'caption': 4},
        'idx_to_word': {0: '<pad>', 1: '<unk>', 2: '<eos>', 3: 'test', 4: 'caption'},
        'vocab_size': 5
    })()
    
    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)
    
    print_section("3. SYSTEM INFORMATION")
    
    print(f"\nCPU: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB total, {psutil.virtual_memory().available / (1024**3):.2f} GB available")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    print_section("4. MODEL PARAMETER COUNTS")
    
    spatial_params = list(video_unet.spatial_unet.parameters())
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.bottleneck_fusion.parameters()) + list(video_unet.low_res_fusion.parameters())
    
    spatial_scalar_count = sum(p.numel() for p in spatial_params)
    temporal_scalar_count = sum(p.numel() for p in temporal_params)
    fusion_scalar_count = sum(p.numel() for p in fusion_params)
    
    print(f"\nSpatial parameter tensors: {len(spatial_params)}")
    print(f"Spatial scalar parameters: {spatial_scalar_count:,}")
    print(f"Temporal parameter tensors: {len(temporal_params)}")
    print(f"Temporal scalar parameters: {temporal_scalar_count:,}")
    print(f"Fusion parameter tensors: {len(fusion_params)}")
    print(f"Fusion scalar parameters: {fusion_scalar_count:,}")
    print(f"Total trainable scalar parameters: {temporal_scalar_count + fusion_scalar_count:,}")
    
    print_section("5. MEMORY FOOTPRINT")
    
    # Measure memory before forward pass
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**2)
    
    # Generate test data
    clean_video = torch.randn(batch_size, 3, num_frames, image_size, image_size)
    clean_video = (clean_video - clean_video.min()) / (clean_video.max() - clean_video.min()) * 2 - 1
    
    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)
    
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward diffusion
    videos_flat = clean_video.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
    t_flat = t.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)
    noisy_videos, noise = diffusion.q_sample(videos_flat, t_flat)
    noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)
    
    # Measure memory after data loading
    mem_after_data = process.memory_info().rss / (1024**2)
    
    # Forward pass
    video_unet.eval()
    with torch.no_grad():
        predicted_noise = video_unet(noisy_videos, t, text_emb, text_mask)
    
    # Measure memory after forward pass
    mem_after_forward = process.memory_info().rss / (1024**2)
    
    print(f"\nMemory usage:")
    print(f"  Baseline: {mem_before:.2f} MB")
    print(f"  After data loading: {mem_after_data:.2f} MB (+{mem_after_data - mem_before:.2f} MB)")
    print(f"  After forward pass: {mem_after_forward:.2f} MB (+{mem_after_forward - mem_after_data:.2f} MB)")
    print(f"  Peak overhead: {mem_after_forward - mem_before:.2f} MB")
    
    print_section("6. FORWARD PASS TIMING")
    
    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            _ = video_unet(noisy_videos, t, text_emb, text_mask)
    
    # Measure forward pass time
    num_iterations = 10
    forward_times = []
    
    for _ in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            _ = video_unet(noisy_videos, t, text_emb, text_mask)
        end = time.time()
        forward_times.append(end - start)
    
    avg_forward_time = np.mean(forward_times)
    std_forward_time = np.std(forward_times)
    
    print(f"\nForward pass (inference):")
    print(f"  Average: {avg_forward_time:.4f} s")
    print(f"  Std dev: {std_forward_time:.4f} s")
    print(f"  Min: {np.min(forward_times):.4f} s")
    print(f"  Max: {np.max(forward_times):.4f} s")
    
    print_section("7. TRAINING STEP TIMING")
    
    video_unet.train()
    
    # Create optimizer
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.bottleneck_fusion.parameters()) + list(video_unet.low_res_fusion.parameters())
    
    import torch.optim as optim
    optimizer = optim.Adam([
        {'params': temporal_params, 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': fusion_params, 'lr': 1e-4, 'weight_decay': 1e-6}
    ])
    
    # Warm-up
    for _ in range(3):
        predicted_noise = video_unet(noisy_videos, t, text_emb, text_mask)
        predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        loss = nn.functional.mse_loss(predicted_noise_flat, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Measure training step time
    num_iterations = 10
    step_times = []
    
    for _ in range(num_iterations):
        start = time.time()
        
        predicted_noise = video_unet(noisy_videos, t, text_emb, text_mask)
        predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        loss = nn.functional.mse_loss(predicted_noise_flat, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end = time.time()
        step_times.append(end - start)
    
    avg_step_time = np.mean(step_times)
    std_step_time = np.std(step_times)
    
    print(f"\nTraining step (forward + backward + optimizer):")
    print(f"  Average: {avg_step_time:.4f} s")
    print(f"  Std dev: {std_step_time:.4f} s")
    print(f"  Min: {np.min(step_times):.4f} s")
    print(f"  Max: {np.max(step_times):.4f} s")
    
    print_section("8. DDIM SAMPLING TIMING")
    
    # Test DDIM sampling
    video_unet.eval()
    
    # Generate a clean video
    clean_video = torch.randn(batch_size, 3, num_frames, image_size, image_size)
    clean_video = (clean_video - clean_video.min()) / (clean_video.max() - clean_video.min()) * 2 - 1
    
    # Sample from noise
    noisy_video = torch.randn_like(clean_video)
    
    # DDIM sampling
    num_inference_steps = 50
    sampling_times = []
    
    for _ in range(3):  # Warm-up
        with torch.no_grad():
            x = noisy_video.clone()
            for i in range(num_inference_steps):
                t = torch.full((batch_size,), (num_inference_steps - 1 - i) * (1000 // num_inference_steps))
                pred_noise = video_unet(x, t, text_emb, text_mask)
                # Simplified DDIM step (not full implementation)
                x = x - 0.1 * pred_noise
    
    for _ in range(5):
        start = time.time()
        with torch.no_grad():
            x = noisy_video.clone()
            for i in range(num_inference_steps):
                t = torch.full((batch_size,), (num_inference_steps - 1 - i) * (1000 // num_inference_steps))
                pred_noise = video_unet(x, t, text_emb, text_mask)
                x = x - 0.1 * pred_noise
        end = time.time()
        sampling_times.append(end - start)
    
    avg_sampling_time = np.mean(sampling_times)
    
    print(f"\nDDIM sampling ({num_inference_steps} steps):")
    print(f"  Average: {avg_sampling_time:.4f} s")
    print(f"  Per step: {avg_sampling_time / num_inference_steps:.4f} s")
    
    print_section("9. TRAINING TIME ESTIMATION")
    
    num_steps_target = 500
    estimated_total_time = avg_step_time * num_steps_target
    
    print(f"\nEstimated training time for {num_steps_target} steps:")
    print(f"  Total: {estimated_total_time / 60:.2f} minutes")
    print(f"  Per step: {avg_step_time:.4f} s")
    print(f"  Per 50 steps: {avg_step_time * 50 / 60:.2f} minutes")
    print(f"  Per 100 steps: {avg_step_time * 100 / 60:.2f} minutes")
    
    print_section("10. BENCHMARK SUMMARY")
    
    results = {
        'cpu_cores_physical': psutil.cpu_count(logical=False),
        'cpu_cores_logical': psutil.cpu_count(logical=True),
        'ram_total_gb': psutil.virtual_memory().total / (1024**3),
        'ram_available_gb': psutil.virtual_memory().available / (1024**3),
        'spatial_param_tensors': len(spatial_params),
        'spatial_param_scalars': int(spatial_scalar_count),
        'temporal_param_tensors': len(temporal_params),
        'temporal_param_scalars': int(temporal_scalar_count),
        'fusion_param_tensors': len(fusion_params),
        'fusion_param_scalars': int(fusion_scalar_count),
        'trainable_param_scalars': int(temporal_scalar_count + fusion_scalar_count),
        'memory_baseline_mb': float(mem_before),
        'memory_after_data_mb': float(mem_after_data),
        'memory_after_forward_mb': float(mem_after_forward),
        'memory_overhead_mb': float(mem_after_forward - mem_before),
        'forward_pass_avg_s': float(avg_forward_time),
        'forward_pass_std_s': float(std_forward_time),
        'training_step_avg_s': float(avg_step_time),
        'training_step_std_s': float(std_step_time),
        'ddim_sampling_avg_s': float(avg_sampling_time),
        'ddim_sampling_per_step_s': float(avg_sampling_time / num_inference_steps),
        'estimated_500_steps_min': float(estimated_total_time / 60),
        'estimated_500_steps_hours': float(estimated_total_time / 3600),
    }
    
    with open("cpu_benchmark_stage3_2_5.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to cpu_benchmark_stage3_2_5.json")
    print(f"\nRecommendation: {num_steps_target} steps will take approximately {estimated_total_time / 60:.2f} minutes")
    
    return results


if __name__ == "__main__":
    benchmark_cpu()
