"""
Quick CPU performance benchmark for video generation.

Simplified benchmark to measure basic performance metrics.
"""

import os
import sys
import torch
import numpy as np
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


def load_model(checkpoint_path, num_frames=16, image_size=64, device='cpu'):
    """Load the video model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load tokenizer
    tokenizer_path = "data/text_image/tokenizer/tokenizer.json"
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

    # Create spatial U-Net
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

    # Create text encoder
    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=True,
        use_temporal=True
    )

    # Load model state
    video_unet.load_state_dict(checkpoint['model_state_dict'])
    video_unet = video_unet.to(device)
    video_unet.eval()

    # Create tokenizer
    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()

    return video_unet


def main():
    print("=" * 60)
    print("QUICK CPU PERFORMANCE BENCHMARK")
    print("=" * 60)

    checkpoint_path = "checkpoints_video_smoke/checkpoint_step_000010.pt"
    num_frames = 16
    image_size = 64
    device = 'cpu'

    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, num_frames, image_size, device)

    # Encode a sample prompt
    prompt = "A red circle moves from left to right."
    token_ids, mask = model.tokenizer.encode(prompt, max_length=32)
    max_len = len(token_ids)
    token_ids = token_ids + [0] * (32 - max_len)
    mask = mask + [0] * (32 - len(mask))

    text_emb_tokens = torch.tensor([token_ids]).to(device)
    text_mask = torch.tensor([mask]).to(device)

    with torch.no_grad():
        text_emb, _ = model.text_encoder(text_emb_tokens, text_mask)

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    shape = (1, 3, num_frames, image_size, image_size)

    print("\nBenchmarking model forward pass (3 runs)...")
    forward_times = []

    for i in range(3):
        x = torch.randn(shape, device=device)
        t = torch.randint(0, 1000, (shape[0],), device=device)

        start_time = time.time()
        with torch.no_grad():
            output = model(x, t, text_emb, text_mask)
        elapsed = time.time() - start_time
        forward_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")

    avg_forward = np.mean(forward_times)
    print(f"  Average: {avg_forward:.4f}s")
    print(f"  Forward passes per second: {1/avg_forward:.2f}")

    print("\nBenchmarking DDIM sampling (10 steps, 2 runs)...")
    sampling_times = []

    for run in range(2):
        print(f"  Run {run+1}...")
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(999, 0, 10, dtype=torch.long, device=device)

        start_time = time.time()

        for i, t in enumerate(timesteps):
            t_batch = t.expand(1)
            predicted_noise = model(x, t_batch, text_emb, text_mask)

            _, alpha_bar_t, _ = diffusion.schedule.get(t_batch, device)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            x_0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

            if i < 9:
                t_prev = timesteps[i + 1]
                t_prev_batch = t_prev.expand(1)
                _, alpha_bar_t_prev, _ = diffusion.schedule.get(t_prev_batch, device)
            else:
                alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

            sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
            sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)
            x = sqrt_alpha_bar_t_prev * x_0_pred + sqrt_one_minus_alpha_bar_t_prev * predicted_noise

        elapsed = time.time() - start_time
        sampling_times.append(elapsed)
        print(f"    Total time: {elapsed:.2f}s")

    avg_sampling = np.mean(sampling_times)
    print(f"  Average: {avg_sampling:.2f}s")
    print(f"  Time per step: {avg_sampling/10:.4f}s")
    print(f"  FPS: {num_frames/avg_sampling:.2f}")

    # Save results
    results = {
        'checkpoint': checkpoint_path,
        'num_frames': num_frames,
        'image_size': image_size,
        'device': device,
        'forward_pass': {
            'avg_time': float(avg_forward),
            'passes_per_second': float(1/avg_forward)
        },
        'ddim_sampling_10_steps': {
            'avg_time': float(avg_sampling),
            'time_per_step': float(avg_sampling/10),
            'fps': float(num_frames/avg_sampling)
        }
    }

    with open('quick_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to quick_benchmark_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
