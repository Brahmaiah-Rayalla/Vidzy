"""
CPU performance benchmarking for video generation.

Measures the performance of video generation on CPU across different configurations.
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


def load_model(checkpoint_path, num_frames=16, image_size=64, device='cpu'):
    """Load the video model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
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


def ddim_sample_step(model, x, t, text_emb, text_mask, diffusion):
    """Perform a single DDIM sampling step."""
    batch_size = x.shape[0]

    # Predict noise at current timestep
    t_batch = t.expand(batch_size)
    predicted_noise = model(x, t_batch, text_emb, text_mask)

    # Get schedule values for current and previous timesteps
    _, alpha_bar_t, _ = diffusion.schedule.get(t_batch, x.device)

    # Predict x_0
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    x_0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

    return x_0_pred, predicted_noise


def benchmark_sampling(
    model,
    diffusion,
    text_emb,
    text_mask,
    shape,
    num_sampling_steps,
    device='cpu',
    num_runs=2
):
    """
    Benchmark DDIM sampling performance.

    Args:
        model: Video UNet model
        diffusion: Diffusion process
        text_emb: Text embeddings
        text_mask: Text attention mask
        shape: Output shape [batch, channels, frames, height, width]
        num_sampling_steps: Number of DDIM steps
        device: Device
        num_runs: Number of benchmark runs

    Returns:
        Dictionary of benchmark results
    """
    batch_size, channels, num_frames, height, width = shape

    # Create subsampled timesteps
    total_timesteps = diffusion.schedule.num_timesteps
    timesteps = torch.linspace(total_timesteps - 1, 0, num_sampling_steps, dtype=torch.long, device=device)

    times = []
    step_times = []

    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}...")
        start_time = time.time()

        # Start from pure Gaussian noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion loop
        for i, t in enumerate(timesteps):
            step_start = time.time()

            # Perform sampling step
            x_0_pred, predicted_noise = ddim_sample_step(model, x, t, text_emb, text_mask, diffusion)

            # Get schedule values for previous timestep
            if i < num_sampling_steps - 1:
                t_prev = timesteps[i + 1]
                t_prev_batch = t_prev.expand(batch_size)
                _, alpha_bar_t_prev, _ = diffusion.schedule.get(t_prev_batch, device)
            else:
                alpha_bar_t_prev = torch.ones_like(diffusion.schedule.get(t, device)[1])

            # DDIM update
            sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
            sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)

            # Compute x_{t-1}
            x = sqrt_alpha_bar_t_prev * x_0_pred + sqrt_one_minus_alpha_bar_t_prev * predicted_noise

            step_time = time.time() - step_start
            step_times.append(step_time)

        total_time = time.time() - start_time
        times.append(total_time)

    # Compute statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    avg_step_time = np.mean(step_times)
    std_step_time = np.std(step_times)

    return {
        'avg_total_time': float(avg_time),
        'std_total_time': float(std_time),
        'min_total_time': float(min_time),
        'max_total_time': float(max_time),
        'avg_step_time': float(avg_step_time),
        'std_step_time': float(std_step_time),
        'num_runs': num_runs,
        'num_steps': num_sampling_steps
    }


def benchmark_model_forward(model, shape, device='cpu', num_runs=5):
    """
    Benchmark the model forward pass (single step).

    Args:
        model: Video UNet model
        shape: Input shape [batch, channels, frames, height, width]
        device: Device
        num_runs: Number of benchmark runs

    Returns:
        Dictionary of benchmark results
    """
    print("Benchmarking model forward pass...")

    times = []

    for run in range(num_runs):
        x = torch.randn(shape, device=device)
        t = torch.randint(0, 1000, (shape[0],), device=device)
        text_emb = torch.randn(shape[0], 32, 256, device=device)
        text_mask = torch.ones(shape[0], 32, device=device)

        start_time = time.time()

        with torch.no_grad():
            output = model(x, t, text_emb, text_mask)

        elapsed = time.time() - start_time
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    return {
        'avg_time': float(avg_time),
        'std_time': float(std_time),
        'min_time': float(min_time),
        'max_time': float(max_time),
        'num_runs': num_runs
    }


def run_comprehensive_benchmark(
    checkpoint_path,
    num_frames=16,
    image_size=64,
    device='cpu',
    output_file=None
):
    """
    Run comprehensive CPU performance benchmarks.

    Args:
        checkpoint_path: Path to model checkpoint
        num_frames: Number of frames
        image_size: Image size
        device: Device
        output_file: Optional output file for results

    Returns:
        Dictionary of all benchmark results
    """
    print("=" * 60)
    print("CPU PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Frames: {num_frames}")
    print(f"Resolution: {image_size}x{image_size}")
    print(f"Device: {device}")
    print("=" * 60)
    print()

    # Load model
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

    all_results = {}

    # Benchmark model forward pass
    print("1. Benchmarking single forward pass...")
    forward_results = benchmark_model_forward(model, shape, device, num_runs=5)
    all_results['forward_pass'] = forward_results
    print(f"   Average time: {forward_results['avg_time']:.4f}s")
    print()

    # Benchmark different numbers of sampling steps
    print("2. Benchmarking DDIM sampling with different step counts...")
    step_counts = [5, 10, 25]
    sampling_results = {}

    for num_steps in step_counts:
        print(f"   Testing {num_steps} steps...")
        results = benchmark_sampling(
            model, diffusion, text_emb, text_mask, shape,
            num_sampling_steps=num_steps, device=device, num_runs=2
        )
        sampling_results[f'{num_steps}_steps'] = results
        print(f"     Average total time: {results['avg_total_time']:.2f}s")
        print(f"     Average step time: {results['avg_step_time']:.4f}s")
        print()

    all_results['sampling'] = sampling_results

    # Compute performance metrics
    print("3. Performance Summary:")
    print("=" * 60)

    # Throughput metrics
    avg_25_step_time = sampling_results['25_steps']['avg_total_time']
    fps = num_frames / avg_25_step_time
    print(f"  Frames per second (25 steps): {fps:.2f}")
    print(f"  Seconds per frame (25 steps): {avg_25_step_time/num_frames:.4f}s")
    print(f"  Average forward pass time: {forward_results['avg_time']:.4f}s")
    print(f"  Forward passes per second: {1/forward_results['avg_time']:.2f}")
    print("=" * 60)

    all_results['summary'] = {
        'fps_25_steps': float(fps),
        'seconds_per_frame_25_steps': float(avg_25_step_time/num_frames),
        'forward_passes_per_second': float(1/forward_results['avg_time'])
    }

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results to {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="CPU performance benchmarking for video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive benchmark
  python benchmark_cpu.py --checkpoint checkpoints_video_smoke/checkpoint_step_000010.pt --output benchmark_results.json

  # Benchmark with different resolution
  python benchmark_cpu.py --checkpoint checkpoints_video_smoke/checkpoint_step_000010.pt --image_size 128 --output benchmark_128.json
        """
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames (default: 16)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Image resolution (default: 64)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device (default: cpu)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for benchmark results"
    )

    args = parser.parse_args()

    run_comprehensive_benchmark(
        checkpoint_path=args.checkpoint,
        num_frames=args.num_frames,
        image_size=args.image_size,
        device=args.device,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
