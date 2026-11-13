"""
CPU Inference Benchmark for Stage 2.3.2

Benchmarks DDPM and DDIM sampling configurations with:
- Latency measurements (total, median, 90th percentile)
- Time per image
- Time per denoising step
- Peak process RAM
- CPU utilization
- Model loading time
- Images per second

Tests:
- DDPM: 1000 steps (1 repetition due to slow speed)
- DDIM: 25, 50, 100, 200 steps (multiple repetitions)
"""
import os
import sys
import time
import psutil
import statistics
import json
import csv
from datetime import datetime

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device


def optimize_cpu_threads(num_threads):
    """Optimize PyTorch CPU thread configuration."""
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, num_threads // 2))


def run_benchmark():
    # Load config
    with open('configs/conditional_v0.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    set_seed(42)

    print("="*70)
    print("CPU INFERENCE BENCHMARK - STAGE 2.3.2")
    print("="*70)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {psutil.cpu_count(logical=True)} physical, {psutil.cpu_count()} logical")
    print("="*70)

    # Optimize threads (default to 8 for benchmark)
    num_threads = 8
    optimize_cpu_threads(num_threads)
    print(f"Configured threads: {num_threads} main, {max(1, num_threads // 2)} interop")

    # Load tokenizer
    print("\nLoading tokenizer...")
    train_metadata_path = "data/text_image/train_metadata.json"
    with open(train_metadata_path, 'r') as f:
        train_metadata = json.load(f)
    captions = [m['caption'] for m in train_metadata]
    tokenizer = SimpleTokenizer(
        vocab_size=config['text_encoder']['vocab_size'],
        min_freq=config['text_encoder']['min_freq']
    )
    tokenizer.build_vocab(captions)

    # Load models and measure loading time
    print("\nLoading models...")
    load_start = time.time()

    text_encoder = TextEncoder(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=config['text_encoder']['d_model'],
        n_heads=config['text_encoder']['n_heads'],
        n_layers=config['text_encoder']['n_layers'],
        d_ff=config['text_encoder']['d_ff'],
        max_seq_len=config['text_encoder']['max_seq_len'],
        dropout=0.0
    ).to(device)

    model = ConditionalUNet(
        in_channels=config['model']['in_channels'],
        model_channels=config['model']['model_channels'],
        channel_mult=config['model']['channel_mult'],
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=config['model']['attention_resolutions'],
        dropout=0.0,
        num_timesteps=config['diffusion']['num_timesteps'],
        text_dim=config['text_encoder']['d_model'],
        n_heads=config.get('n_heads', 4),
        use_cross_attn=config['model']['use_cross_attn']
    ).to(device)

    checkpoint_path = "checkpoints_conditional/checkpoint_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])

    text_encoder.eval()
    model.eval()

    load_time = time.time() - load_start
    print(f"Model loading time: {load_time:.2f}s")

    # Create diffusion sampler
    diffusion = DiffusionSampler(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule=config['diffusion']['beta_schedule']
    )

    # Prepare text embeddings
    prompt = "A small red circle on a blue background."
    max_len = config['text_encoder']['max_seq_len']
    token_ids, attention_mask = tokenizer.encode(prompt, max_length=max_len)
    token_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device).float()

    with torch.no_grad():
        text_emb, _ = text_encoder(token_ids, attention_mask)

    # Benchmark configurations
    configs = [
        {'method': 'DDIM', 'steps': 25, 'repetitions': 5},
        {'method': 'DDIM', 'steps': 50, 'repetitions': 5},
        {'method': 'DDIM', 'steps': 100, 'repetitions': 3},
        {'method': 'DDIM', 'steps': 200, 'repetitions': 3},
        {'method': 'DDPM', 'steps': 1000, 'repetitions': 1},  # Only 1 due to slow speed
    ]

    batch_size = 1
    results = []

    for config_data in configs:
        method = config_data['method']
        steps = config_data['steps']
        repetitions = config_data['repetitions']

        print(f"\n{'='*70}")
        print(f"BENCHMARK: {method} {steps} steps")
        print(f"Repetitions: {repetitions}")
        print(f"{'='*70}")

        latencies = []
        memories = []
        cpu_utils = []

        for rep in range(repetitions):
            print(f"\nRepetition {rep + 1}/{repetitions}...")

            # Warm-up (except for DDPM which is too slow)
            if method == 'DDIM' and rep == 0:
                print("  Warm-up iteration...")
                with torch.no_grad():
                    _ = diffusion.sample(
                        model=model,
                        batch_size=batch_size,
                        channels=config['model']['in_channels'],
                        image_size=config['model']['image_size'],
                        num_sampling_steps=steps,
                        device=device,
                        clip_x0=False,
                        text_emb=text_emb,
                        text_mask=attention_mask
                    )

            # Measure
            mem_before = psutil.Process().memory_info().rss / (1024**2)
            cpu_before = psutil.cpu_percent(interval=0.1)

            start = time.time()
            with torch.no_grad():
                if method == 'DDPM':
                    samples = diffusion.sample_ddpm(
                        model=model,
                        batch_size=batch_size,
                        channels=config['model']['in_channels'],
                        image_size=config['model']['image_size'],
                        device=device,
                        text_emb=text_emb,
                        text_mask=attention_mask
                    )
                else:  # DDIM
                    samples = diffusion.sample(
                        model=model,
                        batch_size=batch_size,
                        channels=config['model']['in_channels'],
                        image_size=config['model']['image_size'],
                        num_sampling_steps=steps,
                        device=device,
                        clip_x0=False,
                        text_emb=text_emb,
                        text_mask=attention_mask
                    )
            elapsed = time.time() - start

            mem_after = psutil.Process().memory_info().rss / (1024**2)
            cpu_after = psutil.cpu_percent(interval=0.1)

            mem_used = mem_after - mem_before
            cpu_used = (cpu_before + cpu_after) / 2

            latencies.append(elapsed)
            memories.append(mem_used)
            cpu_utils.append(cpu_used)

            print(f"  Time: {elapsed:.2f}s")
            print(f"  Memory: {mem_used:.2f} MB")
            print(f"  CPU: {cpu_used:.1f}%")

        # Calculate statistics
        if latencies:
            median_latency = statistics.median(latencies)
            p90_latency = statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else max(latencies)
            avg_latency = statistics.mean(latencies)
            avg_memory = statistics.mean(memories)
            avg_cpu = statistics.mean(cpu_utils)

            time_per_image = avg_latency / batch_size
            time_per_step = avg_latency / steps
            images_per_sec = batch_size / avg_latency

            result = {
                'method': method,
                'steps': steps,
                'repetitions': repetitions,
                'batch_size': batch_size,
                'total_latency_avg_s': avg_latency,
                'total_latency_median_s': median_latency,
                'total_latency_p90_s': p90_latency,
                'time_per_image_s': time_per_image,
                'time_per_step_s': time_per_step,
                'images_per_sec': images_per_sec,
                'peak_memory_mb': max(memories),
                'avg_memory_mb': avg_memory,
                'avg_cpu_percent': avg_cpu,
                'model_load_time_s': load_time,
                'num_threads': num_threads,
                'timestamp': datetime.now().isoformat()
            }

            results.append(result)

            print(f"\nStatistics:")
            print(f"  Median latency: {median_latency:.2f}s")
            print(f"  90th percentile: {p90_latency:.2f}s")
            print(f"  Time per image: {time_per_image:.2f}s")
            print(f"  Time per step: {time_per_step*1000:.1f}ms")
            print(f"  Images/sec: {images_per_sec:.3f}")
            print(f"  Peak memory: {max(memories):.2f} MB")

    # Save results to CSV
    os.makedirs("outputs_conditional", exist_ok=True)
    csv_path = "outputs_conditional/cpu_benchmark_results.csv"

    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"Results saved to {csv_path}")
    print(f"{'='*70}")

    # Print summary table
    print("\nSUMMARY TABLE")
    print("="*70)
    print(f"{'Method':<8} {'Steps':<8} {'Latency(s)':<12} {'Time/Img(s)':<12} {'Img/sec':<12} {'RAM(MB)':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['method']:<8} {r['steps']:<8} {r['total_latency_median_s']:<12.2f} "
              f"{r['time_per_image_s']:<12.2f} {r['images_per_sec']:<12.3f} {r['peak_memory_mb']:<10.2f}")
    print("="*70)

    return results


if __name__ == "__main__":
    results = run_benchmark()
