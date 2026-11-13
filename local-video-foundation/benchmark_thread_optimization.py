"""
CPU Thread Optimization Benchmark for Stage 2.3.2

Benchmarks different thread configurations to find optimal setting.
Tests 1, 2, 4, and 8 threads where supported.
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
    # Note: set_num_interop_threads can only be called once at startup
    # We skip it here to avoid the error


def run_thread_benchmark():
    # Load config
    with open('configs/conditional_v0.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    set_seed(42)

    # Set interop threads once at startup
    torch.set_num_interop_threads(4)

    print("="*70)
    print("CPU THREAD OPTIMIZATION BENCHMARK - STAGE 2.3.2")
    print("="*70)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU: {psutil.cpu_count(logical=True)} physical, {psutil.cpu_count()} logical")
    print("="*70)

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

    # Load models
    print("\nLoading models...")
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

    checkpoint = torch.load("checkpoints_conditional/checkpoint_final.pt", map_location=device)
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    text_encoder.eval()
    model.eval()

    # Prepare text embeddings
    prompt = "A small red circle on a blue background."
    max_len = config['text_encoder']['max_seq_len']
    token_ids, attention_mask = tokenizer.encode(prompt, max_length=max_len)
    token_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device).float()

    with torch.no_grad():
        text_emb, _ = text_encoder(token_ids, attention_mask)

    # Test thread configurations
    thread_configs = [1, 2, 4, 8]
    ddim_steps = 50  # Use 50 steps for thread benchmark
    repetitions = 3

    results = []

    for num_threads in thread_configs:
        print(f"\n{'='*70}")
        print(f"THREAD CONFIGURATION: {num_threads} threads")
        print(f"{'='*70}")

        # Configure threads
        optimize_cpu_threads(num_threads)
        print(f"Configured: {num_threads} main, {max(1, num_threads // 2)} interop")

        # Create new sampler for each thread config
        diffusion = DiffusionSampler(
            num_timesteps=config['diffusion']['num_timesteps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            schedule=config['diffusion']['beta_schedule']
        )

        latencies = []
        cpu_utils = []

        for rep in range(repetitions):
            print(f"\nRepetition {rep + 1}/{repetitions}...")

            # Warm-up
            if rep == 0:
                print("  Warm-up iteration...")
                with torch.no_grad():
                    _ = diffusion.sample(
                        model=model,
                        batch_size=1,
                        channels=config['model']['in_channels'],
                        image_size=config['model']['image_size'],
                        num_sampling_steps=ddim_steps,
                        device=device,
                        clip_x0=False,
                        text_emb=text_emb,
                        text_mask=attention_mask
                    )

            # Measure
            cpu_before = psutil.cpu_percent(interval=0.1)
            start = time.time()

            with torch.no_grad():
                samples = diffusion.sample(
                    model=model,
                    batch_size=1,
                    channels=config['model']['in_channels'],
                    image_size=config['model']['image_size'],
                    num_sampling_steps=ddim_steps,
                    device=device,
                    clip_x0=False,
                    text_emb=text_emb,
                    text_mask=attention_mask
                )

            elapsed = time.time() - start
            cpu_after = psutil.cpu_percent(interval=0.1)

            latencies.append(elapsed)
            cpu_utils.append((cpu_before + cpu_after) / 2)

            print(f"  Time: {elapsed:.2f}s")
            print(f"  CPU: {cpu_utils[-1]:.1f}%")

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        avg_cpu = statistics.mean(cpu_utils)

        result = {
            'num_threads': num_threads,
            'ddim_steps': ddim_steps,
            'repetitions': repetitions,
            'avg_latency_s': avg_latency,
            'median_latency_s': median_latency,
            'time_per_step_ms': (avg_latency / ddim_steps) * 1000,
            'avg_cpu_percent': avg_cpu,
            'throughput_img_per_sec': 1 / avg_latency,
            'timestamp': datetime.now().isoformat()
        }

        results.append(result)

        print(f"\nStatistics:")
        print(f"  Average latency: {avg_latency:.2f}s")
        print(f"  Median latency: {median_latency:.2f}s")
        print(f"  Time per step: {(avg_latency / ddim_steps) * 1000:.1f}ms")
        print(f"  Average CPU: {avg_cpu:.1f}%")
        print(f"  Throughput: {1 / avg_latency:.3f} img/sec")

    # Save results
    os.makedirs("outputs_conditional", exist_ok=True)
    csv_path = "outputs_conditional/thread_optimization_results.csv"

    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"Results saved to {csv_path}")
    print(f"{'='*70}")

    # Print summary
    print("\nSUMMARY TABLE")
    print("="*70)
    print(f"{'Threads':<10} {'Latency(s)':<12} {'Time/Step(ms)':<15} {'CPU(%)':<10} {'Img/sec':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['num_threads']:<10} {r['median_latency_s']:<12.2f} "
              f"{r['time_per_step_ms']:<15.1f} {r['avg_cpu_percent']:<10.1f} {r['throughput_img_per_sec']:<12.3f}")
    print("="*70)

    # Find optimal configuration
    best_result = min(results, key=lambda x: x['median_latency_s'])
    print(f"\nOPTIMAL CONFIGURATION: {best_result['num_threads']} threads")
    print(f"  Median latency: {best_result['median_latency_s']:.2f}s")
    print(f"  Throughput: {best_result['throughput_img_per_sec']:.3f} img/sec")

    return results


if __name__ == "__main__":
    results = run_thread_benchmark()
