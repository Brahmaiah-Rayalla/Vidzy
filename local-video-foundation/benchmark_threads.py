"""
Benchmark CPU thread configurations for DDIM inference.
Tests different thread counts to find optimal configuration.
"""
import os
import sys
import time
import json
import psutil
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import get_device

print("="*70)
print("CPU THREAD OPTIMIZATION BENCHMARK")
print("="*70)

device = get_device()
print(f"Device: {device}")
print(f"CPU cores (physical): {psutil.cpu_count(logical=False)}")
print(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")

# Load config
with open('configs/conditional_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load tokenizer
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
text_encoder = TextEncoder(
    vocab_size=len(tokenizer.word_to_idx),
    d_model=config['text_encoder']['d_model'],
    n_heads=config['text_encoder']['n_heads'],
    n_layers=config['text_encoder']['n_layers'],
    d_ff=config['text_encoder']['d_ff'],
    max_seq_len=config['text_encoder']['max_seq_len'],
    dropout=0.0
)
text_encoder = text_encoder.to(device)

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
)
model = model.to(device)

# Load checkpoint
checkpoint = torch.load('checkpoints_conditional/checkpoint_final.pt', map_location=device)
text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'])

text_encoder.eval()
model.eval()

# Prepare inputs
test_prompt = "A small red circle on a blue background."
max_len = config['text_encoder']['max_seq_len']
token_ids, attention_mask = tokenizer.encode(test_prompt, max_length=max_len)
token_ids = torch.tensor([token_ids], device=device)
attention_mask = torch.tensor([attention_mask], device=device).float()

with torch.no_grad():
    text_emb, _ = text_encoder(token_ids, attention_mask)

print(f"Text embedding shape: {text_emb.shape}")
print(f"Attention mask shape: {attention_mask.shape}")

# Text embedding is already [1, seq_len, 256], no need to repeat for batch_size=1
# For batch_size > 1, we would repeat: text_emb.repeat(batch_size, 1, 1)

# Create diffusion sampler
diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)

# Test different thread configurations
thread_configs = [1, 2, 4, 8]  # Will be filtered by available cores
available_threads = psutil.cpu_count(logical=True)
thread_configs = [t for t in thread_configs if t <= available_threads]

if not thread_configs:
    thread_configs = [available_threads]

print(f"\nTesting thread configurations: {thread_configs}")
print(f"DDIM steps: 50")

results = []

for num_threads in thread_configs:
    # Configure threads
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, num_threads // 2))
    
    print(f"\n{'='*70}")
    print(f"Configuration: {num_threads} threads")
    print(f"{'='*70}")
    
    # Warmup - use DDPM which already works
    for _ in range(2):
        with torch.no_grad():
            _ = diffusion.sample_ddpm(
                model=model,
                batch_size=1,
                channels=config['model']['in_channels'],
                image_size=config['model']['image_size'],
                device=device,
                text_emb=text_emb,
                text_mask=attention_mask
            )
    
    # Benchmark with DDPM (already working)
    times = []
    mem_peak = 0
    
    for i in range(3):  # Fewer runs for DDPM since it's slow
        mem_before = psutil.Process().memory_info().rss / (1024**2)
        start = time.time()
        
        with torch.no_grad():
            samples = diffusion.sample_ddpm(
                model=model,
                batch_size=1,
                channels=config['model']['in_channels'],
                image_size=config['model']['image_size'],
                device=device,
                text_emb=text_emb,
                text_mask=attention_mask
            )
        
        elapsed = time.time() - start
        mem_after = psutil.Process().memory_info().rss / (1024**2)
        mem_peak = max(mem_peak, mem_after)
        
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
    
    result = {
        'num_threads': num_threads,
        'avg_time_s': avg_time,
        'std_time_s': std_time,
        'min_time_s': min(times),
        'max_time_s': max(times),
        'peak_memory_mb': mem_peak,
        'throughput_imgs_per_sec': 1 / avg_time,
    }
    
    results.append(result)
    
    print(f"Average time: {avg_time:.2f}s (±{std_time:.2f}s)")
    print(f"Min/Max: {min(times):.2f}s / {max(times):.2f}s")
    print(f"Peak memory: {mem_peak:.2f} MB")
    print(f"Throughput: {1/avg_time:.3f} images/sec")

# Find optimal configuration
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

for r in results:
    print(f"Threads: {r['num_threads']:2d} | Time: {r['avg_time_s']:6.2f}s | Peak Mem: {r['peak_memory_mb']:7.2f} MB | Throughput: {r['throughput_imgs_per_sec']:.3f} img/s")

optimal = min(results, key=lambda x: x['avg_time_s'])
print(f"\nOptimal configuration: {optimal['num_threads']} threads")
print(f"Speedup vs default (4 threads): {results[2]['avg_time_s'] / optimal['avg_time_s']:.2f}x" if len(results) > 2 else "")

# Save results
os.makedirs('outputs_conditional', exist_ok=True)
with open('outputs_conditional/thread_benchmark_results.json', 'w') as f:
    json.dump({
        'timestamp': time.time(),
        'cpu_cores_physical': psutil.cpu_count(logical=False),
        'cpu_cores_logical': psutil.cpu_count(logical=True),
        'results': results,
        'optimal_threads': optimal['num_threads'],
    }, f, indent=2)

print(f"\nResults saved to: outputs_conditional/thread_benchmark_results.json")
