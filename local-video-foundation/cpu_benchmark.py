"""
Comprehensive CPU benchmarking for Stage 2.1 text-to-image model.
Measures system specs, memory usage, latency, and identifies bottlenecks.
"""
import os
import sys
import time
import json
import psutil
import platform
import torch
import yaml
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import get_device

print("="*70)
print("CPU-FIRST ARCHITECTURE BENCHMARK - STAGE 2.1")
print("="*70)

# ============================================================================
# 1. SYSTEM INFORMATION
# ============================================================================
print("\n" + "="*70)
print("1. SYSTEM INFORMATION")
print("="*70)

system_info = {
    'platform': platform.system(),
    'platform_release': platform.release(),
    'platform_version': platform.version(),
    'architecture': platform.machine(),
    'processor': platform.processor(),
    'python_version': platform.python_version(),
    'pytorch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    'cpu_count_physical': psutil.cpu_count(logical=False),
    'cpu_count_logical': psutil.cpu_count(logical=True),
    'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
    'ram_total_gb': psutil.virtual_memory().total / (1024**3),
    'ram_available_gb': psutil.virtual_memory().available / (1024**3),
}

print(f"Platform: {system_info['platform']} {system_info['platform_release']}")
print(f"Architecture: {system_info['architecture']}")
print(f"Python: {system_info['python_version']}")
print(f"PyTorch: {system_info['pytorch_version']}")
print(f"CUDA Available: {system_info['cuda_available']}")
print(f"CPU Cores (Physical): {system_info['cpu_count_physical']}")
print(f"CPU Cores (Logical): {system_info['cpu_count_logical']}")
print(f"CPU Frequency: {system_info['cpu_freq']:.0f} MHz" if system_info['cpu_freq'] else "CPU Frequency: N/A")
print(f"RAM Total: {system_info['ram_total_gb']:.2f} GB")
print(f"RAM Available: {system_info['ram_available_gb']:.2f} GB")

# PyTorch CPU threads
print(f"PyTorch CPU threads: {torch.get_num_threads()}")
print(f"PyTorch interop threads: {torch.get_num_interop_threads()}")

# ============================================================================
# 2. MODEL LOADING BENCHMARK
# ============================================================================
print("\n" + "="*70)
print("2. MODEL LOADING BENCHMARK")
print("="*70)

device = torch.device('cpu')  # Force CPU
print(f"Device: {device}")

# Load config
with open('configs/conditional_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Measure model loading time and memory
import json
train_metadata_path = "data/text_image/train_metadata.json"
with open(train_metadata_path, 'r') as f:
    train_metadata = json.load(f)

captions = [m['caption'] for m in train_metadata]
tokenizer = SimpleTokenizer(
    vocab_size=config['text_encoder']['vocab_size'],
    min_freq=config['text_encoder']['min_freq']
)
tokenizer.build_vocab(captions)

# Load models and measure memory
mem_before = psutil.Process().memory_info().rss / (1024**2)  # MB

load_start = time.time()
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
load_time = time.time() - load_start

mem_after = psutil.Process().memory_info().rss / (1024**2)  # MB
mem_used = mem_after - mem_before

# Count parameters
text_encoder_params = sum(p.numel() for p in text_encoder.parameters())
model_params = sum(p.numel() for p in model.parameters())
total_params = text_encoder_params + model_params

print(f"Text encoder parameters: {text_encoder_params:,}")
print(f"U-Net parameters: {model_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Model loading time: {load_time:.3f}s")
print(f"Memory after model load: {mem_after:.2f} MB")
print(f"Memory used by models: {mem_used:.2f} MB")

# Load checkpoint
checkpoint_path = "checkpoints_conditional/checkpoint_final.pt"
checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024**2)
print(f"Checkpoint size: {checkpoint_size_mb:.2f} MB")

checkpoint_load_start = time.time()
checkpoint = torch.load(checkpoint_path, map_location=device)
text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'])
checkpoint_load_time = time.time() - checkpoint_load_start

print(f"Checkpoint loading time: {checkpoint_load_time:.3f}s")

mem_after_checkpoint = psutil.Process().memory_info().rss / (1024**2)
print(f"Memory after checkpoint load: {mem_after_checkpoint:.2f} MB")

model_info = {
    'text_encoder_params': text_encoder_params,
    'unet_params': model_params,
    'total_params': total_params,
    'checkpoint_size_mb': checkpoint_size_mb,
    'model_load_time_s': load_time,
    'checkpoint_load_time_s': checkpoint_load_time,
    'memory_after_load_mb': mem_after,
    'memory_after_checkpoint_mb': mem_after_checkpoint,
}

# ============================================================================
# 3. INFERENCE BENCHMARK - TEXT ENCODING
# ============================================================================
print("\n" + "="*70)
print("3. INFERENCE BENCHMARK - TEXT ENCODING")
print("="*70)

text_encoder.eval()
model.eval()

test_prompt = "A small red circle on a blue background."
max_len = config['text_encoder']['max_seq_len']

# Warmup
for _ in range(3):
    token_ids, attention_mask = tokenizer.encode(test_prompt, max_length=max_len)
    token_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device).float()
    with torch.no_grad():
        text_emb, _ = text_encoder(token_ids, attention_mask)

# Benchmark
mem_before_encode = psutil.Process().memory_info().rss / (1024**2)
encode_times = []

for i in range(10):
    start = time.time()
    token_ids, attention_mask = tokenizer.encode(test_prompt, max_length=max_len)
    token_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device).float()
    with torch.no_grad():
        text_emb, _ = text_encoder(token_ids, attention_mask)
    encode_times.append(time.time() - start)

mem_after_encode = psutil.Process().memory_info().rss / (1024**2)

avg_encode_time = np.mean(encode_times)
std_encode_time = np.std(encode_times)
mem_encode_used = mem_after_encode - mem_before_encode

print(f"Text encoding (10 runs):")
print(f"  Average time: {avg_encode_time*1000:.2f} ms")
print(f"  Std dev: {std_encode_time*1000:.2f} ms")
print(f"  Memory used: {mem_encode_used:.2f} MB")
print(f"  Text embedding shape: {text_emb.shape}")

text_encoding_info = {
    'avg_time_ms': avg_encode_time * 1000,
    'std_time_ms': std_encode_time * 1000,
    'memory_used_mb': mem_encode_used,
    'embedding_shape': list(text_emb.shape),
}

# ============================================================================
# 4. INFERENCE BENCHMARK - SAMPLING
# ============================================================================
print("\n" + "="*70)
print("4. INFERENCE BENCHMARK - DDPM SAMPLING")
print("="*70)

diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)

# Warmup
for _ in range(2):
    with torch.no_grad():
        diffusion.sample_ddpm(
            model=model,
            batch_size=1,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            device=device,
            text_emb=text_emb,
            text_mask=attention_mask
        )

# Benchmark with memory tracking
mem_before_sample = psutil.Process().memory_info().rss / (1024**2)
sample_times = []
step_times = []
peak_memory = 0

for i in range(3):  # Run 3 times for stability
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
    sample_time = time.time() - start
    sample_times.append(sample_time)

    # Check peak memory
    mem_current = psutil.Process().memory_info().rss / (1024**2)
    peak_memory = max(peak_memory, mem_current)

mem_after_sample = psutil.Process().memory_info().rss / (1024**2)

avg_sample_time = np.mean(sample_times)
std_sample_time = np.std(sample_times)
mem_sample_used = mem_after_sample - mem_before_sample
estimated_step_time = avg_sample_time / config['diffusion']['num_timesteps']

print(f"DDPM Sampling (3 runs, {config['diffusion']['num_timesteps']} steps):")
print(f"  Average total time: {avg_sample_time:.2f}s")
print(f"  Std dev: {std_sample_time:.2f}s")
print(f"  Estimated time per step: {estimated_step_time*1000:.2f} ms")
print(f"  Memory used: {mem_sample_used:.2f} MB")
print(f"  Peak memory: {peak_memory:.2f} MB")
print(f"  Output shape: {samples.shape}")
print(f"  Output range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")

sampling_info = {
    'num_timesteps': config['diffusion']['num_timesteps'],
    'avg_total_time_s': avg_sample_time,
    'std_total_time_s': std_sample_time,
    'avg_step_time_ms': estimated_step_time * 1000,
    'memory_used_mb': mem_sample_used,
    'peak_memory_mb': peak_memory,
    'output_shape': list(samples.shape),
}

# ============================================================================
# 5. END-TO-END INFERENCE BENCHMARK
# ============================================================================
print("\n" + "="*70)
print("5. END-TO-END INFERENCE BENCHMARK")
print("="*70)

e2e_times = []

for i in range(5):
    start = time.time()

    # Tokenize
    token_ids, attention_mask = tokenizer.encode(test_prompt, max_length=max_len)
    token_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device).float()

    # Encode
    with torch.no_grad():
        text_emb, _ = text_encoder(token_ids, attention_mask)

    # Sample
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

    e2e_time = time.time() - start
    e2e_times.append(e2e_time)

avg_e2e_time = np.mean(e2e_times)
std_e2e_time = np.std(e2e_times)

print(f"End-to-end inference (5 runs):")
print(f"  Average time: {avg_e2e_time:.2f}s")
print(f"  Std dev: {std_e2e_time:.2f}s")
print(f"  Throughput: {1/avg_e2e_time:.3f} images/second")

e2e_info = {
    'avg_time_s': avg_e2e_time,
    'std_time_s': std_e2e_time,
    'throughput_imgs_per_sec': 1 / avg_e2e_time,
}

# ============================================================================
# 6. ARCHITECTURE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("6. ARCHITECTURE ANALYSIS")
print("="*70)

# Analyze model components
print("\nModel Architecture Breakdown:")
print(f"  Base channels: {config['model']['model_channels']}")
print(f"  Channel multipliers: {config['model']['channel_mult']}")
print(f"  Resolution levels: {len(config['model']['channel_mult'])}")
print(f"  Residual blocks per level: {config['model']['num_res_blocks']}")
print(f"  Cross-attention: {config['model']['use_cross_attn']}")
print(f"  Attention heads: {config.get('n_heads', 4)}")

print("\nText Encoder Architecture:")
print(f"  Embedding dim: {config['text_encoder']['d_model']}")
print(f"  Attention heads: {config['text_encoder']['n_heads']}")
print(f"  Transformer layers: {config['text_encoder']['n_layers']}")
print(f"  FFN dim: {config['text_encoder']['d_ff']}")
print(f"  Max sequence length: {config['text_encoder']['max_seq_len']}")

# Calculate FLOPs estimate (rough)
# U-Net: ~64 base channels, 4 levels, 2 blocks each
# Cross-attention at each level
# Text encoder: 4 transformer layers

print("\nComplexity Estimates:")
print(f"  Image size: {config['model']['image_size']}x{config['model']['image_size']}")
print(f"  Diffusion steps: {config['diffusion']['num_timesteps']}")
print(f"  Total model parameters: {total_params:,}")
print(f"  Params per 64x64 pixel: {total_params / (64*64):.1f}")

architecture_info = {
    'base_channels': config['model']['model_channels'],
    'channel_mult': config['model']['channel_mult'],
    'num_levels': len(config['model']['channel_mult']),
    'num_res_blocks': config['model']['num_res_blocks'],
    'cross_attention': config['model']['use_cross_attn'],
    'attention_heads': config.get('n_heads', 4),
    'text_encoder_d_model': config['text_encoder']['d_model'],
    'text_encoder_n_layers': config['text_encoder']['n_layers'],
    'text_encoder_d_ff': config['text_encoder']['d_ff'],
    'image_size': config['model']['image_size'],
    'diffusion_steps': config['diffusion']['num_timesteps'],
}

# ============================================================================
# 7. BOTTLENECK ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("7. BOTTLENECK ANALYSIS")
print("="*70)

print("\nTime Breakdown:")
print(f"  Text encoding: {avg_encode_time*1000:.2f} ms ({avg_encode_time/avg_e2e_time*100:.1f}%)")
print(f"  Diffusion sampling: {avg_sample_time*1000:.2f} ms ({avg_sample_time/avg_e2e_time*100:.1f}%)")
print(f"  Total: {avg_e2e_time*1000:.2f} ms")

print("\nMemory Breakdown:")
print(f"  Model + checkpoint: {mem_after_checkpoint:.2f} MB")
print(f"  Peak during sampling: {peak_memory:.2f} MB")
print(f"  Memory overhead: {peak_memory - mem_after_checkpoint:.2f} MB")

print("\nPotential Bottlenecks:")
if avg_sample_time / avg_e2e_time > 0.95:
    print("  [PRIMARY] Diffusion sampling dominates inference time")
    print("  -> Consider reducing timesteps or using DDIM")
if text_encoder_params / total_params > 0.2:
    print("  [SECONDARY] Text encoder is large relative to U-Net")
    print("  -> Consider smaller d_model or fewer layers")
if peak_memory > system_info['ram_available_gb'] * 1024 * 0.5:
    print("  [MEMORY] Peak memory uses >50% of available RAM")
    print("  -> Consider reducing batch size or model size")
if estimated_step_time * 1000 > 100:
    print("  [PERFORMANCE] Each diffusion step takes >100ms")
    print("  -> Consider model optimization or fewer steps")

bottleneck_info = {
    'text_encoding_pct': avg_encode_time / avg_e2e_time * 100,
    'sampling_pct': avg_sample_time / avg_e2e_time * 100,
    'text_encoder_params_pct': text_encoder_params / total_params * 100,
    'peak_memory_pct_of_ram': peak_memory / (system_info['ram_available_gb'] * 1024) * 100,
    'step_time_ms': estimated_step_time * 1000,
}

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("8. SAVING BENCHMARK RESULTS")
print("="*70)

results = {
    'timestamp': datetime.now().isoformat(),
    'system_info': system_info,
    'model_info': model_info,
    'text_encoding_info': text_encoding_info,
    'sampling_info': sampling_info,
    'e2e_info': e2e_info,
    'architecture_info': architecture_info,
    'bottleneck_info': bottleneck_info,
}

output_path = 'outputs_conditional/cpu_benchmark_results.json'
os.makedirs('outputs_conditional', exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {output_path}")

print("\n" + "="*70)
print("BENCHMARK COMPLETE")
print("="*70)
print("\nSummary:")
print(f"  Total parameters: {total_params:,}")
print(f"  End-to-end latency: {avg_e2e_time:.2f}s")
print(f"  Throughput: {1/avg_e2e_time:.3f} images/sec")
print(f"  Peak memory: {peak_memory:.2f} MB")
print(f"  Time per diffusion step: {estimated_step_time*1000:.2f} ms")
