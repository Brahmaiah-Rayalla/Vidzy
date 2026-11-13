"""
Simple CPU thread benchmark using forward pass only.
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
from video_foundation.utils.reproducibility import get_device

print("="*70)
print("CPU THREAD OPTIMIZATION BENCHMARK (FORWARD PASS)")
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

# Prepare model inputs
x = torch.randn(1, 3, 64, 64, device=device)
t = torch.tensor([999], device=device)

print(f"\nInput shape: {x.shape}")
print(f"Text embedding shape: {text_emb.shape}")

# Test different thread configurations
thread_configs = [1, 2, 4, 8]
available_threads = psutil.cpu_count(logical=True)
thread_configs = [t for t in thread_configs if t <= available_threads]

if not thread_configs:
    thread_configs = [available_threads]

print(f"\nTesting thread configurations: {thread_configs}")
print("Note: Setting interop threads once at startup (cannot change after)")

# Set interop threads once (maximum across all configs)
max_threads = max(thread_configs)
torch.set_num_interop_threads(max(1, max_threads // 2))

results = []

for num_threads in thread_configs:
    # Configure threads (only num_threads, not interop)
    torch.set_num_threads(num_threads)
    
    print(f"\n{'='*70}")
    print(f"Configuration: {num_threads} threads")
    print(f"{'='*70}")
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x, t, text_emb, attention_mask)
    
    # Benchmark forward pass
    times = []
    
    for i in range(50):
        start = time.time()
        with torch.no_grad():
            _ = model(x, t, text_emb, attention_mask)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
    
    result = {
        'num_threads': num_threads,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
    }
    
    results.append(result)
    
    print(f"Average time: {avg_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
    print(f"Min/Max: {min(times)*1000:.2f} ms / {max(times)*1000:.2f} ms")

# Find optimal configuration
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

for r in results:
    print(f"Threads: {r['num_threads']:2d} | Avg: {r['avg_time_ms']:6.2f} ms | Min: {r['min_time_ms']:6.2f} ms | Max: {r['max_time_ms']:6.2f} ms")

optimal = min(results, key=lambda x: x['avg_time_ms'])
print(f"\nOptimal configuration: {optimal['num_threads']} threads")
print(f"Speedup vs 1 thread: {results[0]['avg_time_ms'] / optimal['avg_time_ms']:.2f}x")

# Estimate DDIM performance
print(f"\n{'='*70}")
print("DDIM PERFORMANCE ESTIMATES")
print(f"{'='*70}")

for r in results:
    ddim_50_est = r['avg_time_ms'] * 50 / 1000  # seconds
    ddim_100_est = r['avg_time_ms'] * 100 / 1000
    print(f"Threads: {r['num_threads']:2d} | DDIM 50: {ddim_50_est:5.2f}s | DDIM 100: {ddim_100_est:5.2f}s")

# Save results
os.makedirs('outputs_conditional', exist_ok=True)
with open('outputs_conditional/thread_benchmark_results.json', 'w') as f:
    json.dump({
        'timestamp': time.time(),
        'cpu_cores_physical': psutil.cpu_count(logical=False),
        'cpu_cores_logical': psutil.cpu_count(logical=True),
        'results': results,
        'optimal_threads': optimal['num_threads'],
        'speedup_vs_1_thread': results[0]['avg_time_ms'] / optimal['avg_time_ms'],
    }, f, indent=2)

print(f"\nResults saved to: outputs_conditional/thread_benchmark_results.json")
