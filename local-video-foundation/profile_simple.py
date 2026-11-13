"""
Simple forward pass profiling to identify bottlenecks.
"""
import os
import sys
import time
import torch
import yaml
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet

print("="*70)
print("SIMPLE MODEL PROFILING")
print("="*70)

device = torch.device('cpu')

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

# Prepare diffusion inputs
x = torch.randn(1, 3, 64, 64, device=device)
t = torch.tensor([999], device=device)  # High timestep

print(f"\nInput shape: {x.shape}")
print(f"Text embedding shape: {text_emb.shape}")
print(f"Timestep: {t.item()}")

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = model(x, t, text_emb, attention_mask)

# Benchmark forward pass
forward_times = []
for i in range(20):
    start = time.time()
    with torch.no_grad():
        output = model(x, t, text_emb, attention_mask)
    forward_times.append(time.time() - start)

avg_forward = sum(forward_times) / len(forward_times)
std_forward = (sum((t - avg_forward)**2 for t in forward_times) / len(forward_times))**0.5

print(f"\nForward pass (20 runs):")
print(f"  Average: {avg_forward*1000:.2f} ms")
print(f"  Std dev: {std_forward*1000:.2f} ms")
print(f"  Min: {min(forward_times)*1000:.2f} ms")
print(f"  Max: {max(forward_times)*1000:.2f} ms")
print(f"  Output shape: {output.shape}")

# Test at different timesteps
print("\nForward pass at different timesteps:")
for timestep in [0, 250, 500, 750, 999]:
    t_test = torch.tensor([timestep], device=device)
    times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            _ = model(x, t_test, text_emb, attention_mask)
        times.append(time.time() - start)
    avg = sum(times) / len(times)
    print(f"  Timestep {timestep}: {avg*1000:.2f} ms")

# Estimate DDIM speedup
print("\n" + "="*70)
print("DDIM SPEEDUP ESTIMATE")
print("="*70)
ddpm_steps = 1000
ddim_steps_options = [50, 100, 200]

for ddim_steps in ddim_steps_options:
    estimated_time = avg_forward * ddim_steps
    speedup = ddpm_steps / ddim_steps
    print(f"DDIM {ddim_steps} steps: {estimated_time:.2f}s (speedup: {speedup:.1f}x)")

print(f"\nCurrent DDPM (1000 steps): {avg_forward * 1000:.2f}s")
print(f"DDIM (50 steps): {avg_forward * 50:.2f}s")
print(f"Speedup: {1000/50:.1f}x")
