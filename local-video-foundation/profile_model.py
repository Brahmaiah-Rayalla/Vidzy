"""
Profile the U-Net model to identify CPU bottlenecks.
Measures time per layer and operation during a single forward pass.
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
from video_foundation.utils.reproducibility import get_device

print("="*70)
print("MODEL PROFILING - IDENTIFYING CPU BOTTLENECKS")
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
t = torch.tensor([999], device=device)  # High timestep (most compute)
text_emb = text_emb
text_mask = attention_mask

print("\nProfiling single forward pass at timestep 999 (maximum complexity)...")
print(f"Input shape: {x.shape}")
print(f"Text embedding shape: {text_emb.shape}")

# Warmup
for _ in range(3):
    with torch.no_grad():
        _ = model(x, t, text_emb, text_mask)

# Profile forward pass
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with torch.no_grad():
        for _ in range(10):  # Run 10 times for stable measurement
            output = model(x, t, text_emb, text_mask)

print("\n" + "="*70)
print("TOP 20 CPU OPERATIONS BY TIME")
print("="*70)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

print("\n" + "="*70)
print("TOP 20 OPERATIONS BY CALL COUNT")
print("="*70)
print(prof.key_averages().table(sort_by="count", row_limit=20))

# Save profile
prof_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=50)
with open('outputs_conditional/model_profile.txt', 'w') as f:
    f.write(prof_table)
print("\nDetailed profile saved to: outputs_conditional/model_profile.txt")

# Test with fewer timesteps using DDIM approach
print("\n" + "="*70)
print("TESTING DDIM FAST SAMPLING (100 steps)")
print("="*70)

from video_foundation.diffusion.sampler import DiffusionSampler
diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)

# Warmup
for _ in range(2):
    with torch.no_grad():
        _ = diffusion.sample(
            model=model,
            batch_size=1,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            num_sampling_steps=50,
            device=device,
            clip_x0=True
        )

# Benchmark DDIM with 50 steps
import time
ddim_times = []

for i in range(3):
    start = time.time()
    with torch.no_grad():
        samples = diffusion.sample(
            model=model,
            batch_size=1,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            num_sampling_steps=50,
            device=device,
            clip_x0=True
        )
    ddim_times.append(time.time() - start)

avg_ddim_time = sum(ddim_times) / len(ddim_times)
print(f"DDIM 50 steps - Average time: {avg_ddim_time:.2f}s")
print(f"DDIM 50 steps - Time per step: {avg_ddim_time/50*1000:.2f} ms")
print(f"DDIM 50 steps - Throughput: {1/avg_ddim_time:.3f} images/sec")

# Benchmark DDIM with 100 steps
ddim_times_100 = []

for i in range(3):
    start = time.time()
    with torch.no_grad():
        samples = diffusion.sample(
            model=model,
            batch_size=1,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            num_sampling_steps=100,
            device=device,
            clip_x0=True
        )
    ddim_times_100.append(time.time() - start)

avg_ddim_time_100 = sum(ddim_times_100) / len(ddim_times_100)
print(f"DDIM 100 steps - Average time: {avg_ddim_time_100:.2f}s")
print(f"DDIM 100 steps - Time per step: {avg_ddim_time_100/100*1000:.2f} ms")
print(f"DDIM 100 steps - Throughput: {1/avg_ddim_time_100:.3f} images/sec")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"DDPM 1000 steps: {674.62}s (benchmark)")
print(f"DDIM 50 steps: {avg_ddim_time:.2f}s")
print(f"DDIM 100 steps: {avg_ddim_time_100:.2f}s")
print(f"\nSpeedup (DDIM 50 vs DDPM 1000): {674.62/avg_ddim_time:.1f}x")
print(f"Speedup (DDIM 100 vs DDPM 1000): {674.62/avg_ddim_time_100:.1f}x")
