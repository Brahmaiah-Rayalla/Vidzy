"""
Smoke test for conditional text-to-image pipeline.

Tests:
1. Dataset generation
2. Tokenizer and text encoder
3. Conditional U-Net forward pass
4. Training step (no optimization)
"""

import torch
import yaml
from video_foundation.data.text_image_dataset import TextImageShapeGenerator
from video_foundation.data.text_image_loader import TextImageDataset
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device

print("="*70)
print("STAGE 2 SMOKE TEST")
print("="*70)

# Load config
with open('configs/conditional_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = get_device()
set_seed(42)

# Test 1: Dataset generation
print("\nTest 1: Dataset generation")
print("-" * 40)
generator = TextImageShapeGenerator(
    image_size=config['dataset']['image_size'],
    seed=config['dataset']['seed'],
    max_objects=config['dataset']['max_objects']
)
metadata = generator.generate_dataset(
    num_images=10,
    output_dir='data/smoke_test',
    save_preview=False
)
print(f"Generated {len(metadata)} examples")
print(f"Sample caption: {metadata[0]['caption']}")

# Test 2: Tokenizer and text encoder
print("\nTest 2: Tokenizer and text encoder")
print("-" * 40)
captions = [m['caption'] for m in metadata]
tokenizer = SimpleTokenizer(
    vocab_size=config['text_encoder']['vocab_size'],
    min_freq=config['text_encoder']['min_freq']
)
tokenizer.build_vocab(captions)

text_encoder = TextEncoder(
    vocab_size=len(tokenizer.word_to_idx),
    d_model=config['text_encoder']['d_model'],
    n_heads=config['text_encoder']['n_heads'],
    n_layers=config['text_encoder']['n_layers'],
    d_ff=config['text_encoder']['d_ff'],
    max_seq_len=config['text_encoder']['max_seq_len'],
    dropout=config['text_encoder']['dropout']
)
text_encoder = text_encoder.to(device)

# Test encoding
caption = metadata[0]['caption']
token_ids, attention_mask = tokenizer.encode(caption, max_length=config['text_encoder']['max_seq_len'])
token_ids = torch.tensor([token_ids], device=device)
attention_mask = torch.tensor([attention_mask], device=device).float()

with torch.no_grad():
    text_emb, mask = text_encoder(token_ids, attention_mask)

print(f"Caption: {caption}")
print(f"Token IDs shape: {token_ids.shape}")
print(f"Text embedding shape: {text_emb.shape}")
print(f"Text embedding mean: {text_emb.mean():.4f}, std: {text_emb.std():.4f}")

# Test 3: Conditional U-Net forward pass
print("\nTest 3: Conditional U-Net forward pass")
print("-" * 40)
model = ConditionalUNet(
    in_channels=config['model']['in_channels'],
    model_channels=config['model']['model_channels'],
    channel_mult=config['model']['channel_mult'],
    num_res_blocks=config['model']['num_res_blocks'],
    attention_resolutions=config['model']['attention_resolutions'],
    dropout=config['model']['dropout'],
    num_timesteps=config['diffusion']['num_timesteps'],
    text_dim=config['text_encoder']['d_model'],
    n_heads=config['n_heads'],
    use_cross_attn=config['model']['use_cross_attn']
)
model = model.to(device)

# Create dummy inputs
batch_size = 2
x = torch.randn(batch_size, 3, 64, 64, device=device)
timesteps = torch.randint(0, 1000, (batch_size,), device=device)
text_emb_batch = text_emb.repeat(batch_size, 1, 1)
attention_mask_batch = attention_mask.repeat(batch_size, 1)

with torch.no_grad():
    predicted_noise = model(x, timesteps, text_emb_batch, attention_mask_batch)

print(f"Input shape: {x.shape}")
print(f"Predicted noise shape: {predicted_noise.shape}")
print(f"Predicted noise mean: {predicted_noise.mean():.4f}, std: {predicted_noise.std():.4f}")

# Test 4: Training step
print("\nTest 4: Training step")
print("-" * 40)
diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)

# Create dummy batch
x_0 = torch.randn(batch_size, 3, 64, 64, device=device)
timesteps = torch.randint(0, 1000, (batch_size,), device=device)

# Forward diffusion
noise = torch.randn_like(x_0)
x_t, noise = diffusion.process.q_sample(x_0, timesteps, noise)

# Predict noise
predicted_noise = model(x_t, timesteps, text_emb_batch, attention_mask_batch)

# Compute loss
loss = torch.nn.functional.mse_loss(predicted_noise, noise)

print(f"Loss: {loss.item():.6f}")

# Test gradient flow
loss.backward()
print(f"Gradient flow: OK (loss backward completed)")

print("\n" + "="*70)
print("SMOKE TEST COMPLETE - ALL TESTS PASSED")
print("="*70)
