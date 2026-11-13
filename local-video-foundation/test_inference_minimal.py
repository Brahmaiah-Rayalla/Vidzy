"""
Test inference with the minimal checkpoint.
"""

import os
import torch
import yaml
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device
from PIL import Image
from torchvision.utils import make_grid

print("="*70)
print("TESTING INFERENCE WITH MINIMAL CHECKPOINT")
print("="*70)

# Load config
with open('configs/conditional_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = get_device()
set_seed(42)

# Load checkpoint
print("\nLoading checkpoint...")
checkpoint = torch.load('checkpoints_conditional_minimal/checkpoint_minimal.pt', map_location=device)

# Build tokenizer from the training data
print("Building tokenizer...")
import json
with open('data/conditional_minimal/train_metadata.json', 'r') as f:
    train_metadata = json.load(f)

captions = [m['caption'] for m in train_metadata]
tokenizer = SimpleTokenizer(vocab_size=1000, min_freq=1)
tokenizer.build_vocab(captions)

# Load models
print("Loading models...")
text_encoder = TextEncoder(
    vocab_size=len(tokenizer.word_to_idx),
    d_model=config['text_encoder']['d_model'],
    n_heads=config['text_encoder']['n_heads'],
    n_layers=config['text_encoder']['n_layers'],
    d_ff=config['text_encoder']['d_ff'],
    max_seq_len=config['text_encoder']['max_seq_len'],
    dropout=0.0
)
text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
text_encoder = text_encoder.to(device)
text_encoder.eval()

model = ConditionalUNet(
    in_channels=config['model']['in_channels'],
    model_channels=config['model']['model_channels'],
    channel_mult=config['model']['channel_mult'],
    num_res_blocks=config['model']['num_res_blocks'],
    attention_resolutions=config['model']['attention_resolutions'],
    dropout=0.0,
    num_timesteps=config['diffusion']['num_timesteps'],
    text_dim=config['text_encoder']['d_model'],
    n_heads=config['n_heads'],
    use_cross_attn=config['model']['use_cross_attn']
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Create diffusion sampler
diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)

# Test with a prompt from the training set
test_prompt = train_metadata[0]['caption']
print(f"\nTest prompt: {test_prompt}")

# Tokenize
max_len = config['text_encoder']['max_seq_len']
token_ids, attention_mask = tokenizer.encode(test_prompt, max_length=max_len)
token_ids = torch.tensor([token_ids], device=device)
attention_mask = torch.tensor([attention_mask], device=device).float()

# Encode text
with torch.no_grad():
    text_emb, _ = text_encoder(token_ids, attention_mask)

# Generate with fast sampling
print("\nGenerating images (fast sampling, 100 steps)...")
batch_size = 4
text_emb = text_emb.repeat(batch_size, 1, 1)
attention_mask = attention_mask.repeat(batch_size, 1)

x = torch.randn((batch_size, 3, 64, 64), device=device)
timesteps = torch.arange(999, -1, -10, device=device, dtype=torch.long)

for i, t in enumerate(timesteps):
    t_batch = t.expand(batch_size)
    with torch.no_grad():
        predicted_noise = model(x, t_batch, text_emb, attention_mask)

    # Get schedule values
    alpha, alpha_bar, one_minus_alpha_bar = diffusion.schedule.get(t_batch, device)
    beta = 1 - alpha

    sqrt_alpha = torch.sqrt(alpha)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)

    # Predict x_0
    x_0_pred = (x - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar

    # Get previous alpha_bar
    if t > 0:
        alpha_bar_prev = diffusion.schedule.alphas_cumprod[t - 1].to(device)
    else:
        alpha_bar_prev = torch.tensor(1.0).to(device)

    sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)

    # Compute posterior mean
    posterior_mean = (sqrt_alpha_bar_prev * beta / one_minus_alpha_bar) * x_0_pred + \
                   (sqrt_alpha * (1 - alpha_bar_prev) / one_minus_alpha_bar) * x

    # Compute posterior variance
    if t > 0:
        posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
    else:
        posterior_variance = torch.tensor(0.0).to(device)

    # Sample from posterior
    if t > 0:
        noise = torch.randn_like(x)
        x = posterior_mean + torch.sqrt(posterior_variance) * noise
    else:
        x = posterior_mean

samples = torch.clamp(x, -1.0, 1.0)

print(f"Generated images shape: {samples.shape}")
print(f"Generated images range: [{samples.min():.4f}, {samples.max():.4f}]")
print(f"Generated images mean: {samples.mean():.4f}, std: {samples.std():.4f}")

# Save images
os.makedirs('outputs_conditional/inference', exist_ok=True)
samples_save = (samples + 1) / 2.0
samples_save = torch.clamp(samples_save, 0, 1)
samples_save = (samples_save * 255).byte()
grid = make_grid(samples_save, nrow=2, padding=2)
grid = grid.permute(1, 2, 0).cpu().numpy()
grid = Image.fromarray(grid)
grid.save('outputs_conditional/inference/test_minimal.png')
print("Saved to outputs_conditional/inference/test_minimal.png")

print("\n" + "="*70)
print("INFERENCE TEST COMPLETE")
print("="*70)
