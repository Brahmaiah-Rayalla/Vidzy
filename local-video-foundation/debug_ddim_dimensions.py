"""
Debug DDIM vs DDPM text conditioning tensor dimensions.
"""
import os
import sys
import torch
import yaml
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import get_device

print("="*70)
print("DEBUGGING DDIM VS DDPM TEXT CONDITIONING")
print("="*70)

device = get_device()
print(f"Device: {device}")

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

# Create diffusion sampler
diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)

# Prepare inputs
test_prompt = "A small red circle on a blue background."
max_len = config['text_encoder']['max_seq_len']

print(f"\n=== TEXT ENCODING ===")
print(f"Prompt: {test_prompt}")
print(f"Max sequence length: {max_len}")

token_ids, attention_mask = tokenizer.encode(test_prompt, max_length=max_len)
print(f"Token IDs (first 10): {token_ids[:10]}")
print(f"Token IDs length: {len(token_ids)}")
print(f"Attention mask length: {len(attention_mask)}")

token_ids_tensor = torch.tensor([token_ids], device=device)
attention_mask_tensor = torch.tensor([attention_mask], device=device).float()

print(f"\nToken IDs tensor shape: {token_ids_tensor.shape}")
print(f"Attention mask tensor shape: {attention_mask_tensor.shape}")

with torch.no_grad():
    text_emb, attention_mask_out = text_encoder(token_ids_tensor, attention_mask_tensor)

print(f"\nText encoder output shape: {text_emb.shape}")
print(f"Attention mask output shape: {attention_mask_out.shape}")

# Prepare diffusion inputs
x = torch.randn(1, 3, 64, 64, device=device)
t = torch.tensor([999], device=device)

print(f"\n=== MODEL INPUTS ===")
print(f"Image tensor shape: {x.shape}")
print(f"Timestep tensor shape: {t.shape}")
print(f"Text embedding shape: {text_emb.shape}")
print(f"Attention mask shape: {attention_mask_out.shape}")

# Test DDPM (works)
print(f"\n=== TESTING DDPM (WORKS) ===")
try:
    with torch.no_grad():
        predicted_noise_ddpm = model(x, t, text_emb, attention_mask_out)
    print(f"DDPM successful!")
    print(f"DDPM output shape: {predicted_noise_ddpm.shape}")
except Exception as e:
    print(f"DDPM failed: {e}")

# Test DDIM (fails)
print(f"\n=== TESTING DDIM (FAILS) ===")
print(f"Attempting to call model with same inputs as DDPM...")
try:
    with torch.no_grad():
        predicted_noise_ddim = model(x, t, text_emb, attention_mask_out)
    print(f"DDIM successful!")
    print(f"DDIM output shape: {predicted_noise_ddim.shape}")
except Exception as e:
    print(f"DDIM failed: {e}")
    print(f"\nThis is the error we need to fix - DDIM uses same model call as DDPM")
    print(f"The error must be in how DDIM calls the model, not the model itself")

# Now test through the actual sampler
print(f"\n=== TESTING THROUGH DDPM SAMPLER ===")
try:
    with torch.no_grad():
        samples_ddpm = diffusion.sample_ddpm(
            model=model,
            batch_size=1,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            device=device,
            text_emb=text_emb,
            text_mask=attention_mask_out
        )
    print(f"DDPM sampler successful!")
    print(f"DDPM output shape: {samples_ddpm.shape}")
except Exception as e:
    print(f"DDPM sampler failed: {e}")

print(f"\n=== TESTING THROUGH DDIM SAMPLER ===")
try:
    with torch.no_grad():
        samples_ddim = diffusion.sample(
            model=model,
            batch_size=1,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            num_sampling_steps=50,
            device=device,
            clip_x0=True,
            text_emb=text_emb,
            text_mask=attention_mask_out
        )
    print(f"DDIM sampler successful!")
    print(f"DDIM output shape: {samples_ddim.shape}")
except Exception as e:
    print(f"DDIM sampler failed: {e}")
    import traceback
    traceback.print_exc()
