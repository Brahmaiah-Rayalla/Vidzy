"""
Evaluation with contrasting prompts to test text control.
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
print("EVALUATION WITH CONTRASTING PROMPTS")
print("="*70)

# Load config
with open('configs/conditional_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = get_device()

# Load checkpoint
print("\nLoading checkpoint...")
checkpoint = torch.load('checkpoints_conditional_minimal/checkpoint_minimal.pt', map_location=device)

# Build tokenizer
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
    n_heads=config.get('n_heads', 4),
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

# Contrasting prompt pairs
prompt_pairs = [
    ("A small red circle on a blue background.", "A small blue circle on a red background."),
    ("A large green triangle near the center.", "A large green triangle near the left."),
    ("A yellow square on a white background.", "A yellow square on a black background."),
]

def generate_images(prompt, seed):
    """Generate images for a single prompt."""
    set_seed(seed)

    # Tokenize
    max_len = config['text_encoder']['max_seq_len']
    token_ids, attention_mask = tokenizer.encode(prompt, max_length=max_len)
    token_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device).float()

    # Encode text
    with torch.no_grad():
        text_emb, _ = text_encoder(token_ids, attention_mask)

    # Generate
    batch_size = 2
    text_emb = text_emb.repeat(batch_size, 1, 1)
    attention_mask = attention_mask.repeat(batch_size, 1)

    x = torch.randn((batch_size, 3, 64, 64), device=device)
    timesteps = torch.arange(999, -1, -10, device=device, dtype=torch.long)

    for t in timesteps:
        t_batch = t.expand(batch_size)
        with torch.no_grad():
            predicted_noise = model(x, t_batch, text_emb, attention_mask)

        alpha, alpha_bar, one_minus_alpha_bar = diffusion.schedule.get(t_batch, device)
        beta = 1 - alpha

        sqrt_alpha = torch.sqrt(alpha)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)

        x_0_pred = (x - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar

        if t > 0:
            alpha_bar_prev = diffusion.schedule.alphas_cumprod[t - 1].to(device)
        else:
            alpha_bar_prev = torch.tensor(1.0).to(device)

        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)

        posterior_mean = (sqrt_alpha_bar_prev * beta / one_minus_alpha_bar) * x_0_pred + \
                       (sqrt_alpha * (1 - alpha_bar_prev) / one_minus_alpha_bar) * x

        if t > 0:
            posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        else:
            posterior_variance = torch.tensor(0.0).to(device)

        if t > 0:
            noise = torch.randn_like(x)
            x = posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            x = posterior_mean

    return torch.clamp(x, -1.0, 1.0)

# Generate for all prompt pairs
os.makedirs('outputs_conditional/evaluation', exist_ok=True)
all_images = []
all_prompts = []

for i, (prompt1, prompt2) in enumerate(prompt_pairs):
    print(f"\nGenerating pair {i+1}:")
    print(f"  Prompt 1: {prompt1}")
    print(f"  Prompt 2: {prompt2}")

    # Generate with same seed for consistency
    images1 = generate_images(prompt1, seed=42 + i * 2)
    images2 = generate_images(prompt2, seed=42 + i * 2 + 1)

    all_images.append(images1)
    all_images.append(images2)
    all_prompts.append(prompt1)
    all_prompts.append(prompt2)

# Create comparison grid
all_images = torch.cat(all_images, dim=0)
all_images_save = (all_images + 1) / 2.0
all_images_save = torch.clamp(all_images_save, 0, 1)
all_images_save = (all_images_save * 255).byte()
grid = make_grid(all_images_save, nrow=4, padding=4)
grid = grid.permute(1, 2, 0).cpu().numpy()
grid = Image.fromarray(grid)
grid.save('outputs_conditional/evaluation/contrasting_prompts.png')
print("\nSaved comparison grid to outputs_conditional/evaluation/contrasting_prompts.png")

# Save prompts
with open('outputs_conditional/evaluation/prompts.txt', 'w') as f:
    for i, prompt in enumerate(all_prompts):
        f.write(f"{i+1}. {prompt}\n")
print("Saved prompts to outputs_conditional/evaluation/prompts.txt")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
print("\nNOTE: This is a minimally trained model (10 steps, 50 examples).")
print("The model is not expected to show strong text control.")
print("This evaluation tests the pipeline, not the quality.")
