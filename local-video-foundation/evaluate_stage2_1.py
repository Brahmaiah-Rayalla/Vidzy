"""
Evaluation script for Stage 2.1 - Extended Text-to-Image Training.
Tests text control with contrasting prompts using the final checkpoint.
"""
import os
import torch
import yaml
import json
import numpy as np
from PIL import Image
from torchvision.utils import make_grid

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device

print("="*70)
print("STAGE 2.1 EVALUATION - TEXT CONTROL TEST")
print("="*70)

# Load config
with open('configs/conditional_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = get_device()
print(f"\nDevice: {device}")

# Load checkpoint
checkpoint_path = "checkpoints_conditional/checkpoint_final.pt"
print(f"\nLoading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)
print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")

# Build tokenizer from training data
print("\nBuilding tokenizer...")
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
    n_heads=config['model'].get('n_heads', 4),
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

# Define contrasting prompt pairs
prompt_pairs = [
    ("A small red circle on a blue background.", "A small blue circle on a red background."),
    ("A large green triangle near the center.", "A large green triangle near the left."),
    ("A yellow square on a white background.", "A yellow square on a black background."),
    ("A small red circle in the center.", "A large red circle in the center."),
    ("A green triangle on the left.", "A green triangle on the right."),
]

def generate_images(prompt, seed, num_samples=2):
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

    # Generate multiple samples
    text_emb = text_emb.repeat(num_samples, 1, 1)
    attention_mask = attention_mask.repeat(num_samples, 1)

    with torch.no_grad():
        samples = diffusion.sample_ddpm(
            model=model,
            batch_size=num_samples,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            device=device,
            text_emb=text_emb,
            text_mask=attention_mask
        )

    return torch.clamp(samples, -1.0, 1.0)

# Generate for all prompt pairs
os.makedirs('outputs_conditional/stage2_1_evaluation', exist_ok=True)
all_images = []
all_prompts = []
sample_stats = []

print("\n" + "="*70)
print("GENERATING SAMPLES FOR CONTRASTING PROMPTS")
print("="*70)

for i, (prompt1, prompt2) in enumerate(prompt_pairs):
    print(f"\nPair {i+1}:")
    print(f"  Prompt 1: {prompt1}")
    print(f"  Prompt 2: {prompt2}")

    # Generate with same seed for consistency
    images1 = generate_images(prompt1, seed=42 + i * 2, num_samples=2)
    images2 = generate_images(prompt2, seed=42 + i * 2 + 1, num_samples=2)

    # Compute statistics
    stats1 = {
        'mean': images1.mean().item(),
        'std': images1.std().item(),
        'min': images1.min().item(),
        'max': images1.max().item()
    }
    stats2 = {
        'mean': images2.mean().item(),
        'std': images2.std().item(),
        'min': images2.min().item(),
        'max': images2.max().item()
    }

    sample_stats.append({
        'pair': i+1,
        'prompt1': prompt1,
        'prompt2': prompt2,
        'stats1': stats1,
        'stats2': stats2
    })

    print(f"  Prompt 1 stats: mean={stats1['mean']:.3f}, std={stats1['std']:.3f}")
    print(f"  Prompt 2 stats: mean={stats2['mean']:.3f}, std={stats2['std']:.3f}")

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
grid_path = 'outputs_conditional/stage2_1_evaluation/contrasting_prompts.png'
grid.save(grid_path)
print(f"\nSaved comparison grid to {grid_path}")

# Save prompts
with open('outputs_conditional/stage2_1_evaluation/prompts.txt', 'w') as f:
    for i, prompt in enumerate(all_prompts):
        f.write(f"{i+1}. {prompt}\n")
print("Saved prompts to outputs_conditional/stage2_1_evaluation/prompts.txt")

# Save statistics
with open('outputs_conditional/stage2_1_evaluation/sample_stats.json', 'w') as f:
    json.dump(sample_stats, f, indent=2)
print("Saved statistics to outputs_conditional/stage2_1_evaluation/sample_stats.json")

# Print summary
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)
print(f"Total prompt pairs: {len(prompt_pairs)}")
print(f"Total images generated: {len(all_images)}")
print(f"Images per prompt: 2")

# Compute overall statistics
all_stats = {
    'mean': all_images.mean().item(),
    'std': all_images.std().item(),
    'min': all_images.min().item(),
    'max': all_images.max().item()
}
print(f"\nOverall statistics:")
print(f"  Mean: {all_stats['mean']:.3f}")
print(f"  Std: {all_stats['std']:.3f}")
print(f"  Range: [{all_stats['min']:.3f}, {all_stats['max']:.3f}]")

# Check for mode collapse
if all_stats['std'] < 0.1:
    print("\nWARNING: Low standard deviation suggests potential mode collapse")
else:
    print("\nOK: Reasonable sample diversity")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
print("\nReview the generated images to assess text control:")
print(f"  {grid_path}")
print("\nText control is considered successful if:")
print("  - Images from contrasting prompts show clear differences")
print("  - Color, position, and size attributes are correctly rendered")
print("  - Samples are diverse (not all identical)")
