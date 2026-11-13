"""
Minimal training run to verify the conditional pipeline.
"""

import os
import torch
import yaml
import json
from video_foundation.data.text_image_dataset import TextImageShapeGenerator, create_train_val_test_splits
from video_foundation.data.text_image_loader import TextImageDataset
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device
import torch.optim as optim

print("="*70)
print("MINIMAL CONDITIONAL TRAINING RUN")
print("="*70)

# Load config
with open('configs/conditional_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = get_device()
set_seed(42)

# Generate minimal dataset
print("\nGenerating minimal dataset (50 examples)...")
generator = TextImageShapeGenerator(
    image_size=config['dataset']['image_size'],
    seed=config['dataset']['seed'],
    max_objects=config['dataset']['max_objects']
)
metadata = generator.generate_dataset(
    num_images=50,
    output_dir='data/conditional_minimal',
    save_preview=False
)

# Create splits
create_train_val_test_splits(metadata, 'data/conditional_minimal')

# Build tokenizer
print("\nBuilding tokenizer...")
import json
with open('data/conditional_minimal/train_metadata.json', 'r') as f:
    train_metadata = json.load(f)

captions = [m['caption'] for m in train_metadata]
tokenizer = SimpleTokenizer(
    vocab_size=config['text_encoder']['vocab_size'],
    min_freq=1  # Lower threshold for small dataset
)
tokenizer.build_vocab(captions)

# Create datasets
print("Loading datasets...")
train_dataset = TextImageDataset(
    metadata_path='data/conditional_minimal/train_metadata.json',
    image_dir='data/conditional_minimal',
    image_size=config['model']['image_size'],
    normalize=True
)

def collate_fn(batch):
    """Custom collate function to handle dictionary batches."""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    return {'image': images, 'caption': captions}

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

# Create models
print("Creating models...")
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

# Create diffusion sampler
diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)

# Create optimizer
optimizer = optim.AdamW(
    list(model.parameters()) + list(text_encoder.parameters()),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# Training loop
print("\nTraining for 10 steps...")
model.train()
text_encoder.train()

for step in range(10):
    for batch in train_loader:
        x_0 = batch['image'].to(device)
        captions = batch['caption']

        # Tokenize
        max_len = config['text_encoder']['max_seq_len']
        token_ids_list = []
        attention_masks_list = []

        for caption in captions:
            token_ids, attention_mask = tokenizer.encode(caption, max_length=max_len)
            token_ids_list.append(token_ids)
            attention_masks_list.append(attention_mask)

        token_ids = torch.tensor(token_ids_list, device=device)
        attention_mask = torch.tensor(attention_masks_list, device=device).float()

        # Encode text
        text_emb, _ = text_encoder(token_ids, attention_mask)

        # Sample timesteps
        batch_size = x_0.shape[0]
        timesteps = torch.randint(0, config['diffusion']['num_timesteps'], (batch_size,), device=device)

        # Compute loss
        noise = torch.randn_like(x_0)
        x_t, noise = diffusion.process.q_sample(x_0, timesteps, noise)
        predicted_noise = model(x_t, timesteps, text_emb, attention_mask)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step + 1}: Loss = {loss.item():.6f}")
        step += 1

        if step >= 10:
            break

    if step >= 10:
        break

# Save checkpoint
print("\nSaving checkpoint...")
os.makedirs('checkpoints_conditional_minimal', exist_ok=True)
torch.save({
    'step': 10,
    'model_state_dict': model.state_dict(),
    'text_encoder_state_dict': text_encoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'seed': 42,
}, 'checkpoints_conditional_minimal/checkpoint_minimal.pt')

print("Checkpoint saved to checkpoints_conditional_minimal/checkpoint_minimal.pt")

print("\n" + "="*70)
print("MINIMAL TRAINING COMPLETE")
print("="*70)
