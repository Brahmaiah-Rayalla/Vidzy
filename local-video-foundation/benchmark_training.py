"""
Benchmark training speed to estimate time for full training.
"""
import os
import sys
import time
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.data.text_image_dataset import TextImageShapeGenerator, create_train_val_test_splits
from video_foundation.data.text_image_loader import TextImageDataset
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device

# Load config
with open('configs/conditional_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Override for benchmark
config['dataset']['num_images'] = 100  # Small dataset for benchmark
config['training']['num_steps'] = 20  # Only 20 steps for benchmark

print("="*60)
print("TRAINING SPEED BENCHMARK")
print("="*60)
print(f"Device: {get_device()}")
print(f"Benchmark steps: {config['training']['num_steps']}")
print(f"Dataset size: {config['dataset']['num_images']}")
print(f"Batch size: {config['training']['batch_size']}")
print("="*60)

# Setup
device = get_device()
set_seed(42)

# Generate dataset
data_dir = "data/benchmark"
print(f"\nGenerating dataset in {data_dir}...")
generator = TextImageShapeGenerator(
    image_size=config['dataset']['image_size'],
    seed=config['dataset']['seed'],
    max_objects=config['dataset'].get('max_objects', 2)
)
metadata = generator.generate_dataset(
    num_images=config['dataset']['num_images'],
    output_dir=data_dir,
    save_preview=False
)

create_train_val_test_splits(metadata, data_dir, 0.8, 0.1, 0.1)

# Build tokenizer
train_metadata_path = os.path.join(data_dir, 'train_metadata.json')
import json
with open(train_metadata_path, 'r') as f:
    train_metadata = json.load(f)

captions = [m['caption'] for m in train_metadata]
tokenizer = SimpleTokenizer(
    vocab_size=config['text_encoder']['vocab_size'],
    min_freq=config['text_encoder']['min_freq']
)
tokenizer.build_vocab(captions)

# Create dataset
train_dataset = TextImageDataset(
    metadata_path=train_metadata_path,
    image_dir=data_dir,
    image_size=config['model']['image_size'],
    normalize=True
)

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    return {'image': images, 'caption': captions}

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    collate_fn=collate_fn
)

# Create models
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

# Create diffusion
diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)

# Create optimizer
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(text_encoder.parameters()),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# Benchmark training
print("\nStarting benchmark...")
model.train()
text_encoder.train()

total_time = 0
step_times = []

for step in range(config['training']['num_steps']):
    for batch in train_loader:
        if step >= config['training']['num_steps']:
            break

        start_time = time.time()

        # Move to device
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
        timesteps = torch.randint(
            0, config['diffusion']['num_timesteps'],
            (batch_size,), device=device
        )

        # Forward diffusion
        noise = torch.randn_like(x_0)
        x_t, noise = diffusion.process.q_sample(x_0, timesteps, noise)

        # Predict noise
        predicted_noise = model(x_t, timesteps, text_emb, attention_mask)

        # Loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_time = time.time() - start_time
        step_times.append(step_time)
        total_time += step_time

        print(f"Step {step+1}/{config['training']['num_steps']}: {step_time:.2f}s, loss: {loss.item():.6f}")

# Calculate statistics
avg_step_time = sum(step_times) / len(step_times)
median_step_time = sorted(step_times)[len(step_times) // 2]

print("\n" + "="*60)
print("BENCHMARK RESULTS")
print("="*60)
print(f"Total time for {config['training']['num_steps']} steps: {total_time:.2f}s")
print(f"Average step time: {avg_step_time:.2f}s")
print(f"Median step time: {median_step_time:.2f}s")
print()
print("ESTIMATES FOR FULL TRAINING:")
print(f"500 steps: ~{avg_step_time * 500 / 60:.1f} minutes")
print(f"1,000 steps: ~{avg_step_time * 1000 / 60:.1f} minutes")
print(f"5,000 steps: ~{avg_step_time * 5000 / 60:.1f} minutes (~{avg_step_time * 5000 / 3600:.1f} hours)")
print("="*60)
