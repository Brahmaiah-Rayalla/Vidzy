"""
Overfitting experiment to verify temporal module learning.

Trains on a very small dataset (10 samples) for many epochs to verify
that temporal modules can learn temporal relationships.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess
from video_foundation.data.synthetic_video_dataset import SyntheticVideoGenerator
from torch.utils.data import Dataset


class SyntheticVideoDatasetWrapper(Dataset):
    """PyTorch Dataset wrapper for SyntheticVideoGenerator."""

    def __init__(self, num_samples, num_frames, image_size, seed=42):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.image_size = image_size
        self.generator = SyntheticVideoGenerator(
            image_size=image_size,
            num_frames=num_frames,
            seed=seed
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a single video
        frames, caption, metadata = self.generator.generate_single_video()

        # Convert frames to tensor
        frames_np = np.stack([np.array(frame) for frame in frames])  # [frames, h, w, c]
        frames_np = frames_np.transpose(0, 3, 1, 2)  # [frames, c, h, w]
        frames_tensor = torch.from_numpy(frames_np).float() / 127.5 - 1.0  # Normalize to [-1, 1]

        return {
            'video': frames_tensor,
            'caption': caption,
            'metadata': metadata
        }


def train_overfit_epoch(model, diffusion, dataloader, optimizer, device, epoch):
    """Train for one epoch and return loss."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Get data
        videos = batch['video'].to(device)  # [batch, 3, frames, h, w]
        captions = batch['caption']

        batch_size, channels, num_frames, height, width = videos.shape

        # Tokenize captions
        token_ids_list = []
        mask_list = []
        for caption in captions:
            token_ids, mask = model.tokenizer.encode(caption, max_length=32)
            # Pad to 32
            token_ids = token_ids + [0] * (32 - len(token_ids))
            mask = mask + [0] * (32 - len(mask))
            token_ids_list.append(token_ids)
            mask_list.append(mask)

        text_emb_tokens = torch.tensor(token_ids_list).to(device)
        text_mask = torch.tensor(mask_list).to(device)

        # Encode text
        with torch.no_grad():
            text_emb, _ = model.text_encoder(text_emb_tokens, text_mask)

        # Sample timesteps
        t = torch.randint(0, diffusion.schedule.num_timesteps, (batch_size,), device=device)

        # Add noise - manually compute for 5D video tensors
        noise = torch.randn_like(videos)
        _, alpha_bar, one_minus_alpha_bar = diffusion.schedule.get(t, device)
        # Reshape schedule values for 5D: [batch, 1, 1, 1, 1]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1, 1)
        one_minus_alpha_bar = one_minus_alpha_bar.view(-1, 1, 1, 1, 1)
        noisy_videos = torch.sqrt(alpha_bar) * videos + torch.sqrt(one_minus_alpha_bar) * noise

        # Predict noise
        predicted_noise = model(noisy_videos, t, text_emb, text_mask)

        # Compute loss
        loss = nn.functional.mse_loss(predicted_noise, noise)

        # Compute loss
        loss = nn.functional.mse_loss(predicted_noise, noise)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 5 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, diffusion, dataloader, device):
    """Evaluate model and return loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            videos = batch['video'].to(device)
            captions = batch['caption']

            batch_size, channels, num_frames, height, width = videos.shape

            # Tokenize captions
            token_ids_list = []
            mask_list = []
            for caption in captions:
                token_ids, mask = model.tokenizer.encode(caption, max_length=32)
                token_ids = token_ids + [0] * (32 - len(token_ids))
                mask = mask + [0] * (32 - len(mask))
                token_ids_list.append(token_ids)
                mask_list.append(mask)

            text_emb_tokens = torch.tensor(token_ids_list).to(device)
            text_mask = torch.tensor(mask_list).to(device)

            # Encode text
            text_emb, _ = model.text_encoder(text_emb_tokens, text_mask)

            # Sample timesteps
            t = torch.randint(0, diffusion.schedule.num_timesteps, (batch_size,), device=device)

            # Add noise - manually compute for 5D video tensors
            noise = torch.randn_like(videos)
            _, alpha_bar, one_minus_alpha_bar = diffusion.schedule.get(t, device)
            # Reshape schedule values for 5D: [batch, 1, 1, 1, 1]
            alpha_bar = alpha_bar.view(-1, 1, 1, 1, 1)
            one_minus_alpha_bar = one_minus_alpha_bar.view(-1, 1, 1, 1, 1)
            noisy_videos = torch.sqrt(alpha_bar) * videos + torch.sqrt(one_minus_alpha_bar) * noise

            # Predict noise
            predicted_noise = model(noisy_videos, t, text_emb, text_mask)

            # Compute loss
            loss = nn.functional.mse_loss(predicted_noise, noise)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    # Configuration
    device = 'cpu'
    num_samples = 10  # Very small dataset for overfitting
    num_epochs = 10  # Many epochs to overfit
    batch_size = 2
    num_frames = 16
    image_size = 64
    learning_rate = 1e-4
    output_dir = "checkpoints_overfit"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("OVERFITTING EXPERIMENT: Temporal Module Learning")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset size: {num_samples} samples")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()

    # Load tokenizer
    tokenizer_path = "data/text_image/tokenizer/tokenizer.json"
    with open(tokenizer_path, 'r') as f:
        data = json.load(f)

    idx_to_word = {}
    word_to_idx = {}
    for key, value in data['idx_to_word'].items():
        if key.startswith('<'):
            idx = int(value)
            token = key
            idx_to_word[idx] = token
            word_to_idx[token] = idx
        else:
            idx = int(key)
            token = value
            idx_to_word[idx] = token
            word_to_idx[token] = idx

    vocab_size = len(word_to_idx)

    # Create spatial U-Net
    spatial_unet = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=1000,
        text_dim=256,
        n_heads=4,
        use_cross_attn=True
    )

    # Create text encoder
    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=True,
        use_temporal=True
    )

    # Load spatial weights from Stage 2 checkpoint
    spatial_checkpoint_path = "checkpoints_conditional_minimal/checkpoint_step_000010.pt"
    if os.path.exists(spatial_checkpoint_path):
        print(f"Loading spatial weights from {spatial_checkpoint_path}")
        spatial_checkpoint = torch.load(spatial_checkpoint_path, map_location='cpu')
        video_unet.load_spatial_weights(spatial_checkpoint['model_state_dict'])
    else:
        print("Warning: Spatial checkpoint not found, using random initialization")

    video_unet = video_unet.to(device)

    # Create tokenizer
    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()

    # Create small synthetic dataset
    print(f"Creating synthetic dataset with {num_samples} samples...")
    dataset = SyntheticVideoDatasetWrapper(
        num_samples=num_samples,
        num_frames=num_frames,
        image_size=image_size,
        seed=42
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # Create optimizer (only temporal parameters)
    temporal_params = [p for n, p in video_unet.named_parameters() if 'temporal' in n]
    optimizer = optim.Adam(temporal_params, lr=learning_rate)

    print(f"Training temporal parameters: {sum(p.numel() for p in temporal_params):,}")
    print()

    # Training loop
    train_losses = []
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        start_time = time.time()

        # Train
        train_loss = train_overfit_epoch(video_unet, diffusion, dataloader, optimizer, device, epoch+1)
        train_losses.append(train_loss)

        epoch_time = time.time() - start_time

        print(f"  Train Loss: {train_loss:.6f}, Time: {epoch_time:.2f}s")

        # Save checkpoint if loss improved
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(output_dir, f"checkpoint_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': video_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'train_losses': train_losses
            }, checkpoint_path)
            print(f"  Saved best checkpoint (loss: {best_loss:.6f})")

        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1:03d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': video_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'train_losses': train_losses
            }, checkpoint_path)
            print(f"  Saved checkpoint at epoch {epoch+1}")

        print()

    # Final evaluation
    print("=" * 60)
    print("OVERFITTING EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Best train loss: {best_loss:.6f}")
    print(f"Loss reduction: {train_losses[0] - train_losses[-1]:.6f}")
    print()

    # Save training history
    history = {
        'train_losses': train_losses,
        'best_loss': best_loss,
        'num_samples': num_samples,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }

    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Saved training history to {history_path}")

    # Check if overfitting was successful
    if train_losses[-1] < 0.1 and train_losses[0] - train_losses[-1] > 0.5:
        print("SUCCESS: Temporal modules appear to be learning (loss decreased significantly)")
    else:
        print("WARNING: Temporal modules may not be learning effectively")


if __name__ == "__main__":
    main()
