"""
Controlled overfitting experiment for Stage 3.2.

Trains the temporal modules on a small, controlled dataset to determine
if they can learn specific video trajectories.

Key settings:
- Freeze spatial backbone
- Freeze text encoder
- Train only temporal modules
- Batch size 1
- 16 frames at 64x64
- Generate samples at multiple training intervals
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


class ControlledVideoDataset(Dataset):
    """Dataset for controlled video data."""

    def __init__(self, data_dir, num_frames=16, image_size=64):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing train split
            num_frames: Number of frames per video
            image_size: Size of each frame
        """
        self.data_dir = os.path.join(data_dir, "train")
        self.num_frames = num_frames
        self.image_size = image_size

        # Load metadata
        metadata_path = os.path.join(self.data_dir, "all_metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load captions
        captions_path = os.path.join(self.data_dir, "captions.txt")
        with open(captions_path, 'r') as f:
            self.captions = [line.strip() for line in f.readlines()]

        # Get video directories
        self.video_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])

        print(f"Loaded {len(self.video_dirs)} videos from controlled dataset")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        """Load a video sample."""
        video_dir = os.path.join(self.data_dir, self.video_dirs[idx])

        # Load frames
        frame_paths = sorted([
            f for f in os.listdir(video_dir)
            if f.endswith('.png')
        ])

        frames = []
        for frame_path in frame_paths[:self.num_frames]:
            frame = Image.open(os.path.join(video_dir, frame_path))
            frame = frame.resize((self.image_size, self.image_size))
            frame = np.array(frame).astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
            frames.append(frame)

        # Stack frames: [num_frames, height, width, channels] -> [channels, num_frames, height, width]
        video = np.stack(frames, axis=0)  # [num_frames, h, w, c]
        video = video.transpose(3, 0, 1, 2)  # [c, num_frames, h, w]

        # Get caption
        caption = self.captions[idx]

        return {
            'video': torch.from_numpy(video).float(),
            'caption': caption,
            'metadata': self.metadata[idx]
        }


def run_controlled_overfitting(
    data_dir,
    output_dir,
    num_epochs=50,
    batch_size=1,
    num_frames=16,
    image_size=64,
    learning_rate=1e-4,
    device='cpu',
    spatial_checkpoint_path=None,
    sample_interval=10,
    seed=42
):
    """
    Run controlled overfitting experiment.

    Args:
        data_dir: Directory containing controlled video data
        output_dir: Directory to save checkpoints and samples
        num_epochs: Number of training epochs
        batch_size: Batch size (should be 1 for overfitting)
        num_frames: Number of frames per video
        image_size: Size of each frame
        learning_rate: Learning rate
        device: Device to train on
        spatial_checkpoint_path: Path to Stage 2 spatial checkpoint
        sample_interval: Generate samples every N epochs
        seed: Random seed
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CONTROLLED OVERFITTING EXPERIMENT")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print("=" * 60)
    print()

    # Load dataset
    print("Loading controlled dataset...")
    dataset = ControlledVideoDataset(data_dir, num_frames=num_frames, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Dataset size: {len(dataset)}")
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

    # Create models
    print("Creating models...")
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

    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Load spatial checkpoint if provided
    if spatial_checkpoint_path and os.path.exists(spatial_checkpoint_path):
        print(f"Loading spatial checkpoint from {spatial_checkpoint_path}")
        checkpoint = torch.load(spatial_checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        spatial_unet.load_state_dict(state_dict, strict=False)

        # Load text encoder from checkpoint
        if 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    else:
        print("Warning: Spatial checkpoint not found, using random initialization")

    # Create video U-Net with temporal modules
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=True,  # Freeze spatial backbone
        use_temporal=True
    )

    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()

    # Move to device
    video_unet = video_unet.to(device)

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # Create optimizer (only temporal parameters)
    temporal_params = [p for n, p in video_unet.named_parameters() if 'temporal' in n]
    optimizer = optim.Adam(temporal_params, lr=learning_rate)

    print(f"Temporal parameters: {sum(p.numel() for p in temporal_params):,}")
    print()

    # Training loop
    global_step = 0
    best_loss = float('inf')
    train_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        video_unet.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Training epoch {epoch + 1}")
        for batch in pbar:
            # Get batch
            videos = batch['video'].to(device)  # [batch, channels, num_frames, h, w]
            captions = batch['caption']

            # Encode captions
            batch_size = videos.shape[0]
            text_embs = []
            text_masks = []

            for caption in captions:
                token_ids, mask = video_unet.tokenizer.encode(caption, max_length=32)
                text_embs.append(token_ids)
                text_masks.append(mask)

            # Pad token sequences
            max_len = max(len(emb) for emb in text_embs)
            padded_embs = []
            padded_masks = []
            for emb, mask in zip(text_embs, text_masks):
                padded = emb + [0] * (max_len - len(emb))
                padded_mask = mask + [0] * (max_len - len(mask))
                padded_embs.append(padded)
                padded_masks.append(padded_mask)

            text_emb = torch.tensor(padded_embs).to(device)
            text_mask = torch.tensor(padded_masks).to(device)

            # Encode text using text encoder
            text_emb_encoded, _ = video_unet.text_encoder(text_emb, text_mask)
            text_emb = text_emb_encoded

            # Sample timesteps
            timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

            # Forward diffusion (frame-by-frame)
            videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
            timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)

            noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)

            # Reshape back to video format
            noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

            # Predict noise
            predicted_noise = video_unet(noisy_videos, timesteps, text_emb, text_mask)

            # Compute loss
            predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
            loss = nn.functional.mse_loss(predicted_noise_flat, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_epoch_loss)
        print(f"Average training loss: {avg_epoch_loss:.4f}")

        # Save checkpoint if loss improved
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': video_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'train_losses': train_losses
            }, checkpoint_path)
            print(f"Saved best checkpoint (loss: {best_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1:03d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': video_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'train_losses': train_losses
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

    # Save final checkpoint
    final_checkpoint_path = os.path.join(output_dir, 'checkpoint_final.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': video_unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
        'train_losses': train_losses
    }, final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    # Save training history
    history = {
        'train_losses': train_losses,
        'best_loss': best_loss,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'data_dir': data_dir
    }

    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Saved training history to {history_path}")

    print("\n" + "=" * 60)
    print("OVERFITTING EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Final loss: {avg_epoch_loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Loss reduction: {train_losses[0] - train_losses[-1]:.4f}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Controlled overfitting experiment")
    parser.add_argument("--data_dir", type=str, default="data/controlled_video", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints_controlled_overfit", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--spatial_checkpoint", type=str, default=None, help="Spatial checkpoint path")
    parser.add_argument("--sample_interval", type=int, default=10, help="Sample generation interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_controlled_overfitting(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        device=args.device,
        spatial_checkpoint_path=args.spatial_checkpoint,
        sample_interval=args.sample_interval,
        seed=args.seed
    )
