"""
Video diffusion training for Stage 3.1.

Trains a video generation model with temporal convolution modules
on synthetic video data.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.conditional_unet import ConditionalUNet
from models.text_encoder import TextEncoder
from models.video_unet import VideoUNet
from diffusion.schedule import GaussianDiffusionSchedule
from diffusion.process import DiffusionProcess


class SyntheticVideoDataset(Dataset):
    """Dataset for synthetic video data."""

    def __init__(self, data_dir, split='train', num_frames=16, image_size=64):
        """
        Initialize dataset.

        Args:
            data_dir: Root data directory
            split: Dataset split ('train', 'val', 'test')
            num_frames: Number of frames per video
            image_size: Size of each frame
        """
        self.data_dir = os.path.join(data_dir, split)
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
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('video_')
        ])

        print(f"Loaded {len(self.video_dirs)} videos from {split} split")

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


def train_video(
    data_dir,
    output_dir,
    num_epochs=10,
    batch_size=1,
    num_frames=16,
    image_size=64,
    learning_rate=1e-4,
    device='cpu',
    spatial_checkpoint_path=None,
    checkpoint_interval=100,
    seed=42
):
    """
    Train video diffusion model.

    Args:
        data_dir: Directory containing synthetic video data
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        num_frames: Number of frames per video
        image_size: Size of each frame
        learning_rate: Learning rate
        device: Device to train on
        spatial_checkpoint_path: Path to Stage 2 spatial checkpoint
        checkpoint_interval: Save checkpoint every N steps
        seed: Random seed
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=num_frames, image_size=image_size)

    # Try to load val dataset, skip if not available
    val_dir = os.path.join(data_dir, 'val')
    if os.path.exists(val_dir) and os.path.exists(os.path.join(val_dir, 'all_metadata.json')):
        val_dataset = SyntheticVideoDataset(data_dir, split='val', num_frames=num_frames, image_size=image_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"Val samples: {len(val_dataset)}")
    else:
        val_dataset = None
        val_loader = None
        print("Val split not found, skipping validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")

    # Load tokenizer or create simple word-based tokenizer
    import json
    tokenizer_path = os.path.join(data_dir, 'train', '..', '..', 'text_image', 'tokenizer', 'tokenizer.json')

    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)

        # Reconstruct tokenizer
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
    else:
        # Create simple word-based tokenizer on the fly
        print("Tokenizer not found, creating simple word-based tokenizer from captions")

        # Collect all words from captions
        all_words = set()
        for caption in train_dataset.captions:
            all_words.update(caption.lower().split())

        # Create word-to-index mapping
        word_to_idx = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        for i, word in enumerate(sorted(all_words), start=3):
            word_to_idx[word] = i

        idx_to_word = {v: k for k, v in word_to_idx.items()}
        vocab_size = len(word_to_idx)

        print(f"Created tokenizer with vocab size: {vocab_size}")

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
        print("No spatial checkpoint provided or not found, training from scratch")

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
        ),
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': vocab_size
    })()

    # Move to device
    video_unet = video_unet.to(device)

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # Create optimizer with separate parameter groups (Stage 3.2.1)
    # Use equal learning rate for fusion gates to enable proper learning
    # (Stage 3.2.3: Fixed overly conservative learning rate that prevented gate learning)
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.bottleneck_fusion.parameters()) + list(video_unet.low_res_fusion.parameters())

    optimizer = optim.Adam([
        {'params': temporal_params, 'lr': learning_rate, 'weight_decay': 1e-5},
        {'params': fusion_params, 'lr': learning_rate, 'weight_decay': 1e-6}
    ])

    print(f"Trainable parameters: {sum(p.numel() for p in video_unet.parameters() if p.requires_grad):,}")
    print(f"Temporal parameters: {sum(p.numel() for p in temporal_params):,}")
    print(f"Fusion parameters: {sum(p.numel() for p in fusion_params):,}")

    # Training loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        video_unet.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}")
        for batch in pbar:
            # Get batch
            videos = batch['video'].to(device)  # [batch, channels, num_frames, h, w]
            captions = batch['caption']

            # Normalize videos (already normalized in dataset)

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
            text_emb = text_emb_encoded  # Use encoded embeddings instead of token IDs

            # Sample timesteps
            timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

            # Forward diffusion (frame-by-frame)
            # Reshape: [batch, channels, frames, h, w] -> [batch*frames, channels, h, w]
            videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
            timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)

            noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)

            # Reshape back to video format
            noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

            # Predict noise
            predicted_noise = video_unet(noisy_videos, timesteps, text_emb, text_mask)

            # Compute loss
            # Reshape predicted noise to match noise shape
            predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)

            loss = nn.functional.mse_loss(predicted_noise_flat, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Temporal feature diagnostics (Stage 3.2.1)
            if global_step % 10 == 0:
                # Log gate values
                bottleneck_gate = video_unet.bottleneck_fusion.gate_activation(video_unet.bottleneck_fusion.gate).item()
                low_res_gate = video_unet.low_res_fusion.gate_activation(video_unet.low_res_fusion.gate).item()

                # Log temporal parameter gradient norms
                temporal_grad_norms = []
                for name, param in video_unet.temporal_modules.named_parameters():
                    if param.grad is not None:
                        temporal_grad_norms.append(param.grad.norm().item())

                fusion_grad_norms = []
                for name, param in video_unet.bottleneck_fusion.named_parameters():
                    if param.grad is not None:
                        fusion_grad_norms.append(param.grad.norm().item())
                for name, param in video_unet.low_res_fusion.named_parameters():
                    if param.grad is not None:
                        fusion_grad_norms.append(param.grad.norm().item())

                if temporal_grad_norms:
                    avg_temporal_grad = np.mean(temporal_grad_norms)
                else:
                    avg_temporal_grad = 0.0

                if fusion_grad_norms:
                    avg_fusion_grad = np.mean(fusion_grad_norms)
                else:
                    avg_fusion_grad = 0.0

                diagnostics = {
                    'step': global_step,
                    'loss': loss.item(),
                    'bottleneck_gate': bottleneck_gate,
                    'low_res_gate': low_res_gate,
                    'temporal_grad_norm': avg_temporal_grad,
                    'fusion_grad_norm': avg_fusion_grad
                }

                # Log to console occasionally
                if global_step % 50 == 0:
                    print(f"\nStep {global_step} diagnostics:")
                    print(f"  Loss: {loss.item():.4f}")
                    print(f"  Bottleneck gate: {bottleneck_gate:.6f}")
                    print(f"  Low-res gate: {low_res_gate:.6f}")
                    print(f"  Temporal grad norm: {avg_temporal_grad:.6f}")
                    print(f"  Fusion grad norm: {avg_fusion_grad:.6f}")

                # Save diagnostics to file
                diagnostics_path = os.path.join(output_dir, 'training_diagnostics.jsonl')
                with open(diagnostics_path, 'a') as f:
                    f.write(json.dumps(diagnostics) + '\n')

            optimizer.step()

            train_loss += loss.item()
            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Save checkpoint
            if global_step % checkpoint_interval == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{global_step:06d}.pt')
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': video_unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation (if available)
        if val_loader is not None:
            video_unet.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    videos = batch['video'].to(device)
                    captions = batch['caption']

                    # Encode captions
                    batch_size = videos.shape[0]
                    text_embs = []
                    text_masks = []

                    for caption in captions:
                        token_ids, mask = video_unet.tokenizer.encode(caption, max_length=32)
                        text_embs.append(token_ids)
                        text_masks.append(mask)

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
                    text_emb = text_emb_encoded  # Use encoded embeddings instead of token IDs

                    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

                    videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
                    timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)

                    noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)
                    noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

                    predicted_noise = video_unet(noisy_videos, timesteps, text_emb, text_mask)
                    predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)

                    loss = nn.functional.mse_loss(predicted_noise_flat, noise)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Average validation loss: {avg_val_loss:.4f}")

            # Save best checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pt')
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': video_unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, best_checkpoint_path)
                print(f"Saved best checkpoint to {best_checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = os.path.join(output_dir, 'checkpoint_final.pt')
    torch.save({
        'step': global_step,
        'model_state_dict': video_unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss if val_loader is not None else avg_train_loss,
    }, final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train video diffusion model")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_video", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints_video", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (optimized for Stage 3.2.1)")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--spatial_checkpoint", type=str, default=None, help="Spatial checkpoint path")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Checkpoint interval (lower for frequent saves)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_video(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        device=args.device,
        spatial_checkpoint_path=args.spatial_checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
