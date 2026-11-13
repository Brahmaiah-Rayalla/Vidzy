"""
Fixed video diffusion training with explicit spatial backbone protection.

This version explicitly ensures spatial parameters are excluded from the optimizer
and validates they don't change during training.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


class SyntheticVideoDataset(Dataset):
    """Dataset for synthetic video data."""

    def __init__(self, data_dir, split='train', num_frames=16, image_size=64):
        self.data_dir = os.path.join(data_dir, split)
        self.num_frames = num_frames
        self.image_size = image_size

        metadata_path = os.path.join(self.data_dir, "all_metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        captions_path = os.path.join(self.data_dir, "captions.txt")
        with open(captions_path, 'r') as f:
            self.captions = [line.strip() for line in f.readlines()]

        self.video_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('video_')
        ])

        print(f"Loaded {len(self.video_dirs)} videos from {split} split")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = os.path.join(self.data_dir, self.video_dirs[idx])

        frame_paths = sorted([
            f for f in os.listdir(video_dir)
            if f.endswith('.png')
        ])

        frames = []
        for frame_path in frame_paths[:self.num_frames]:
            frame = Image.open(os.path.join(video_dir, frame_path))
            frame = frame.resize((self.image_size, self.image_size))
            frame = np.array(frame).astype(np.float32) / 127.5 - 1.0
            frames.append(frame)

        video = np.stack(frames, axis=0)
        video = video.transpose(3, 0, 1, 2)

        caption = self.captions[idx]

        return {
            'video': torch.from_numpy(video).float(),
            'caption': caption,
            'metadata': self.metadata[idx]
        }


def train_video_fixed(
    data_dir,
    output_dir,
    num_epochs=10,
    batch_size=1,
    num_frames=16,
    image_size=64,
    learning_rate=1e-4,
    device='cpu',
    spatial_checkpoint_path=None,
    checkpoint_interval=10,
    seed=42
):
    """Train video diffusion model with explicit spatial protection."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    train_dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=num_frames, image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")

    # Load tokenizer
    tokenizer_path = os.path.join(data_dir, 'train', '..', '..', 'text_image', 'tokenizer', 'tokenizer.json')

    if os.path.exists(tokenizer_path):
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
    else:
        print("Tokenizer not found, creating simple word-based tokenizer")
        all_words = set()
        for caption in train_dataset.captions:
            all_words.update(caption.lower().split())

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

    # Load spatial checkpoint
    if spatial_checkpoint_path and os.path.exists(spatial_checkpoint_path):
        print(f"Loading spatial checkpoint from {spatial_checkpoint_path}")
        checkpoint = torch.load(spatial_checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        spatial_unet.load_state_dict(state_dict, strict=False)

        if 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    else:
        print("No spatial checkpoint provided, training from scratch")

    # Create video U-Net with frozen spatial backbone
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=True,  # Freeze spatial backbone
        use_temporal=True
    )

    # Verify spatial is frozen
    spatial_params = list(video_unet.spatial_unet.parameters())
    frozen_count = sum(1 for p in spatial_params if not p.requires_grad)
    print(f"Spatial parameters frozen: {frozen_count}/{len(spatial_params)}")

    if frozen_count != len(spatial_params):
        raise RuntimeError(f"Expected all {len(spatial_params)} spatial parameters to be frozen, but only {frozen_count} are frozen")

    # Save initial spatial state for validation
    initial_spatial_state = {name: p.clone() for name, p in video_unet.spatial_unet.named_parameters()}

    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        ),
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': vocab_size
    })()

    video_unet = video_unet.to(device)

    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # EXPLICITLY create optimizer with only temporal and fusion parameters
    # This is the critical fix
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.bottleneck_fusion.parameters()) + list(video_unet.low_res_fusion.parameters())

    # Verify none of these are spatial parameters
    spatial_param_ids = set(id(p) for p in spatial_params)
    temporal_param_ids = set(id(p) for p in temporal_params)
    fusion_param_ids = set(id(p) for p in fusion_params)

    overlap_temporal = spatial_param_ids.intersection(temporal_param_ids)
    overlap_fusion = spatial_param_ids.intersection(fusion_param_ids)

    if overlap_temporal:
        raise RuntimeError(f"Temporal parameters include {len(overlap_temporal)} spatial parameters!")
    if overlap_fusion:
        raise RuntimeError(f"Fusion parameters include {len(overlap_fusion)} spatial parameters!")

    optimizer = optim.Adam([
        {'params': temporal_params, 'lr': learning_rate, 'weight_decay': 1e-5},
        {'params': fusion_params, 'lr': learning_rate, 'weight_decay': 1e-6}
    ])

    print(f"Optimizer parameters:")
    print(f"  Temporal: {len(temporal_params)}")
    print(f"  Fusion: {len(fusion_params)}")
    print(f"  Total: {len(temporal_params) + len(fusion_params)}")

    # Training loop
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        video_unet.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}")
        for batch in pbar:
            videos = batch['video'].to(device)
            captions = batch['caption']

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

            text_emb_encoded, _ = video_unet.text_encoder(text_emb, text_mask)
            text_emb = text_emb_encoded

            timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

            videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
            timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)

            noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)
            noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

            predicted_noise = video_unet(noisy_videos, timesteps, text_emb, text_mask)
            predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)

            loss = nn.functional.mse_loss(predicted_noise_flat, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # VALIDATE: Check that spatial parameters didn't change
            if global_step % 5 == 0:
                max_change = 0.0
                changed_params = 0
                for name, p in video_unet.spatial_unet.named_parameters():
                    if name in initial_spatial_state:
                        change = (p - initial_spatial_state[name]).abs().max().item()
                        if change > max_change:
                            max_change = change
                        if change > 1e-8:
                            changed_params += 1

                if changed_params > 0:
                    print(f"\n[WARNING] Step {global_step}: {changed_params} spatial parameters changed!")
                    print(f"  Max change: {max_change:.8f}")
                    # This should never happen if freeze_spatial=True works

            train_loss += loss.item()
            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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

    # Final validation: ensure spatial parameters still match initial state
    print("\nFinal validation: checking spatial parameters...")
    max_change = 0.0
    changed_params = 0
    for name, p in video_unet.spatial_unet.named_parameters():
        if name in initial_spatial_state:
            change = (p - initial_spatial_state[name]).abs().max().item()
            if change > max_change:
                max_change = change
            if change > 1e-8:
                changed_params += 1

    print(f"Spatial parameter changes after training:")
    print(f"  Changed parameters: {changed_params}/{len(initial_spatial_state)}")
    print(f"  Max change: {max_change:.8f}")

    if changed_params > 0:
        print("[ERROR] Spatial parameters changed during training!")
    else:
        print("[OK] Spatial parameters unchanged - Stage 2.1 performance preserved")

    final_checkpoint_path = os.path.join(output_dir, 'checkpoint_final.pt')
    torch.save({
        'step': global_step,
        'model_state_dict': video_unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
        'spatial_preserved': changed_params == 0,
    }, final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train video diffusion model (fixed)")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_video", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints_video_fixed", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--spatial_checkpoint", type=str, default="../checkpoints_conditional/checkpoint_final.pt", help="Spatial checkpoint")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Checkpoint interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_video_fixed(
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
