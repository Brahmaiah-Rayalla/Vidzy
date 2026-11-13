"""
Resume Stage 3.2.5 training from step 400 to complete 500 steps.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


class SyntheticVideoDataset(Dataset):
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


def resume_training(
    data_dir,
    output_dir,
    resume_checkpoint_path,
    num_steps_to_run=100,
    batch_size=1,
    num_frames=16,
    image_size=64,
    learning_rate=1e-4,
    device='cpu',
    checkpoint_interval=50
):
    print("=" * 80)
    print("RESUMING STAGE 3.2.5 TRAINING")
    print("=" * 80)

    # Load checkpoint
    print(f"\nLoading checkpoint from {resume_checkpoint_path}")
    checkpoint = torch.load(resume_checkpoint_path, map_location='cpu')
    start_step = checkpoint['step']
    print(f"Resuming from step {start_step}")

    # Load datasets
    print("\nLoading datasets...")
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
        all_words = set()
        for caption in train_dataset.captions:
            all_words.update(caption.lower().split())

        word_to_idx = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        for i, word in enumerate(sorted(all_words), start=3):
            word_to_idx[word] = i

        idx_to_word = {v: k for k, v in word_to_idx.items()}
        vocab_size = len(word_to_idx)

    # Create models
    print("\nCreating models...")
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

    # Load Stage 2 checkpoint
    spatial_checkpoint_path = "../checkpoints_conditional/checkpoint_final.pt"
    spatial_checkpoint = torch.load(spatial_checkpoint_path, map_location='cpu')
    if 'model_state_dict' in spatial_checkpoint:
        spatial_unet.load_state_dict(spatial_checkpoint['model_state_dict'], strict=False)
    if 'text_encoder_state_dict' in spatial_checkpoint:
        text_encoder.load_state_dict(spatial_checkpoint['text_encoder_state_dict'])

    # Create video UNet
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=True,
        use_temporal=True
    )

    # Load video checkpoint
    video_unet.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Video checkpoint loaded")

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

    # Create optimizer
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.bottleneck_fusion.parameters()) + list(video_unet.low_res_fusion.parameters())

    optimizer = optim.Adam([
        {'params': temporal_params, 'lr': learning_rate, 'weight_decay': 1e-5},
        {'params': fusion_params, 'lr': learning_rate, 'weight_decay': 1e-6}
    ])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Optimizer state loaded")

    # Training loop
    print(f"\nTraining from step {start_step} to {start_step + num_steps_to_run}")
    global_step = start_step
    training_metrics = []

    while global_step < start_step + num_steps_to_run:
        video_unet.train()
        epoch_loss = 0.0
        steps_this_epoch = 0

        pbar = tqdm(train_loader, desc=f"Step {global_step}/{start_step + num_steps_to_run}")
        for batch in pbar:
            if global_step >= start_step + num_steps_to_run:
                break

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

            epoch_loss += loss.item()
            steps_this_epoch += 1
            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Save checkpoint
            if global_step % checkpoint_interval == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{global_step:06d}.pt')
                torch.save({
                    'step': global_step,
                    'model_state_dict': video_unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"\n[CHECKPOINT] Saved to {checkpoint_path}")

        avg_epoch_loss = epoch_loss / steps_this_epoch if steps_this_epoch > 0 else 0.0
        print(f"\nAverage loss: {avg_epoch_loss:.4f}")

    # Final checkpoint
    final_checkpoint_path = os.path.join(output_dir, 'checkpoint_final.pt')
    torch.save({
        'step': global_step,
        'model_state_dict': video_unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
    }, final_checkpoint_path)
    print(f"\n[FINAL] Saved to {final_checkpoint_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total steps: {global_step}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    resume_training(
        data_dir="../data/synthetic_video",
        output_dir="checkpoints_stage3_2_5",
        resume_checkpoint_path="checkpoints_stage3_2_5/checkpoint_step_000400.pt",
        num_steps_to_run=100,
        device='cpu'
    )
