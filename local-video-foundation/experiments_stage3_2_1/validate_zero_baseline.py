"""
Validate zero-prediction baseline with identical inputs for Stage 3.2.3 Task 3.

Compares zero prediction, frame-independent, and temporal models using the same inputs.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess
from video_foundation.training.train_video import SyntheticVideoDataset


def validate_zero_baseline(data_dir, checkpoint_path, num_samples=10):
    """Validate zero-prediction baseline with identical inputs."""
    print("=" * 80)
    print("ZERO-PREDICTION BASELINE VALIDATION (IDENTICAL INPUTS)")
    print("=" * 80)

    # Load dataset
    print("\nLoading dataset...")
    dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=16, image_size=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

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
        for caption in dataset.captions:
            all_words.update(caption.lower().split())
        word_to_idx = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        for i, word in enumerate(sorted(all_words), start=3):
            word_to_idx[word] = i
        idx_to_word = {v: k for k, v in word_to_idx.items()}
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

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        spatial_unet.load_state_dict(state_dict, strict=False)
        if 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

    # Create frame-independent model
    video_unet_frame_independent = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=False
    )
    video_unet_frame_independent.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()

    # Create temporal model
    video_unet_temporal = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )
    video_unet_temporal.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()

    if checkpoint_path and os.path.exists(checkpoint_path):
        video_unet_temporal.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # Evaluation
    print(f"\nEvaluating {num_samples} samples with identical inputs...")
    zero_losses = []
    frame_losses = []
    temporal_losses = []

    for i, batch in enumerate(tqdm(dataloader, total=num_samples)):
        if i >= num_samples:
            break

        videos = batch['video']
        captions = batch['caption']
        batch_size = videos.shape[0]
        num_frames = 16
        image_size = 64

        # Encode captions
        text_embs = []
        text_masks = []
        for caption in captions:
            token_ids, mask = video_unet_frame_independent.tokenizer.encode(caption, max_length=32)
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

        text_emb = torch.tensor(padded_embs)
        text_mask = torch.tensor(padded_masks)
        text_emb_encoded, _ = video_unet_frame_independent.text_encoder(text_emb, text_mask)
        text_emb = text_emb_encoded

        # Sample timesteps (same for all models)
        torch.manual_seed(i)
        timesteps = torch.randint(0, 1000, (batch_size,))

        # Forward diffusion (same for all models)
        videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)

        torch.manual_seed(i)
        noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)
        noisy_videos_reshaped = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

        # A. Zero prediction
        zero_pred = torch.zeros_like(noise)
        zero_loss = nn.functional.mse_loss(zero_pred, noise)
        zero_losses.append(zero_loss.item())

        # B. Frame-independent
        with torch.no_grad():
            frame_pred = video_unet_frame_independent(noisy_videos_reshaped, timesteps, text_emb, text_mask)
            frame_pred_flat = frame_pred.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
            frame_loss = nn.functional.mse_loss(frame_pred_flat, noise)
            frame_losses.append(frame_loss.item())

        # C. Temporal
        with torch.no_grad():
            temporal_pred = video_unet_temporal(noisy_videos_reshaped, timesteps, text_emb, text_mask)
            temporal_pred_flat = temporal_pred.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
            temporal_loss = nn.functional.mse_loss(temporal_pred_flat, noise)
            temporal_losses.append(temporal_loss.item())

    # Compute statistics
    zero_mean = np.mean(zero_losses)
    zero_std = np.std(zero_losses)
    frame_mean = np.mean(frame_losses)
    frame_std = np.std(frame_losses)
    temporal_mean = np.mean(temporal_losses)
    temporal_std = np.std(temporal_losses)

    print("\n" + "=" * 80)
    print("RESULTS (IDENTICAL INPUTS)")
    print("=" * 80)
    print(f"\nZero Prediction:  {zero_mean:.4f} +/- {zero_std:.4f}")
    print(f"Frame-Independent: {frame_mean:.4f} +/- {frame_std:.4f}")
    print(f"Temporal:         {temporal_mean:.4f} +/- {temporal_std:.4f}")

    print(f"\nComparisons:")
    print(f"  Frame vs Zero:  {frame_mean - zero_mean:+.4f} ({(frame_mean - zero_mean) / zero_mean * 100:+.1f}%)")
    print(f"  Temporal vs Zero: {temporal_mean - zero_mean:+.4f} ({(temporal_mean - zero_mean) / zero_mean * 100:+.1f}%)")
    print(f"  Temporal vs Frame: {temporal_mean - frame_mean:+.4f} ({(temporal_mean - frame_mean) / frame_mean * 100:+.1f}%)")

    print(f"\nStatus:")
    print(f"  Zero baseline ~1.0: {'[PASS]' if 0.9 < zero_mean < 1.1 else '[FAIL]'}")
    print(f"  Frame < Zero: {'[PASS]' if frame_mean < zero_mean else '[FAIL]'}")
    print(f"  Temporal < Zero: {'[PASS]' if temporal_mean < zero_mean else '[FAIL]'}")
    print(f"  Temporal < Frame: {'[PASS]' if temporal_mean < frame_mean else '[FAIL]'}")

    print("\n" + "=" * 80)

    return {
        'zero_mean': float(zero_mean),
        'zero_std': float(zero_std),
        'frame_mean': float(frame_mean),
        'frame_std': float(frame_std),
        'temporal_mean': float(temporal_mean),
        'temporal_std': float(temporal_std),
        'frame_vs_zero': float(frame_mean - zero_mean),
        'temporal_vs_zero': float(temporal_mean - zero_mean),
        'temporal_vs_frame': float(temporal_mean - frame_mean)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate zero-prediction baseline")
    parser.add_argument("--data_dir", type=str, default="data/controlled_video", help="Data directory")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints_controlled/checkpoint_best.pt", help="Checkpoint path")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples")
    args = parser.parse_args()

    results = validate_zero_baseline(args.data_dir, args.checkpoint_path, args.num_samples)

    with open("zero_baseline_validation.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to zero_baseline_validation.json")


if __name__ == "__main__":
    main()
