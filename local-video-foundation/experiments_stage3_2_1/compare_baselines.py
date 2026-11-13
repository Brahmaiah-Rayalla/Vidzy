"""
Compare frame-independent vs temporal model baselines for Stage 3.2.2.

This script compares:
A. Frame-independent spatial baseline (use_temporal=False)
B. Temporal model with untrained temporal modules
C. Trained temporal checkpoint
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


def load_models(device='cpu'):
    """Load models for comparison."""
    # Load dataset for tokenizer
    data_dir = "data/controlled_video"
    dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=16, image_size=64)

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

    # Create base models
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

    # Load checkpoint if available
    checkpoint_path = "checkpoints_controlled/checkpoint_best.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        spatial_unet.load_state_dict(state_dict, strict=False)
        if 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

    return spatial_unet, text_encoder, word_to_idx


def evaluate_model(model, dataloader, diffusion, device, name):
    """Evaluate a single model."""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
            videos = batch['video'].to(device)
            captions = batch['caption']
            batch_size = videos.shape[0]
            num_frames = 16
            image_size = 64

            # Encode captions
            text_embs = []
            text_masks = []
            for caption in captions:
                token_ids, mask = model.tokenizer.encode(caption, max_length=32)
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
            text_emb_encoded, _ = model.text_encoder(text_emb, text_mask)
            text_emb = text_emb_encoded

            # Sample timesteps
            timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

            # Forward diffusion
            videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
            timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)

            noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)
            noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

            # Predict noise
            predicted_noise = model(noisy_videos, timesteps, text_emb, text_mask)
            predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)

            loss = nn.functional.mse_loss(predicted_noise_flat, noise)
            losses.append(loss.item())

    return np.mean(losses), np.std(losses)


def main():
    print("=" * 80)
    print("BASELINE COMPARISON: Frame-Independent vs Temporal")
    print("=" * 80)

    device = 'cpu'
    data_dir = "data/controlled_video"

    # Load base models
    spatial_unet, text_encoder, word_to_idx = load_models(device)

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # Load dataset
    dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=16, image_size=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    results = {}

    # A. Frame-independent baseline (use_temporal=False)
    print("\nA. Creating frame-independent baseline...")
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
    video_unet_frame_independent = video_unet_frame_independent.to(device)

    mean_loss, std_loss = evaluate_model(video_unet_frame_independent, dataloader, diffusion, device, "Frame-Independent")
    results['frame_independent'] = {'mean': mean_loss, 'std': std_loss}
    print(f"   Frame-Independent Loss: {mean_loss:.4f} +/- {std_loss:.4f}")

    # B. Temporal model with untrained temporal modules
    print("\nB. Creating temporal model with untrained modules...")
    # Reload fresh spatial unet
    spatial_unet_untrained = ConditionalUNet(
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
    text_encoder_untrained = TextEncoder(
        vocab_size=len(word_to_idx),
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    video_unet_untrained = VideoUNet(
        spatial_unet=spatial_unet_untrained,
        text_encoder=text_encoder_untrained,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )
    video_unet_untrained.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()
    video_unet_untrained = video_unet_untrained.to(device)

    mean_loss, std_loss = evaluate_model(video_unet_untrained, dataloader, diffusion, device, "Untrained Temporal")
    results['untrained_temporal'] = {'mean': mean_loss, 'std': std_loss}
    print(f"   Untrained Temporal Loss: {mean_loss:.4f} +/- {std_loss:.4f}")

    # C. Trained temporal checkpoint
    print("\nC. Loading trained temporal checkpoint...")
    video_unet_trained = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )
    video_unet_trained.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()

    checkpoint_path = "checkpoints_controlled/checkpoint_best.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        video_unet_trained.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"   Loaded checkpoint from {checkpoint_path}")

    video_unet_trained = video_unet_trained.to(device)

    mean_loss, std_loss = evaluate_model(video_unet_trained, dataloader, diffusion, device, "Trained Temporal")
    results['trained_temporal'] = {'mean': mean_loss, 'std': std_loss}
    print(f"   Trained Temporal Loss: {mean_loss:.4f} +/- {std_loss:.4f}")

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nFrame-Independent:  {results['frame_independent']['mean']:.4f} +/- {results['frame_independent']['std']:.4f}")
    print(f"Untrained Temporal: {results['untrained_temporal']['mean']:.4f} +/- {results['untrained_temporal']['std']:.4f}")
    print(f"Trained Temporal:   {results['trained_temporal']['mean']:.4f} +/- {results['trained_temporal']['std']:.4f}")

    print("\nIMPROVEMENTS:")
    print(f"Untrained vs Frame-Independent: {results['untrained_temporal']['mean'] - results['frame_independent']['mean']:.4f}")
    print(f"Trained vs Frame-Independent:    {results['trained_temporal']['mean'] - results['frame_independent']['mean']:.4f}")
    print(f"Trained vs Untrained:            {results['trained_temporal']['mean'] - results['untrained_temporal']['mean']:.4f}")

    # Save results
    with open("baseline_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to baseline_comparison.json")


if __name__ == "__main__":
    main()
