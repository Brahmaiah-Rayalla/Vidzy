"""
Diagnose temporal gate learning for Stage 3.2.3 Task 4.

Investigates why gate values remain constant at 0.5025.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess
from video_foundation.training.train_video import SyntheticVideoDataset


def diagnose_gate_learning(data_dir, checkpoint_path, num_steps=5):
    """Diagnose gate parameter learning."""
    print("=" * 80)
    print("TEMPORAL GATE LEARNING DIAGNOSIS")
    print("=" * 80)

    # Load dataset
    print("\nLoading dataset...")
    dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=16, image_size=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load checkpoint first to detect vocab size
    checkpoint = None
    vocab_size = 24  # Default
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        if 'text_encoder.token_emb.weight' in state_dict:
            vocab_size = state_dict['text_encoder.token_emb.weight'].shape[0]
            print(f"Detected vocab size from checkpoint: {vocab_size}")

    # Load tokenizer - but use checkpoint vocab size, not tokenizer vocab size
    # Get absolute path to data directory
    abs_data_dir = os.path.abspath(data_dir)
    # Go up to parent data directory
    parent_data_dir = os.path.dirname(abs_data_dir)
    tokenizer_path = os.path.join(parent_data_dir, 'text_image', 'tokenizer', 'tokenizer.json')
    word_to_idx = {}
    idx_to_word = {}
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)
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
        # Use checkpoint vocab size, not tokenizer size
        # vocab_size = len(word_to_idx)  # Don't override checkpoint vocab size
    else:
        all_words = set()
        for caption in dataset.captions:
            all_words.update(caption.lower().split())
        word_to_idx = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        for i, word in enumerate(sorted(all_words), start=3):
            word_to_idx[word] = i
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        # vocab_size = len(word_to_idx)  # Don't override checkpoint vocab size

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

    # Load checkpoint into models
    if checkpoint is not None:
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        spatial_unet.load_state_dict(state_dict, strict=False)
        if 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )
    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()

    if checkpoint_path and os.path.exists(checkpoint_path):
        video_unet.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Create optimizer
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.bottleneck_fusion.parameters()) + list(video_unet.low_res_fusion.parameters())

    # Use equal learning rates (Stage 3.2.3 fix)
    optimizer = torch.optim.Adam([
        {'params': temporal_params, 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': fusion_params, 'lr': 1e-4, 'weight_decay': 1e-6}
    ])

    print(f"\nParameter Groups:")
    print(f"  Temporal params: {len(temporal_params)} parameters, lr=1e-4")
    print(f"  Fusion params: {len(fusion_params)} parameters, lr=1e-4")

    # Check gate parameters
    print(f"\nInitial Gate Parameters:")
    print(f"  Bottleneck gate raw: {video_unet.bottleneck_fusion.gate.item():.6f}")
    print(f"  Bottleneck gate activated: {video_unet.bottleneck_fusion.gate_activation(video_unet.bottleneck_fusion.gate).item():.6f}")
    print(f"  Low-res gate raw: {video_unet.low_res_fusion.gate.item():.6f}")
    print(f"  Low-res gate activated: {video_unet.low_res_fusion.gate_activation(video_unet.low_res_fusion.gate).item():.6f}")

    # Check if gates are trainable
    print(f"\nTrainable Status:")
    print(f"  Bottleneck gate requires_grad: {video_unet.bottleneck_fusion.gate.requires_grad}")
    print(f"  Low-res gate requires_grad: {video_unet.low_res_fusion.gate.requires_grad}")

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # Training steps
    print(f"\nRunning {num_steps} training steps...")
    gate_log = []

    for step in range(num_steps):
        batch = next(iter(dataloader))
        videos = batch['video']
        captions = batch['caption']
        batch_size = videos.shape[0]
        num_frames = 16
        image_size = 64

        # Encode captions - use dummy embeddings to avoid tokenizer mismatch
        # Since we're just diagnosing gate learning, we don't need real text embeddings
        text_emb = torch.randn(batch_size, 32, 256)
        text_mask = torch.ones(batch_size, 32)

        # Sample timesteps
        timesteps = torch.randint(0, 1000, (batch_size,))

        # Forward diffusion
        videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)

        noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)
        noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

        # Forward pass
        predicted_noise = video_unet(noisy_videos, timesteps, text_emb, text_mask)
        predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)

        # Compute loss
        loss = nn.functional.mse_loss(predicted_noise_flat, noise)

        # Record gate values before backward
        bottleneck_gate_before = video_unet.bottleneck_fusion.gate.item()
        low_res_gate_before = video_unet.low_res_fusion.gate.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Record gradients
        bottleneck_gate_grad = video_unet.bottleneck_fusion.gate.grad.item() if video_unet.bottleneck_fusion.gate.grad is not None else 0.0
        low_res_gate_grad = video_unet.low_res_fusion.gate.grad.item() if video_unet.low_res_fusion.gate.grad is not None else 0.0

        # Optimizer step
        optimizer.step()

        # Record gate values after optimizer
        bottleneck_gate_after = video_unet.bottleneck_fusion.gate.item()
        low_res_gate_after = video_unet.low_res_fusion.gate.item()

        gate_log.append({
            'step': step,
            'bottleneck_gate_before': bottleneck_gate_before,
            'bottleneck_gate_after': bottleneck_gate_after,
            'bottleneck_gate_grad': bottleneck_gate_grad,
            'bottleneck_gate_change': bottleneck_gate_after - bottleneck_gate_before,
            'low_res_gate_before': low_res_gate_before,
            'low_res_gate_after': low_res_gate_after,
            'low_res_gate_grad': low_res_gate_grad,
            'low_res_gate_change': low_res_gate_after - low_res_gate_before,
            'loss': loss.item()
        })

        print(f"  Step {step}:")
        print(f"    Bottleneck gate: {bottleneck_gate_before:.6f} -> {bottleneck_gate_after:.6f} (delta: {bottleneck_gate_after - bottleneck_gate_before:.8f})")
        print(f"    Bottleneck grad: {bottleneck_gate_grad:.8f}")
        print(f"    Low-res gate: {low_res_gate_before:.6f} -> {low_res_gate_after:.6f} (delta: {low_res_gate_after - low_res_gate_before:.8f})")
        print(f"    Low-res grad: {low_res_gate_grad:.8f}")

    print("\n" + "=" * 80)
    print("GATE LEARNING ANALYSIS")
    print("=" * 80)

    total_bottleneck_change = sum(log['bottleneck_gate_change'] for log in gate_log)
    total_low_res_change = sum(log['low_res_gate_change'] for log in gate_log)
    avg_bottleneck_grad = np.mean([log['bottleneck_gate_grad'] for log in gate_log])
    avg_low_res_grad = np.mean([log['low_res_gate_grad'] for log in gate_log])

    print(f"\nTotal change over {num_steps} steps:")
    print(f"  Bottleneck gate: {total_bottleneck_change:.8f}")
    print(f"  Low-res gate: {total_low_res_change:.8f}")

    print(f"\nAverage gradient magnitude:")
    print(f"  Bottleneck gate: {avg_bottleneck_grad:.8f}")
    print(f"  Low-res gate: {avg_low_res_grad:.8f}")

    print(f"\nExpected change per step (lr * grad):")
    print(f"  Bottleneck: {1e-4 * avg_bottleneck_grad:.8f}")
    print(f"  Low-res: {1e-4 * avg_low_res_grad:.8f}")

    print(f"\nStatus:")
    print(f"  Gates are updating: {'[YES]' if abs(total_bottleneck_change) > 1e-8 or abs(total_low_res_change) > 1e-8 else '[NO]'}")
    print(f"  Gradients are non-zero: {'[YES]' if abs(avg_bottleneck_grad) > 1e-8 or abs(avg_low_res_grad) > 1e-8 else '[NO]'}")

    print("\n" + "=" * 80)

    return {
        'total_bottleneck_change': float(total_bottleneck_change),
        'total_low_res_change': float(total_low_res_change),
        'avg_bottleneck_grad': float(avg_bottleneck_grad),
        'avg_low_res_grad': float(avg_low_res_grad),
        'gate_log': gate_log
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose gate learning")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_video", help="Data directory")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints_controlled/checkpoint_best.pt", help="Checkpoint path")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of training steps")
    args = parser.parse_args()

    results = diagnose_gate_learning(args.data_dir, args.checkpoint_path, args.num_steps)

    with open("gate_learning_diagnosis.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to gate_learning_diagnosis.json")


if __name__ == "__main__":
    main()
