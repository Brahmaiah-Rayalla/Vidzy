"""
Compare frame-independent video model with Stage 2.1 reference.

Investigate why the frame-independent baseline performs worse than zero-prediction
when the same spatial backbone performs excellently in Stage 2.1.
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


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def compare_frame_independent(
    stage2_checkpoint_path,
    video_checkpoint_path,
    data_dir
):
    print_section("COMPARISON: FRAME-INDEPENDENT VS STAGE 2.1")

    # Detect vocab size from Stage 2 checkpoint
    print(f"\n1. Loading Stage 2.1 checkpoint: {stage2_checkpoint_path}")
    stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location='cpu')

    if 'text_encoder_state_dict' in stage2_checkpoint and 'token_emb.weight' in stage2_checkpoint['text_encoder_state_dict']:
        vocab_size = stage2_checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]
        print(f"Detected vocab size: {vocab_size}")
    else:
        vocab_size = 42
        print(f"Using default vocab size: {vocab_size}")

    # Create Stage 2.1 model
    print("\n2. Creating Stage 2.1 model...")
    stage2_spatial = ConditionalUNet(
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

    stage2_text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Load Stage 2 checkpoint
    if 'model_state_dict' in stage2_checkpoint:
        stage2_spatial.load_state_dict(stage2_checkpoint['model_state_dict'], strict=False)
    if 'text_encoder_state_dict' in stage2_checkpoint:
        stage2_text_encoder.load_state_dict(stage2_checkpoint['text_encoder_state_dict'])
    print("Stage 2.1 checkpoint loaded")

    # Create video model with same spatial backbone
    print("\n3. Creating video model (frame-independent mode)...")
    video_spatial = ConditionalUNet(
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

    video_text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Load Stage 2 checkpoint into video spatial backbone
    if 'model_state_dict' in stage2_checkpoint:
        video_spatial.load_state_dict(stage2_checkpoint['model_state_dict'], strict=False)
    if 'text_encoder_state_dict' in stage2_checkpoint:
        video_text_encoder.load_state_dict(stage2_checkpoint['text_encoder_state_dict'])

    # Create video U-Net (frame-independent)
    video_unet = VideoUNet(
        spatial_unet=video_spatial,
        text_encoder=video_text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=False,  # Don't freeze for comparison
        use_temporal=False  # Frame-independent mode
    )

    # Load video checkpoint if available
    if video_checkpoint_path and os.path.exists(video_checkpoint_path):
        print(f"\n4. Loading video checkpoint: {video_checkpoint_path}")
        video_checkpoint = torch.load(video_checkpoint_path, map_location='cpu')
        if 'model_state_dict' in video_checkpoint:
            video_unet.load_state_dict(video_checkpoint['model_state_dict'], strict=False)
        print("Video checkpoint loaded")
    else:
        print("\n4. No video checkpoint provided, using Stage 2 weights only")

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    print_section("5. GENERATE TEST DATA")

    # Create synthetic video data
    print("\nGenerating synthetic video data...")
    batch_size = 1
    num_frames = 16
    image_size = 64

    # Generate a single clean video
    clean_video = torch.randn(batch_size, 3, num_frames, image_size, image_size)
    clean_video = (clean_video - clean_video.min()) / (clean_video.max() - clean_video.min()) * 2 - 1

    print(f"Clean video shape: {clean_video.shape}")
    print(f"Clean video range: [{clean_video.min().item():.6f}, {clean_video.max().item():.6f}]")

    # Use dummy text embedding
    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)

    # Sample timestep
    t = torch.randint(0, 1000, (batch_size,))
    print(f"\nTimestep: {t.item()}")

    print_section("6. STAGE 2.1 MODEL (SINGLE FRAME)")

    # Test Stage 2.1 on a single frame
    single_frame = clean_video[:, :, 0, :, :]  # [1, 3, 64, 64]
    print(f"Single frame shape: {single_frame.shape}")

    # Forward diffusion
    noisy_frame, noise_frame = diffusion.q_sample(single_frame, t)
    print(f"Noisy frame shape: {noisy_frame.shape}")

    # Stage 2.1 prediction
    stage2_pred = stage2_spatial(noisy_frame, t, text_emb, text_mask)
    stage2_mse = nn.functional.mse_loss(stage2_pred, noise_frame)

    print(f"\nStage 2.1 single-frame MSE: {stage2_mse.item():.6f}")
    print(f"Stage 2.1 prediction mean: {stage2_pred.mean().item():.6f}")
    print(f"Stage 2.1 prediction std: {stage2_pred.std().item():.6f}")

    # Correlation
    correlation_stage2 = torch.corrcoef(torch.stack([
        noise_frame.flatten(),
        stage2_pred.flatten()
    ]))[0, 1].item()
    print(f"Stage 2.1 correlation: {correlation_stage2:.6f}")

    print_section("7. VIDEO MODEL (FRAME-INDEPENDENT)")

    # Test video model on same frame (frame-independent mode)
    # Need to expand to video format
    video_input = clean_video[:, :, :1, :, :]  # [1, 3, 1, 64, 64]
    print(f"Video input shape: {video_input.shape}")

    # Forward diffusion (frame-by-frame)
    video_flat = video_input.permute(0, 2, 1, 3, 4).reshape(batch_size * 1, 3, image_size, image_size)
    t_flat = t.unsqueeze(1).expand(-1, 1).reshape(batch_size * 1)
    noisy_video, noise_video = diffusion.q_sample(video_flat, t_flat)
    noisy_video = noisy_video.reshape(batch_size, 1, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

    # Video model prediction (frame-independent)
    video_pred = video_unet(noisy_video, t, text_emb, text_mask)
    video_pred_flat = video_pred.permute(0, 2, 1, 3, 4).reshape(batch_size * 1, 3, image_size, image_size)
    video_mse = nn.functional.mse_loss(video_pred_flat, noise_video)

    print(f"\nVideo model frame-independent MSE: {video_mse.item():.6f}")
    print(f"Video prediction mean: {video_pred.mean().item():.6f}")
    print(f"Video prediction std: {video_pred.std().item():.6f}")

    # Correlation
    correlation_video = torch.corrcoef(torch.stack([
        noise_video.flatten(),
        video_pred_flat.flatten()
    ]))[0, 1].item()
    print(f"Video model correlation: {correlation_video:.6f}")

    print_section("8. COMPARISON")

    print("\nPerformance comparison:")
    print(f"  Stage 2.1 (single frame):    MSE = {stage2_mse.item():.6f}, correlation = {correlation_stage2:.6f}")
    print(f"  Video model (frame-indep):   MSE = {video_mse.item():.6f}, correlation = {correlation_video:.6f}")
    print(f"  Difference:                  MSE = {abs(stage2_mse.item() - video_mse.item()):.6f}")

    if abs(stage2_mse.item() - video_mse.item()) < 0.01:
        print("\n[OK] Frame-independent model matches Stage 2.1 performance")
    else:
        print("\n[ISSUE] Frame-independent model differs from Stage 2.1")
        print("This suggests a problem with how the video model processes frames")

    print_section("9. PARAMETER COMPARISON")

    # Compare weights
    print("\nComparing spatial backbone weights...")
    stage2_params = dict(stage2_spatial.named_parameters())
    video_params = dict(video_spatial.named_parameters())

    total_params = len(stage2_params)
    matching_params = 0
    mismatching_params = 0

    for name in stage2_params.keys():
        if name in video_params:
            if torch.allclose(stage2_params[name], video_params[name], rtol=1e-5, atol=1e-8):
                matching_params += 1
            else:
                mismatching_params += 1
                if mismatching_params <= 3:
                    diff = (stage2_params[name] - video_params[name]).abs().max().item()
                    print(f"  Mismatching: {name}, max diff = {diff:.8f}")

    print(f"\nParameter comparison:")
    print(f"  Total parameters: {total_params}")
    print(f"  Matching: {matching_params}")
    print(f"  Mismatching: {mismatching_params}")

    results = {
        'stage2_mse': float(stage2_mse.item()),
        'video_mse': float(video_mse.item()),
        'stage2_correlation': float(correlation_stage2),
        'video_correlation': float(correlation_video),
        'mse_difference': float(abs(stage2_mse.item() - video_mse.item())),
        'matching_params': matching_params,
        'mismatching_params': mismatching_params,
        'total_params': total_params,
    }

    with open("frame_independent_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to frame_independent_comparison.json")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare frame-independent with Stage 2.1")
    parser.add_argument("--stage2_checkpoint", type=str, default="../checkpoints_conditional/checkpoint_final.pt", help="Stage 2.1 checkpoint")
    parser.add_argument("--video_checkpoint", type=str, default=None, help="Video checkpoint (optional)")
    parser.add_argument("--data_dir", type=str, default="../data/synthetic_video", help="Data directory")
    args = parser.parse_args()

    results = compare_frame_independent(args.stage2_checkpoint, args.video_checkpoint, args.data_dir)


if __name__ == "__main__":
    main()
