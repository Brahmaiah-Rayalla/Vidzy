"""
Validate the fixed video checkpoint against zero-prediction baseline.

Test that the frame-independent and temporal modes now perform correctly.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

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


def validate_fixed_checkpoint(
    video_checkpoint_path,
    stage2_checkpoint_path
):
    print_section("VALIDATE FIXED VIDEO CHECKPOINT")

    # Load Stage 2 checkpoint for vocab size
    print(f"\n1. Loading Stage 2 checkpoint: {stage2_checkpoint_path}")
    stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location='cpu')

    if 'text_encoder_state_dict' in stage2_checkpoint and 'token_emb.weight' in stage2_checkpoint['text_encoder_state_dict']:
        vocab_size = stage2_checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]
        print(f"Detected vocab size: {vocab_size}")
    else:
        vocab_size = 42
        print(f"Using default vocab size: {vocab_size}")

    # Load video checkpoint
    print(f"\n2. Loading video checkpoint: {video_checkpoint_path}")
    video_checkpoint = torch.load(video_checkpoint_path, map_location='cpu')

    # Create models
    print("\n3. Creating models...")
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

    # Load Stage 2 weights
    if 'model_state_dict' in stage2_checkpoint:
        spatial_unet.load_state_dict(stage2_checkpoint['model_state_dict'], strict=False)
    if 'text_encoder_state_dict' in stage2_checkpoint:
        text_encoder.load_state_dict(stage2_checkpoint['text_encoder_state_dict'])

    # Create video UNet
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=False,  # Load all weights
        use_temporal=True
    )

    # Load video checkpoint
    if 'model_state_dict' in video_checkpoint:
        video_unet.load_state_dict(video_checkpoint['model_state_dict'], strict=False)
    print("Video checkpoint loaded")

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    print_section("4. GENERATE TEST DATA")

    # Generate synthetic video
    batch_size = 1
    num_frames = 16
    image_size = 64

    clean_video = torch.randn(batch_size, 3, num_frames, image_size, image_size)
    clean_video = (clean_video - clean_video.min()) / (clean_video.max() - clean_video.min()) * 2 - 1

    print(f"Clean video shape: {clean_video.shape}")

    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)

    t = torch.randint(0, 1000, (batch_size,))
    print(f"Timestep: {t.item()}")

    print_section("5. ZERO-PREDICTION BASELINE")

    # Forward diffusion
    videos_flat = clean_video.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
    t_flat = t.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)
    noisy_videos, noise = diffusion.q_sample(videos_flat, t_flat)
    noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

    # Zero prediction
    zero_prediction = torch.zeros_like(noisy_videos)
    zero_prediction_flat = zero_prediction.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
    zero_mse = nn.functional.mse_loss(zero_prediction_flat, noise)

    print(f"Zero-prediction MSE: {zero_mse.item():.6f}")

    print_section("6. FRAME-INDEPENDENT MODE")

    # Test frame-independent mode
    video_unet.use_temporal = False
    predicted_noise_frame_indep = video_unet(noisy_videos, t, text_emb, text_mask)
    predicted_noise_frame_indep_flat = predicted_noise_frame_indep.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
    frame_indep_mse = nn.functional.mse_loss(predicted_noise_frame_indep_flat, noise)

    correlation_frame_indep = torch.corrcoef(torch.stack([
        noise.flatten(),
        predicted_noise_frame_indep_flat.flatten()
    ]))[0, 1].item()

    print(f"Frame-independent MSE: {frame_indep_mse.item():.6f}")
    print(f"Improvement over zero: {(zero_mse.item() - frame_indep_mse.item()) / zero_mse.item() * 100:.2f}%")
    print(f"Correlation: {correlation_frame_indep:.6f}")

    print_section("7. TEMPORAL MODE")

    # Test temporal mode
    video_unet.use_temporal = True
    predicted_noise_temporal = video_unet(noisy_videos, t, text_emb, text_mask)
    predicted_noise_temporal_flat = predicted_noise_temporal.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
    temporal_mse = nn.functional.mse_loss(predicted_noise_temporal_flat, noise)

    correlation_temporal = torch.corrcoef(torch.stack([
        noise.flatten(),
        predicted_noise_temporal_flat.flatten()
    ]))[0, 1].item()

    print(f"Temporal MSE: {temporal_mse.item():.6f}")
    print(f"Improvement over zero: {(zero_mse.item() - temporal_mse.item()) / zero_mse.item() * 100:.2f}%")
    print(f"Correlation: {correlation_temporal:.6f}")

    # Check gate values
    bottleneck_gate = video_unet.bottleneck_fusion.gate_activation(video_unet.bottleneck_fusion.gate).item()
    low_res_gate = video_unet.low_res_fusion.gate_activation(video_unet.low_res_fusion.gate).item()

    print(f"\nGate values:")
    print(f"  Bottleneck gate: {bottleneck_gate:.6f}")
    print(f"  Low-res gate: {low_res_gate:.6f}")

    print_section("8. SUMMARY")

    print("\nPerformance comparison:")
    print(f"  Zero-prediction:       MSE = {zero_mse.item():.6f}")
    print(f"  Frame-independent:     MSE = {frame_indep_mse.item():.6f}, correlation = {correlation_frame_indep:.6f}")
    print(f"  Temporal:              MSE = {temporal_mse.item():.6f}, correlation = {correlation_temporal:.6f}")

    if frame_indep_mse < zero_mse and temporal_mse < zero_mse:
        print("\n[SUCCESS] Both modes perform better than zero-prediction")
    elif frame_indep_mse < zero_mse:
        print("\n[PARTIAL] Frame-independent mode works, temporal mode needs more training")
    else:
        print("\n[FAILURE] Both modes perform worse than zero-prediction")

    results = {
        'zero_mse': float(zero_mse.item()),
        'frame_independent_mse': float(frame_indep_mse.item()),
        'temporal_mse': float(temporal_mse.item()),
        'frame_independent_correlation': float(correlation_frame_indep),
        'temporal_correlation': float(correlation_temporal),
        'bottleneck_gate': float(bottleneck_gate),
        'low_res_gate': float(low_res_gate),
        'frame_independent_better_than_zero': bool(frame_indep_mse < zero_mse),
        'temporal_better_than_zero': bool(temporal_mse < zero_mse),
    }

    with open("fixed_checkpoint_validation.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to fixed_checkpoint_validation.json")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate fixed checkpoint")
    parser.add_argument("--video_checkpoint", type=str, default="checkpoints_video_test/checkpoint_final.pt", help="Video checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str, default="../checkpoints_conditional/checkpoint_final.pt", help="Stage 2 checkpoint")
    args = parser.parse_args()

    results = validate_fixed_checkpoint(args.video_checkpoint, args.stage2_checkpoint)


if __name__ == "__main__":
    main()
