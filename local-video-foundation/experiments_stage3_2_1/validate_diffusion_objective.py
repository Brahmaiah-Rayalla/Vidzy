"""
Task 3: Validate the diffusion objective.

Compare zero-prediction and trained-model MSE using exactly the same:
- Video batch
- Random noise tensor
- Diffusion timestep
- Normalization
- Loss reduction

Measure:
- Target noise mean and standard deviation
- Predicted noise mean and standard deviation
- Zero-prediction MSE
- Frame-independent model MSE
- Temporal model MSE
- Prediction-target correlation
- Loss across timestep ranges

Investigate why previous measurements reported losses above 1.0 while the latest
10-step experiment reports a loss of 0.0922.
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


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def validate_diffusion_objective(data_dir, checkpoint_path, num_samples=5):
    print_section("TASK 3: VALIDATE DIFFUSION OBJECTIVE")

    # Load dataset
    print("\n1. Loading dataset...")
    dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=16, image_size=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"   Loaded {len(dataset)} videos")

    # Detect vocab size from checkpoint
    checkpoint = None
    vocab_size = 24
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n2. Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        if 'text_encoder.token_emb.weight' in state_dict:
            vocab_size = state_dict['text_encoder.token_emb.weight'].shape[0]
            print(f"   Detected vocab size: {vocab_size}")

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

    # Load checkpoint into models
    if checkpoint is not None:
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        spatial_unet.load_state_dict(state_dict, strict=False)
        if 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

    # Create video U-Net with temporal modules
    video_unet_temporal = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    # Create video U-Net without temporal modules (frame-independent)
    video_unet_frame_independent = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=False
    )

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    print_section("3.1 ZERO-PREDICTION BASELINE")

    # Use same batch, noise, and timestep for all comparisons
    batch = next(iter(dataloader))
    videos = batch['video']
    batch_size = videos.shape[0]
    num_frames = 16
    image_size = 64

    # Use dummy text embeddings
    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)

    # Sample timesteps
    timesteps = torch.randint(0, 1000, (batch_size,))
    print(f"\nTimestep: {timesteps.item()}")

    # Forward diffusion
    videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
    timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)

    noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)
    noisy_videos_reshaped = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

    # Zero-prediction baseline
    zero_prediction = torch.zeros_like(noise)
    zero_mse = nn.functional.mse_loss(zero_prediction, noise)

    print(f"\nZero-prediction baseline:")
    print(f"  Target noise mean: {noise.mean().item():.6f}")
    print(f"  Target noise std: {noise.std().item():.6f}")
    print(f"  Zero-prediction MSE: {zero_mse.item():.6f}")

    print_section("3.2 FRAME-INDEPENDENT MODEL")

    # Frame-independent model prediction
    predicted_noise_frame_independent = video_unet_frame_independent(
        noisy_videos_reshaped, timesteps, text_emb, text_mask
    )
    predicted_noise_frame_independent_flat = predicted_noise_frame_independent.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames, 3, image_size, image_size
    )

    frame_independent_mse = nn.functional.mse_loss(predicted_noise_frame_independent_flat, noise)

    print(f"\nFrame-independent model:")
    print(f"  Predicted noise mean: {predicted_noise_frame_independent_flat.mean().item():.6f}")
    print(f"  Predicted noise std: {predicted_noise_frame_independent_flat.std().item():.6f}")
    print(f"  Frame-independent MSE: {frame_independent_mse.item():.6f}")
    print(f"  Improvement over zero: {(zero_mse.item() - frame_independent_mse.item()) / zero_mse.item() * 100:.2f}%")

    # Correlation
    correlation = torch.corrcoef(torch.stack([
        noise.flatten(),
        predicted_noise_frame_independent_flat.flatten()
    ]))[0, 1].item()
    print(f"  Prediction-target correlation: {correlation:.6f}")

    print_section("3.3 TEMPORAL MODEL")

    # Temporal model prediction
    predicted_noise_temporal = video_unet_temporal(
        noisy_videos_reshaped, timesteps, text_emb, text_mask
    )
    predicted_noise_temporal_flat = predicted_noise_temporal.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames, 3, image_size, image_size
    )

    temporal_mse = nn.functional.mse_loss(predicted_noise_temporal_flat, noise)

    print(f"\nTemporal model:")
    print(f"  Predicted noise mean: {predicted_noise_temporal_flat.mean().item():.6f}")
    print(f"  Predicted noise std: {predicted_noise_temporal_flat.std().item():.6f}")
    print(f"  Temporal MSE: {temporal_mse.item():.6f}")
    print(f"  Improvement over zero: {(zero_mse.item() - temporal_mse.item()) / zero_mse.item() * 100:.2f}%")
    print(f"  Improvement over frame-independent: {(frame_independent_mse.item() - temporal_mse.item()) / frame_independent_mse.item() * 100:.2f}%")

    # Correlation
    correlation_temporal = torch.corrcoef(torch.stack([
        noise.flatten(),
        predicted_noise_temporal_flat.flatten()
    ]))[0, 1].item()
    print(f"  Prediction-target correlation: {correlation_temporal:.6f}")

    print_section("3.4 LOSS ACROSS TIMESTEP RANGES")

    # Sample multiple timesteps to check loss across ranges
    timestep_ranges = [
        (0, 250, "low"),
        (250, 500, "mid-low"),
        (500, 750, "mid-high"),
        (750, 1000, "high")
    ]

    print("\nSampling losses across timestep ranges...")
    losses_by_range = {}

    for t_min, t_max, range_name in timestep_ranges:
        range_losses = []
        for _ in range(5):
            t = torch.randint(t_min, t_max, (batch_size,))
            t_flat = t.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)
            
            noisy, target_noise = diffusion.q_sample(videos_flat, t_flat)
            noisy_reshaped = noisy.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)
            
            pred = video_unet_temporal(noisy_reshaped, t, text_emb, text_mask)
            pred_flat = pred.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
            
            loss = nn.functional.mse_loss(pred_flat, target_noise)
            range_losses.append(loss.item())
        
        avg_loss = np.mean(range_losses)
        losses_by_range[range_name] = {
            'range': [t_min, t_max],
            'avg_loss': float(avg_loss),
            'std': float(np.std(range_losses))
        }
        print(f"  {range_name} ({t_min}-{t_max}): {avg_loss:.6f} +/- {np.std(range_losses):.6f}")

    print_section("3.5 LOSS INVESTIGATION")

    print("\nInvestigating loss discrepancy:")
    print(f"  Previous reported losses: > 1.0")
    print(f"  Latest 10-step experiment loss: 0.0922")
    print(f"  Current zero-prediction MSE: {zero_mse.item():.6f}")
    print(f"  Current frame-independent MSE: {frame_independent_mse.item():.6f}")
    print(f"  Current temporal MSE: {temporal_mse.item():.6f}")
    
    print("\nPossible explanations:")
    print("  1. Different evaluation conditions (batch size, dataset)")
    print("  2. Different loss reduction (mean vs sum)")
    print("  3. Different normalization")
    print("  4. Different checkpoint (trained vs untrained)")
    print("  5. Random variation in timestep sampling")

    print_section("3.6 SUMMARY")

    print("\nDiffusion objective validation:")
    print(f"  Zero-prediction MSE: {zero_mse.item():.6f}")
    print(f"  Frame-independent MSE: {frame_independent_mse.item():.6f}")
    print(f"  Temporal MSE: {temporal_mse.item():.6f}")
    print(f"  Temporal better than frame-independent: {'[YES]' if temporal_mse < frame_independent_mse else '[NO]'}")
    print(f"  Frame-independent better than zero: {'[YES]' if frame_independent_mse < zero_mse else '[NO]'}")

    results = {
        'zero_prediction': {
            'mse': float(zero_mse.item()),
            'target_noise_mean': float(noise.mean().item()),
            'target_noise_std': float(noise.std().item()),
        },
        'frame_independent': {
            'mse': float(frame_independent_mse.item()),
            'predicted_noise_mean': float(predicted_noise_frame_independent_flat.mean().item()),
            'predicted_noise_std': float(predicted_noise_frame_independent_flat.std().item()),
            'correlation': float(correlation),
            'improvement_over_zero': float((zero_mse.item() - frame_independent_mse.item()) / zero_mse.item() * 100),
        },
        'temporal': {
            'mse': float(temporal_mse.item()),
            'predicted_noise_mean': float(predicted_noise_temporal_flat.mean().item()),
            'predicted_noise_std': float(predicted_noise_temporal_flat.std().item()),
            'correlation': float(correlation_temporal),
            'improvement_over_zero': float((zero_mse.item() - temporal_mse.item()) / zero_mse.item() * 100),
            'improvement_over_frame_independent': float((frame_independent_mse.item() - temporal_mse.item()) / frame_independent_mse.item() * 100),
        },
        'losses_by_range': losses_by_range,
        'timestep': int(timesteps.item()),
    }

    with open("diffusion_objective_validation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to diffusion_objective_validation.json")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate diffusion objective")
    parser.add_argument("--data_dir", type=str, default="../data/synthetic_video", help="Data directory")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints_gate_lr_fix/checkpoint_final.pt", help="Checkpoint path")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
    args = parser.parse_args()

    results = validate_diffusion_objective(args.data_dir, args.checkpoint_path, args.num_samples)


if __name__ == "__main__":
    main()
