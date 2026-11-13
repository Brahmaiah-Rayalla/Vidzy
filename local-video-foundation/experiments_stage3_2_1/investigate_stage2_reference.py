"""
Task 1: Establish the Stage 2.1 reference.

Load the original, validated Stage 2.1 image-generation checkpoint independently.
Verify that it still performs better than a zero-noise predictor.
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
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def investigate_stage2_reference(checkpoint_path, data_dir):
    print_section("TASK 1: ESTABLISH STAGE 2.1 REFERENCE")

    # Load checkpoint
    print(f"\n1. Loading Stage 2.1 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'unet_state_dict' in checkpoint:
        state_dict = checkpoint['unet_state_dict']
    else:
        state_dict = checkpoint
    
    print(f"State dict keys (first 20): {list(state_dict.keys())[:20]}")
    print(f"Total parameters in checkpoint: {len(state_dict)}")

    # Detect vocab size from checkpoint
    vocab_size = 24
    if 'text_encoder.token_emb.weight' in state_dict:
        vocab_size = state_dict['text_encoder.token_emb.weight'].shape[0]
        print(f"Detected vocab size from checkpoint: {vocab_size}")
    elif 'text_encoder_state_dict' in checkpoint and 'token_emb.weight' in checkpoint['text_encoder_state_dict']:
        vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]
        print(f"Detected vocab size from text_encoder_state_dict: {vocab_size}")

    # Create original Stage 2.1 model
    print("\n2. Creating original Stage 2.1 model...")
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
    print("\n3. Loading checkpoint into model...")
    missing, unexpected = spatial_unet.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    if len(missing) > 0:
        print(f"Sample missing keys: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"Sample unexpected keys: {unexpected[:5]}")

    if 'text_encoder_state_dict' in checkpoint:
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        print("Loaded text encoder state dict")

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    print_section("1.1 ZERO-PREDICTION BASELINE (STAGE 2.1)")

    # Load a sample image from conditional dataset
    print("\n4. Loading sample image...")
    try:
        from video_foundation.data.dataset import ImageDataset
        image_dataset = ImageDataset(data_dir, split='train', image_size=64)
        sample = image_dataset[0]
        image = sample['image'].unsqueeze(0)  # [1, 3, 64, 64]
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Image range: [{image.min().item():.6f}, {image.max().item():.6f}]")
    except Exception as e:
        print(f"Could not load image dataset: {e}")
        print("Using synthetic image...")
        image = torch.randn(1, 3, 64, 64)
        image = (image - image.min()) / (image.max() - image.min()) * 2 - 1  # Normalize to [-1, 1]

    # Use dummy text embedding
    text_emb = torch.randn(1, 32, 256)
    text_mask = torch.ones(1, 32)

    # Sample timestep
    t = torch.randint(0, 1000, (1,))
    print(f"\nTimestep: {t.item()}")

    # Forward diffusion
    noisy, noise = diffusion.q_sample(image, t)
    print(f"Noisy image shape: {noisy.shape}")
    print(f"Noise shape: {noise.shape}")
    print(f"Noise mean: {noise.mean().item():.6f}")
    print(f"Noise std: {noise.std().item():.6f}")

    # Zero-prediction baseline
    zero_prediction = torch.zeros_like(noise)
    zero_mse = nn.functional.mse_loss(zero_prediction, noise)

    print(f"\nZero-prediction MSE: {zero_mse.item():.6f}")

    print_section("1.2 STAGE 2.1 MODEL PREDICTION")

    # Stage 2.1 model prediction
    predicted_noise = spatial_unet(noisy, t, text_emb, text_mask)
    model_mse = nn.functional.mse_loss(predicted_noise, noise)

    print(f"\nPredicted noise shape: {predicted_noise.shape}")
    print(f"Predicted noise mean: {predicted_noise.mean().item():.6f}")
    print(f"Predicted noise std: {predicted_noise.std().item():.6f}")
    print(f"Model MSE: {model_mse.item():.6f}")
    print(f"Improvement over zero: {(zero_mse.item() - model_mse.item()) / zero_mse.item() * 100:.2f}%")

    # Correlation
    correlation = torch.corrcoef(torch.stack([
        noise.flatten(),
        predicted_noise.flatten()
    ]))[0, 1].item()
    print(f"Prediction-target correlation: {correlation:.6f}")

    print_section("1.3 SUMMARY")

    print("\nStage 2.1 reference validation:")
    print(f"  Zero-prediction MSE: {zero_mse.item():.6f}")
    print(f"  Stage 2.1 model MSE: {model_mse.item():.6f}")
    print(f"  Model better than zero: {'[YES]' if model_mse < zero_mse else '[NO]'}")
    print(f"  Correlation: {correlation:.6f}")

    results = {
        'checkpoint_path': checkpoint_path,
        'zero_prediction_mse': float(zero_mse.item()),
        'model_mse': float(model_mse.item()),
        'improvement_over_zero': float((zero_mse.item() - model_mse.item()) / zero_mse.item() * 100),
        'correlation': float(correlation),
        'model_better_than_zero': bool(model_mse < zero_mse),
        'missing_keys': len(missing),
        'unexpected_keys': len(unexpected),
    }

    with open("stage2_reference_validation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to stage2_reference_validation.json")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Investigate Stage 2.1 reference")
    parser.add_argument("--checkpoint_path", type=str, default="../checkpoints_conditional/checkpoint_final.pt", help="Stage 2.1 checkpoint path")
    parser.add_argument("--data_dir", type=str, default="../data/conditional_minimal", help="Data directory")
    args = parser.parse_args()

    results = investigate_stage2_reference(args.checkpoint_path, args.data_dir)


if __name__ == "__main__":
    main()
