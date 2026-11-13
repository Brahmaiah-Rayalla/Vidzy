"""
Task 1: Verify gate parameterization in detail.

Inspects LearnableTemporalFusion implementation and verifies:
- Raw trainable parameter
- Activated gate value
- Effective residual scaling
- Actual temporal feature contribution
- Optimizer inclusion
- Gradient flow
- Parameter updates
- Numerical measurability
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
from video_foundation.models.video_unet import VideoUNet, LearnableTemporalFusion
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess
from video_foundation.training.train_video import SyntheticVideoDataset


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def verify_gate_parameterization(data_dir, checkpoint_path, num_steps=5):
    print_section("TASK 1: VERIFY GATE PARAMETERIZATION")

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

    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    print("\n4. Inspecting LearnableTemporalFusion implementation...")
    print_section("4.1 GATE ARCHITECTURE")
    
    # Inspect bottleneck fusion
    print("\nBottleneck Fusion Module:")
    print(f"  Type: {type(video_unet.bottleneck_fusion)}")
    print(f"  Use channel-wise: {video_unet.bottleneck_fusion.use_channel_wise}")
    print(f"  Gate activation: {type(video_unet.bottleneck_fusion.gate_activation)}")
    print(f"  Gate parameter shape: {video_unet.bottleneck_fusion.gate.shape}")
    print(f"  Gate parameter dtype: {video_unet.bottleneck_fusion.gate.dtype}")
    print(f"  Gate parameter device: {video_unet.bottleneck_fusion.gate.device}")
    
    # Initial gate values (full precision)
    bottleneck_gate_raw = video_unet.bottleneck_fusion.gate.item()
    bottleneck_gate_activated = video_unet.bottleneck_fusion.gate_activation(
        video_unet.bottleneck_fusion.gate
    ).item()
    
    print(f"\n  Raw gate parameter: {bottleneck_gate_raw:.15f}")
    print(f"  Activated gate (sigmoid): {bottleneck_gate_activated:.15f}")
    print(f"  Sigmoid(0.01) = {torch.sigmoid(torch.tensor(0.01)).item():.15f}")
    
    # Inspect low-res fusion
    print("\nLow-Res Fusion Module:")
    print(f"  Type: {type(video_unet.low_res_fusion)}")
    print(f"  Use channel-wise: {video_unet.low_res_fusion.use_channel_wise}")
    print(f"  Gate activation: {type(video_unet.low_res_fusion.gate_activation)}")
    print(f"  Gate parameter shape: {video_unet.low_res_fusion.gate.shape}")
    print(f"  Gate parameter dtype: {video_unet.low_res_fusion.gate.dtype}")
    print(f"  Gate parameter device: {video_unet.low_res_fusion.gate.device}")
    
    low_res_gate_raw = video_unet.low_res_fusion.gate.item()
    low_res_gate_activated = video_unet.low_res_fusion.gate_activation(
        video_unet.low_res_fusion.gate
    ).item()
    
    print(f"\n  Raw gate parameter: {low_res_gate_raw:.15f}")
    print(f"  Activated gate (sigmoid): {low_res_gate_activated:.15f}")
    print(f"  Sigmoid(0.01) = {torch.sigmoid(torch.tensor(0.01)).item():.15f}")

    print_section("4.2 OPTIMIZER INCLUSION")
    
    # Create optimizer
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.bottleneck_fusion.parameters()) + list(video_unet.low_res_fusion.parameters())
    
    optimizer = torch.optim.Adam([
        {'params': temporal_params, 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': fusion_params, 'lr': 1e-4, 'weight_decay': 1e-6}
    ])
    
    print(f"\nTemporal parameters: {len(temporal_params)}")
    print(f"Fusion parameters: {len(fusion_params)}")
    print(f"Fusion parameters include:")
    for i, p in enumerate(fusion_params):
        print(f"  Param {i}: shape={p.shape}, requires_grad={p.requires_grad}")
    
    # Check if gates are in optimizer
    bottleneck_gate_in_optimizer = any(
        p is video_unet.bottleneck_fusion.gate for p in optimizer.param_groups[1]['params']
    )
    low_res_gate_in_optimizer = any(
        p is video_unet.low_res_fusion.gate for p in optimizer.param_groups[1]['params']
    )
    
    print(f"\nBottleneck gate in optimizer: {bottleneck_gate_in_optimizer}")
    print(f"Low-res gate in optimizer: {low_res_gate_in_optimizer}")

    print_section("4.3 GRADIENT FLOW VERIFICATION")
    
    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)
    
    # Single forward/backward pass to check gradients
    print("\nRunning single forward/backward pass...")
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
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    print(f"\nLoss: {loss.item():.15f}")
    print(f"\nBottleneck gate gradient:")
    print(f"  Has gradient: {video_unet.bottleneck_fusion.gate.grad is not None}")
    if video_unet.bottleneck_fusion.gate.grad is not None:
        print(f"  Gradient value: {video_unet.bottleneck_fusion.gate.grad.item():.15f}")
        print(f"  Gradient magnitude: {abs(video_unet.bottleneck_fusion.gate.grad.item()):.15f}")
    
    print(f"\nLow-res gate gradient:")
    print(f"  Has gradient: {video_unet.low_res_fusion.gate.grad is not None}")
    if video_unet.low_res_fusion.gate.grad is not None:
        print(f"  Gradient value: {video_unet.low_res_fusion.gate.grad.item():.15f}")
        print(f"  Gradient magnitude: {abs(video_unet.low_res_fusion.gate.grad.item()):.15f}")

    print_section("4.4 PARAMETER UPDATE VERIFICATION")
    
    # Record values before optimizer step
    bottleneck_gate_before = video_unet.bottleneck_fusion.gate.item()
    low_res_gate_before = video_unet.low_res_fusion.gate.item()
    
    # Optimizer step
    optimizer.step()
    
    # Record values after optimizer step
    bottleneck_gate_after = video_unet.bottleneck_fusion.gate.item()
    low_res_gate_after = video_unet.low_res_fusion.gate.item()
    
    bottleneck_gate_delta = bottleneck_gate_after - bottleneck_gate_before
    low_res_gate_delta = low_res_gate_after - low_res_gate_before
    
    print(f"\nBottleneck gate:")
    print(f"  Before: {bottleneck_gate_before:.15f}")
    print(f"  After:  {bottleneck_gate_after:.15f}")
    print(f"  Delta:  {bottleneck_gate_delta:.15f}")
    print(f"  Numerically measurable: {abs(bottleneck_gate_delta) > 1e-10}")
    
    print(f"\nLow-res gate:")
    print(f"  Before: {low_res_gate_before:.15f}")
    print(f"  After:  {low_res_gate_after:.15f}")
    print(f"  Delta:  {low_res_gate_delta:.15f}")
    print(f"  Numerically measurable: {abs(low_res_gate_delta) > 1e-10}")

    print_section("4.5 EFFECTIVE TEMPORAL CONTRIBUTION")
    
    # Measure temporal feature contribution during forward pass
    print("\nMeasuring temporal feature contribution...")
    
    # Hook to capture temporal features
    bottleneck_temporal_features = []
    low_res_temporal_features = []
    
    def hook_bottleneck(module, input, output):
        bottleneck_temporal_features.append(output.detach())
    
    def hook_low_res(module, input, output):
        low_res_temporal_features.append(output.detach())
    
    video_unet.temporal_modules['bottleneck'].register_forward_hook(hook_bottleneck)
    video_unet.temporal_modules['low_res'].register_forward_hook(hook_low_res)
    
    # Forward pass again
    optimizer.zero_grad()
    noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)
    noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)
    predicted_noise = video_unet(noisy_videos, timesteps, text_emb, text_mask)
    
    # Compute temporal contribution
    if len(bottleneck_temporal_features) > 0:
        bottleneck_temporal = bottleneck_temporal_features[0]
        print(f"\nBottleneck temporal features:")
        print(f"  Shape: {bottleneck_temporal.shape}")
        print(f"  Mean: {bottleneck_temporal.mean().item():.15f}")
        print(f"  Std: {bottleneck_temporal.std().item():.15f}")
        print(f"  Min: {bottleneck_temporal.min().item():.15f}")
        print(f"  Max: {bottleneck_temporal.max().item():.15f}")
    
    if len(low_res_temporal_features) > 0:
        low_res_temporal = low_res_temporal_features[0]
        print(f"\nLow-res temporal features:")
        print(f"  Shape: {low_res_temporal.shape}")
        print(f"  Mean: {low_res_temporal.mean().item():.15f}")
        print(f"  Std: {low_res_temporal.std().item():.15f}")
        print(f"  Min: {low_res_temporal.min().item():.15f}")
        print(f"  Max: {low_res_temporal.max().item():.15f}")
    
    # Compute effective temporal contribution to final prediction
    # Gate * temporal_features * projection
    bottleneck_gate_current = video_unet.bottleneck_fusion.gate_activation(
        video_unet.bottleneck_fusion.gate
    ).item()
    low_res_gate_current = video_unet.low_res_fusion.gate_activation(
        video_unet.low_res_fusion.gate
    ).item()
    
    print(f"\nEffective temporal contribution scaling:")
    print(f"  Bottleneck gate (activated): {bottleneck_gate_current:.15f}")
    print(f"  Low-res gate (activated): {low_res_gate_current:.15f}")
    print(f"  Combined effective scaling: {bottleneck_gate_current + low_res_gate_current:.15f}")

    print_section("4.6 TRAINING DYNAMICS")
    
    # Run multiple steps to track changes
    print(f"\nRunning {num_steps} training steps...")
    gate_log = []
    
    for step in range(num_steps):
        batch = next(iter(dataloader))
        videos = batch['video']
        batch_size = videos.shape[0]
        
        text_emb = torch.randn(batch_size, 32, 256)
        text_mask = torch.ones(batch_size, 32)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)
        
        noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)
        noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)
        
        predicted_noise = video_unet(noisy_videos, timesteps, text_emb, text_mask)
        predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        
        loss = nn.functional.mse_loss(predicted_noise_flat, noise)
        
        # Record before backward
        bottleneck_gate_raw_before = video_unet.bottleneck_fusion.gate.item()
        low_res_gate_raw_before = video_unet.low_res_fusion.gate.item()
        bottleneck_gate_activated_before = video_unet.bottleneck_fusion.gate_activation(
            video_unet.bottleneck_fusion.gate
        ).item()
        low_res_gate_activated_before = video_unet.low_res_fusion.gate_activation(
            video_unet.low_res_fusion.gate
        ).item()
        
        optimizer.zero_grad()
        loss.backward()
        
        # Record gradients
        bottleneck_gate_grad = video_unet.bottleneck_fusion.gate.grad.item() if video_unet.bottleneck_fusion.gate.grad is not None else 0.0
        low_res_gate_grad = video_unet.low_res_fusion.gate.grad.item() if video_unet.low_res_fusion.gate.grad is not None else 0.0
        
        optimizer.step()
        
        # Record after optimizer
        bottleneck_gate_raw_after = video_unet.bottleneck_fusion.gate.item()
        low_res_gate_raw_after = video_unet.low_res_fusion.gate.item()
        bottleneck_gate_activated_after = video_unet.bottleneck_fusion.gate_activation(
            video_unet.bottleneck_fusion.gate
        ).item()
        low_res_gate_activated_after = video_unet.low_res_fusion.gate_activation(
            video_unet.low_res_fusion.gate
        ).item()
        
        log_entry = {
            'step': int(step),
            'loss': float(loss.item()),
            'bottleneck_gate_raw_before': float(bottleneck_gate_raw_before),
            'bottleneck_gate_raw_after': float(bottleneck_gate_raw_after),
            'bottleneck_gate_activated_before': float(bottleneck_gate_activated_before),
            'bottleneck_gate_activated_after': float(bottleneck_gate_activated_after),
            'bottleneck_gate_grad': float(bottleneck_gate_grad),
            'bottleneck_gate_raw_delta': float(bottleneck_gate_raw_after - bottleneck_gate_raw_before),
            'bottleneck_gate_activated_delta': float(bottleneck_gate_activated_after - bottleneck_gate_activated_before),
            'low_res_gate_raw_before': float(low_res_gate_raw_before),
            'low_res_gate_raw_after': float(low_res_gate_raw_after),
            'low_res_gate_activated_before': float(low_res_gate_activated_before),
            'low_res_gate_activated_after': float(low_res_gate_activated_after),
            'low_res_gate_grad': float(low_res_gate_grad),
            'low_res_gate_raw_delta': float(low_res_gate_raw_after - low_res_gate_raw_before),
            'low_res_gate_activated_delta': float(low_res_gate_activated_after - low_res_gate_activated_before),
        }
        gate_log.append(log_entry)
        
        print(f"  Step {step}:")
        print(f"    Loss: {loss.item():.15f}")
        print(f"    Bottleneck raw: {bottleneck_gate_raw_before:.15f} -> {bottleneck_gate_raw_after:.15f} (delta: {bottleneck_gate_raw_after - bottleneck_gate_raw_before:.15f})")
        print(f"    Bottleneck activated: {bottleneck_gate_activated_before:.15f} -> {bottleneck_gate_activated_after:.15f} (delta: {bottleneck_gate_activated_after - bottleneck_gate_activated_before:.15f})")
        print(f"    Low-res raw: {low_res_gate_raw_before:.15f} -> {low_res_gate_raw_after:.15f} (delta: {low_res_gate_raw_after - low_res_gate_raw_before:.15f})")
        print(f"    Low-res activated: {low_res_gate_activated_before:.15f} -> {low_res_gate_activated_after:.15f} (delta: {low_res_gate_activated_after - low_res_gate_activated_before:.15f})")

    print_section("4.7 SUMMARY")
    
    total_bottleneck_raw_change = sum(log['bottleneck_gate_raw_delta'] for log in gate_log)
    total_low_res_raw_change = sum(log['low_res_gate_raw_delta'] for log in gate_log)
    total_bottleneck_activated_change = sum(log['bottleneck_gate_activated_delta'] for log in gate_log)
    total_low_res_activated_change = sum(log['low_res_gate_activated_delta'] for log in gate_log)
    avg_bottleneck_grad = np.mean([log['bottleneck_gate_grad'] for log in gate_log])
    avg_low_res_grad = np.mean([log['low_res_gate_grad'] for log in gate_log])
    
    print(f"\nTotal change over {num_steps} steps:")
    print(f"  Bottleneck raw: {total_bottleneck_raw_change:.15f}")
    print(f"  Bottleneck activated: {total_bottleneck_activated_change:.15f}")
    print(f"  Low-res raw: {total_low_res_raw_change:.15f}")
    print(f"  Low-res activated: {total_low_res_activated_change:.15f}")
    
    print(f"\nAverage gradient magnitude:")
    print(f"  Bottleneck: {avg_bottleneck_grad:.15f}")
    print(f"  Low-res: {avg_low_res_grad:.15f}")
    
    print(f"\nVerification results:")
    print(f"  Gates in optimizer: {'[PASS]' if bottleneck_gate_in_optimizer and low_res_gate_in_optimizer else '[FAIL]'}")
    print(f"  Gradients non-zero: {'[PASS]' if abs(avg_bottleneck_grad) > 1e-10 or abs(avg_low_res_grad) > 1e-10 else '[FAIL]'}")
    print(f"  Parameter updates measurable: {'[PASS]' if abs(total_bottleneck_raw_change) > 1e-10 or abs(total_low_res_raw_change) > 1e-10 else '[FAIL]'}")
    print(f"  Effective contribution changes: {'[PASS]' if abs(total_bottleneck_activated_change) > 1e-10 or abs(total_low_res_activated_change) > 1e-10 else '[FAIL]'}")

    results = {
        'gate_parameterization': {
            'bottleneck': {
                'raw_initial': float(bottleneck_gate_raw),
                'activated_initial': float(bottleneck_gate_activated),
                'in_optimizer': bool(bottleneck_gate_in_optimizer),
            },
            'low_res': {
                'raw_initial': float(low_res_gate_raw),
                'activated_initial': float(low_res_gate_activated),
                'in_optimizer': bool(low_res_gate_in_optimizer),
            }
        },
        'gradient_verification': {
            'bottleneck_grad_avg': float(avg_bottleneck_grad),
            'low_res_grad_avg': float(avg_low_res_grad),
            'gradients_nonzero': bool(abs(avg_bottleneck_grad) > 1e-10 or abs(avg_low_res_grad) > 1e-10),
        },
        'update_verification': {
            'bottleneck_total_raw_change': float(total_bottleneck_raw_change),
            'low_res_total_raw_change': float(total_low_res_raw_change),
            'bottleneck_total_activated_change': float(total_bottleneck_activated_change),
            'low_res_total_activated_change': float(total_low_res_activated_change),
            'updates_measurable': bool(abs(total_bottleneck_raw_change) > 1e-10 or abs(total_low_res_raw_change) > 1e-10),
        },
        'training_log': gate_log
    }
    
    with open("gate_parameterization_verification.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to gate_parameterization_verification.json")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify gate parameterization")
    parser.add_argument("--data_dir", type=str, default="../data/synthetic_video", help="Data directory")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints_gate_lr_fix/checkpoint_final.pt", help="Checkpoint path")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of training steps")
    args = parser.parse_args()

    results = verify_gate_parameterization(args.data_dir, args.checkpoint_path, args.num_steps)


if __name__ == "__main__":
    main()
