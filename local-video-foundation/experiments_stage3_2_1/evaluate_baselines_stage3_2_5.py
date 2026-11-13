"""
Stage 3.2.5: Matched Baseline Evaluation

Compares all baselines on identical data with identical conditions:
A. Zero-noise predictor
B. Original Stage 2.1 spatial model
C. Fixed spatial video model with temporal modules disabled
D. Video model with temporal modules enabled
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


def evaluate_baselines(
    stage2_checkpoint_path,
    video_checkpoint_path=None,
    num_samples=10,
    num_frames=16,
    image_size=64,
    device='cpu'
):
    print_section("STAGE 3.2.5 MATCHED BASELINE EVALUATION")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load Stage 2 checkpoint
    print(f"\n[1] Loading Stage 2 checkpoint: {stage2_checkpoint_path}")
    stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location='cpu')
    
    if 'text_encoder_state_dict' in stage2_checkpoint and 'token_emb.weight' in stage2_checkpoint['text_encoder_state_dict']:
        vocab_size = stage2_checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]
    else:
        vocab_size = 42
    
    print(f"Detected vocab size: {vocab_size}")
    
    # Create Stage 2.1 model
    print("\n[2] Creating Stage 2.1 model...")
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
    
    # Load Stage 2 weights
    if 'model_state_dict' in stage2_checkpoint:
        stage2_spatial.load_state_dict(stage2_checkpoint['model_state_dict'], strict=False)
    if 'text_encoder_state_dict' in stage2_checkpoint:
        stage2_text_encoder.load_state_dict(stage2_checkpoint['text_encoder_state_dict'])
    
    print("Stage 2.1 model loaded")
    
    # Create video model with same spatial backbone
    print("\n[3] Creating video model...")
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
    
    # Load Stage 2 weights into video spatial backbone
    if 'model_state_dict' in stage2_checkpoint:
        video_spatial.load_state_dict(stage2_checkpoint['model_state_dict'], strict=False)
    if 'text_encoder_state_dict' in stage2_checkpoint:
        video_text_encoder.load_state_dict(stage2_checkpoint['text_encoder_state_dict'])
    
    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=video_spatial,
        text_encoder=video_text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=False,
        use_temporal=True
    )
    
    # Load video checkpoint if provided
    if video_checkpoint_path and os.path.exists(video_checkpoint_path):
        print(f"\n[4] Loading video checkpoint: {video_checkpoint_path}")
        video_checkpoint = torch.load(video_checkpoint_path, map_location='cpu')
        if 'model_state_dict' in video_checkpoint:
            video_unet.load_state_dict(video_checkpoint['model_state_dict'], strict=False)
        print("Video checkpoint loaded")
    else:
        print("\n[4] No video checkpoint provided, using Stage 2 weights only")
    
    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)
    
    print_section("5. GENERATE CONSISTENT TEST DATA")
    
    # Generate consistent test data (same for all baselines)
    batch_size = num_samples
    
    clean_video = torch.randn(batch_size, 3, num_frames, image_size, image_size)
    clean_video = (clean_video - clean_video.min()) / (clean_video.max() - clean_video.min()) * 2 - 1
    
    print(f"Clean video shape: {clean_video.shape}")
    print(f"Clean video range: [{clean_video.min().item():.6f}, {clean_video.max().item():.6f}]")
    
    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)
    
    t = torch.randint(0, 1000, (batch_size,))
    print(f"\nTimesteps: {t.tolist()}")
    
    # Forward diffusion (same for all baselines)
    videos_flat = clean_video.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
    t_flat = t.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)
    noisy_videos, noise = diffusion.q_sample(videos_flat, t_flat)
    noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)
    
    print(f"Noisy video shape: {noisy_videos.shape}")
    print(f"Noise shape: {noise.shape}")
    print(f"Noise mean: {noise.mean().item():.6f}")
    print(f"Noise std: {noise.std().item():.6f}")
    
    results = {}
    
    print_section("6. BASELINE A: ZERO-NOISE PREDICTOR")
    
    zero_prediction = torch.zeros_like(noisy_videos)
    zero_prediction_flat = zero_prediction.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
    zero_mse = nn.functional.mse_loss(zero_prediction_flat, noise)
    
    print(f"Zero-prediction MSE: {zero_mse.item():.6f}")
    results['zero_mse'] = float(zero_mse.item())
    
    print_section("7. BASELINE B: ORIGINAL STAGE 2.1 SPATIAL MODEL")
    
    # Test Stage 2.1 on single frames
    stage2_mses = []
    stage2_correlations = []
    
    for i in range(num_frames):
        single_frame = clean_video[:, :, i, :, :]
        noisy_frame, noise_frame = diffusion.q_sample(single_frame, t)
        
        stage2_pred = stage2_spatial(noisy_frame, t, text_emb, text_mask)
        stage2_mse = nn.functional.mse_loss(stage2_pred, noise_frame)
        
        correlation = torch.corrcoef(torch.stack([
            noise_frame.flatten(),
            stage2_pred.flatten()
        ]))[0, 1].item()
        
        stage2_mses.append(stage2_mse.item())
        stage2_correlations.append(correlation)
    
    avg_stage2_mse = np.mean(stage2_mses)
    avg_stage2_correlation = np.mean(stage2_correlations)
    
    print(f"Stage 2.1 average MSE (per frame): {avg_stage2_mse:.6f}")
    print(f"Stage 2.1 average correlation: {avg_stage2_correlation:.6f}")
    print(f"Stage 2.1 MSE range: [{np.min(stage2_mses):.6f}, {np.max(stage2_mses):.6f}]")
    
    results['stage2_mse'] = float(avg_stage2_mse)
    results['stage2_correlation'] = float(avg_stage2_correlation)
    results['stage2_mse_std'] = float(np.std(stage2_mses))
    
    print_section("8. BASELINE C: VIDEO MODEL (TEMPORAL DISABLED)")
    
    video_unet.use_temporal = False
    video_unet.eval()
    
    with torch.no_grad():
        predicted_noise_frame_indep = video_unet(noisy_videos, t, text_emb, text_mask)
        predicted_noise_frame_indep_flat = predicted_noise_frame_indep.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        frame_indep_mse = nn.functional.mse_loss(predicted_noise_frame_indep_flat, noise)
        
        correlation_frame_indep = torch.corrcoef(torch.stack([
            noise.flatten(),
            predicted_noise_frame_indep_flat.flatten()
        ]))[0, 1].item()
    
    print(f"Frame-independent MSE: {frame_indep_mse.item():.6f}")
    print(f"Frame-independent correlation: {correlation_frame_indep:.6f}")
    print(f"Predicted noise mean: {predicted_noise_frame_indep.mean().item():.6f}")
    print(f"Predicted noise std: {predicted_noise_frame_indep.std().item():.6f}")
    
    results['frame_independent_mse'] = float(frame_indep_mse.item())
    results['frame_independent_correlation'] = float(correlation_frame_indep)
    
    print_section("9. BASELINE D: VIDEO MODEL (TEMPORAL ENABLED)")
    
    video_unet.use_temporal = True
    video_unet.eval()
    
    with torch.no_grad():
        predicted_noise_temporal = video_unet(noisy_videos, t, text_emb, text_mask)
        predicted_noise_temporal_flat = predicted_noise_temporal.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        temporal_mse = nn.functional.mse_loss(predicted_noise_temporal_flat, noise)
        
        correlation_temporal = torch.corrcoef(torch.stack([
            noise.flatten(),
            predicted_noise_temporal_flat.flatten()
        ]))[0, 1].item()
    
    print(f"Temporal MSE: {temporal_mse.item():.6f}")
    print(f"Temporal correlation: {correlation_temporal:.6f}")
    print(f"Predicted noise mean: {predicted_noise_temporal.mean().item():.6f}")
    print(f"Predicted noise std: {predicted_noise_temporal.std().item():.6f}")
    
    # Gate values
    bottleneck_gate = video_unet.bottleneck_fusion.gate_activation(video_unet.bottleneck_fusion.gate).item()
    low_res_gate = video_unet.low_res_fusion.gate_activation(video_unet.low_res_fusion.gate).item()
    
    print(f"\nGate values:")
    print(f"  Bottleneck gate: {bottleneck_gate:.6f}")
    print(f"  Low-res gate: {low_res_gate:.6f}")
    
    # Temporal improvement
    improvement = (frame_indep_mse.item() - temporal_mse.item()) / frame_indep_mse.item() * 100
    print(f"Temporal improvement over frame-independent: {improvement:.2f}%")
    
    results['temporal_mse'] = float(temporal_mse.item())
    results['temporal_correlation'] = float(correlation_temporal)
    results['bottleneck_gate'] = float(bottleneck_gate)
    results['low_res_gate'] = float(low_res_gate)
    results['temporal_improvement_pct'] = float(improvement)
    
    print_section("10. SUMMARY")
    
    print("\nBaseline comparison (all on identical data):")
    print(f"  A. Zero-prediction:       MSE = {results['zero_mse']:.6f}")
    print(f"  B. Stage 2.1 spatial:     MSE = {results['stage2_mse']:.6f}, correlation = {results['stage2_correlation']:.6f}")
    print(f"  C. Frame-independent:     MSE = {results['frame_independent_mse']:.6f}, correlation = {results['frame_independent_correlation']:.6f}")
    print(f"  D. Temporal:              MSE = {results['temporal_mse']:.6f}, correlation = {results['temporal_correlation']:.6f}")
    
    print("\nImprovement over zero-prediction:")
    print(f"  Stage 2.1:     {(results['zero_mse'] - results['stage2_mse']) / results['zero_mse'] * 100:.2f}%")
    print(f"  Frame-indep:   {(results['zero_mse'] - results['frame_independent_mse']) / results['zero_mse'] * 100:.2f}%")
    print(f"  Temporal:      {(results['zero_mse'] - results['temporal_mse']) / results['zero_mse'] * 100:.2f}%")
    
    print("\nTemporal vs frame-independent:")
    print(f"  MSE difference: {results['frame_independent_mse'] - results['temporal_mse']:.6f}")
    print(f"  Improvement: {results['temporal_improvement_pct']:.2f}%")
    
    # Save results
    with open("baseline_evaluation_stage3_2_5.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to baseline_evaluation_stage3_2_5.json")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3.2.5 Baseline Evaluation")
    parser.add_argument("--stage2_checkpoint", type=str, default="../checkpoints_conditional/checkpoint_final.pt", help="Stage 2 checkpoint")
    parser.add_argument("--video_checkpoint", type=str, default=None, help="Video checkpoint (optional)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    results = evaluate_baselines(
        stage2_checkpoint_path=args.stage2_checkpoint,
        video_checkpoint_path=args.video_checkpoint,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        image_size=args.image_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
