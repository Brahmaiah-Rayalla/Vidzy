"""
Stage 3.2.6 Task 1: Audit Stage 3.2.5 Results - Verify Identical Evaluation Conditions

This script creates a deterministic evaluation fixture to ensure all baselines
are evaluated on EXACTLY the same data with EXACTLY the same conditions.
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


def generate_deterministic_test_data(num_samples=10, num_frames=16, image_size=64, seed=42):
    """
    Generate deterministic test data that can be reused across all evaluations.
    Returns a dictionary containing all tensors needed for evaluation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate clean video
    clean_video = torch.randn(num_samples, 3, num_frames, image_size, image_size)
    clean_video = (clean_video - clean_video.min()) / (clean_video.max() - clean_video.min()) * 2 - 1

    # Generate text embeddings
    text_emb = torch.randn(num_samples, 32, 256)
    text_mask = torch.ones(num_samples, 32)

    # Generate timesteps
    t = torch.randint(0, 1000, (num_samples,))

    return {
        'clean_video': clean_video,
        'text_emb': text_emb,
        'text_mask': text_mask,
        't': t,
        'num_samples': num_samples,
        'num_frames': num_frames,
        'image_size': image_size,
        'seed': seed
    }


def apply_diffusion_noise(test_data, diffusion):
    """
    Apply forward diffusion to the test data.
    This ensures all baselines see the exact same noisy video and noise target.
    """
    clean_video = test_data['clean_video']
    t = test_data['t']
    num_samples = test_data['num_samples']
    num_frames = test_data['num_frames']
    image_size = test_data['image_size']

    # Forward diffusion (same for all baselines)
    videos_flat = clean_video.permute(0, 2, 1, 3, 4).reshape(num_samples * num_frames, 3, image_size, image_size)
    t_flat = t.unsqueeze(1).expand(-1, num_frames).reshape(num_samples * num_frames)
    noisy_videos, noise = diffusion.q_sample(videos_flat, t_flat)
    noisy_videos = noisy_videos.reshape(num_samples, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

    return {
        'noisy_videos': noisy_videos,
        'noise': noise,
        't_flat': t_flat
    }


def evaluate_baseline_zero(noisy_videos, noise, num_samples, num_frames, image_size):
    """Baseline A: Zero-noise predictor"""
    zero_prediction = torch.zeros_like(noisy_videos)
    zero_prediction_flat = zero_prediction.permute(0, 2, 1, 3, 4).reshape(num_samples * num_frames, 3, image_size, image_size)
    zero_mse = nn.functional.mse_loss(zero_prediction_flat, noise)
    return float(zero_mse.item())


def evaluate_baseline_frame_independent(video_unet, noisy_videos, t, text_emb, text_mask, noise, num_samples, num_frames, image_size):
    """Baseline C: Video model with temporal disabled"""
    video_unet.use_temporal = False
    video_unet.eval()

    with torch.no_grad():
        predicted_noise = video_unet(noisy_videos, t, text_emb, text_mask)
        predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(num_samples * num_frames, 3, image_size, image_size)
        mse = nn.functional.mse_loss(predicted_noise_flat, noise)
        correlation = torch.corrcoef(torch.stack([
            noise.flatten(),
            predicted_noise_flat.flatten()
        ]))[0, 1].item()

    return float(mse.item()), float(correlation)


def evaluate_baseline_temporal(video_unet, noisy_videos, t, text_emb, text_mask, noise, num_samples, num_frames, image_size):
    """Baseline D: Video model with temporal enabled"""
    video_unet.use_temporal = True
    video_unet.eval()

    with torch.no_grad():
        predicted_noise = video_unet(noisy_videos, t, text_emb, text_mask)
        predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(num_samples * num_frames, 3, image_size, image_size)
        mse = nn.functional.mse_loss(predicted_noise_flat, noise)
        correlation = torch.corrcoef(torch.stack([
            noise.flatten(),
            predicted_noise_flat.flatten()
        ]))[0, 1].item()

        # Gate values
        bottleneck_gate = video_unet.bottleneck_fusion.gate_activation(video_unet.bottleneck_fusion.gate).item()
        low_res_gate = video_unet.low_res_fusion.gate_activation(video_unet.low_res_fusion.gate).item()

    return float(mse.item()), float(correlation), float(bottleneck_gate), float(low_res_gate)


def audit_stage3_2_5_evaluation(
    stage2_checkpoint_path,
    video_checkpoint_path,
    output_path,
    num_samples=10,
    num_frames=16,
    image_size=64,
    device='cpu'
):
    """
    Audit Stage 3.2.5 evaluation by running all baselines on identical data.
    """
    print_section("STAGE 3.2.6 TASK 1: AUDIT STAGE 3.2.5 EVALUATION CONDITIONS")

    # Load Stage 2 checkpoint
    print(f"\n[1] Loading Stage 2 checkpoint: {stage2_checkpoint_path}")
    stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location='cpu')

    if 'text_encoder_state_dict' in stage2_checkpoint and 'token_emb.weight' in stage2_checkpoint['text_encoder_state_dict']:
        vocab_size = stage2_checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]
    else:
        vocab_size = 42

    print(f"Detected vocab size: {vocab_size}")

    # Create models
    print("\n[2] Creating models...")
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

    print("Stage 2.1 model loaded")

    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=False,
        use_temporal=True
    )

    # Load video checkpoint
    print(f"\n[3] Loading video checkpoint: {video_checkpoint_path}")
    video_checkpoint = torch.load(video_checkpoint_path, map_location='cpu')
    if 'model_state_dict' in video_checkpoint:
        video_unet.load_state_dict(video_checkpoint['model_state_dict'], strict=False)
    print("Video checkpoint loaded")

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    print_section("4. GENERATE DETERMINISTIC TEST DATA")

    # Generate test data ONCE
    test_data = generate_deterministic_test_data(num_samples, num_frames, image_size, seed=42)

    print(f"Test data configuration:")
    print(f"  Num samples: {test_data['num_samples']}")
    print(f"  Num frames: {test_data['num_frames']}")
    print(f"  Image size: {test_data['image_size']}")
    print(f"  Seed: {test_data['seed']}")
    print(f"  Clean video shape: {test_data['clean_video'].shape}")
    print(f"  Clean video range: [{test_data['clean_video'].min().item():.6f}, {test_data['clean_video'].max().item():.6f}]")
    print(f"  Text embedding shape: {test_data['text_emb'].shape}")
    print(f"  Timesteps: {test_data['t'].tolist()}")

    # Apply diffusion noise ONCE
    diffusion_data = apply_diffusion_noise(test_data, diffusion)

    print(f"\nDiffusion data:")
    print(f"  Noisy video shape: {diffusion_data['noisy_videos'].shape}")
    print(f"  Noise shape: {diffusion_data['noise'].shape}")
    print(f"  Noise mean: {diffusion_data['noise'].mean().item():.6f}")
    print(f"  Noise std: {diffusion_data['noise'].std().item():.6f}")

    # Extract tensors for convenience
    noisy_videos = diffusion_data['noisy_videos']
    noise = diffusion_data['noise']
    t = test_data['t']
    text_emb = test_data['text_emb']
    text_mask = test_data['text_mask']

    results = {}

    print_section("5. BASELINE A: ZERO-NOISE PREDICTOR")

    zero_mse = evaluate_baseline_zero(noisy_videos, noise, num_samples, num_frames, image_size)
    print(f"Zero-prediction MSE: {zero_mse:.6f}")
    results['zero_mse'] = zero_mse

    print_section("6. BASELINE C: FRAME-INDEPENDENT (TEMPORAL DISABLED)")

    frame_indep_mse, frame_indep_corr = evaluate_baseline_frame_independent(
        video_unet, noisy_videos, t, text_emb, text_mask, noise, num_samples, num_frames, image_size
    )
    print(f"Frame-independent MSE: {frame_indep_mse:.6f}")
    print(f"Frame-independent correlation: {frame_indep_corr:.6f}")
    results['frame_independent_mse'] = frame_indep_mse
    results['frame_independent_correlation'] = frame_indep_corr

    print_section("7. BASELINE D: TEMPORAL (TEMPORAL ENABLED)")

    temporal_mse, temporal_corr, bottleneck_gate, low_res_gate = evaluate_baseline_temporal(
        video_unet, noisy_videos, t, text_emb, text_mask, noise, num_samples, num_frames, image_size
    )
    print(f"Temporal MSE: {temporal_mse:.6f}")
    print(f"Temporal correlation: {temporal_corr:.6f}")
    print(f"Bottleneck gate: {bottleneck_gate:.6f}")
    print(f"Low-res gate: {low_res_gate:.6f}")
    results['temporal_mse'] = temporal_mse
    results['temporal_correlation'] = temporal_corr
    results['bottleneck_gate'] = bottleneck_gate
    results['low_res_gate'] = low_res_gate

    # Calculate improvement
    improvement = (frame_indep_mse - temporal_mse) / frame_indep_mse * 100
    print(f"\nTemporal improvement over frame-independent: {improvement:.2f}%")
    results['temporal_improvement_pct'] = improvement

    print_section("8. SUMMARY")

    print("\nBaseline comparison (IDENTICAL data conditions):")
    print(f"  A. Zero-prediction:       MSE = {results['zero_mse']:.6f}")
    print(f"  C. Frame-independent:     MSE = {results['frame_independent_mse']:.6f}, correlation = {results['frame_independent_correlation']:.6f}")
    print(f"  D. Temporal:              MSE = {results['temporal_mse']:.6f}, correlation = {results['temporal_correlation']:.6f}")

    print("\nImprovement over zero-prediction:")
    print(f"  Frame-indep:   {(results['zero_mse'] - results['frame_independent_mse']) / results['zero_mse'] * 100:.2f}%")
    print(f"  Temporal:      {(results['zero_mse'] - results['temporal_mse']) / results['zero_mse'] * 100:.2f}%")

    print("\nTemporal vs frame-independent:")
    print(f"  MSE difference: {results['frame_independent_mse'] - results['temporal_mse']:.6f}")
    print(f"  Improvement: {results['temporal_improvement_pct']:.2f}%")

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3.2.6 Task 1: Audit Stage 3.2.5 Evaluation")
    parser.add_argument("--stage2_checkpoint", type=str, default="../checkpoints_conditional/checkpoint_final.pt", help="Stage 2 checkpoint")
    parser.add_argument("--video_checkpoint", type=str, required=True, help="Video checkpoint to audit")
    parser.add_argument("--output", type=str, default="audit_evaluation_results.json", help="Output path")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    results = audit_stage3_2_5_evaluation(
        stage2_checkpoint_path=args.stage2_checkpoint,
        video_checkpoint_path=args.video_checkpoint,
        output_path=args.output,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        image_size=args.image_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
