"""
Controlled same-input loss comparison across checkpoints.

This script loads checkpoints from different steps (25, 50, 75, 100) for both
Experiments B and C, and compares their losses on the same input to understand
training progression.
"""

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
import sys

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
stage3_2_7_dir = base_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation" / "models"))

from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed
from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder
from models.text_encoder import TextEncoder, SimpleTokenizer

def get_diffusion_schedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    """Get diffusion beta schedule."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def add_noise(x0, t, alpha_bars):
    """Add noise to clean video at timestep t."""
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t])
    sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1)
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt, noise

def load_model_from_checkpoint(checkpoint_path, enable_temporal):
    """Load model from checkpoint."""
    device = torch.device('cpu')

    # Load base checkpoint for spatial and text encoder
    stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location=device)
    actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

    # Create spatial UNet
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
    spatial_unet.load_state_dict(checkpoint['model_state_dict'])

    # Create text encoder
    text_encoder_standard = TextEncoder(
        vocab_size=actual_vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64,
        dropout=0.0
    )
    text_encoder_standard.load_state_dict(checkpoint['text_encoder_state_dict'])

    # Create split encoder
    split_encoder = load_from_standard_encoder(
        text_encoder_standard,
        motion_vocab_size=40,
        freeze_transformer=True,
        original_vocab_size=42
    )

    # Create video UNet
    video_unet = VideoUNetFixed(
        spatial_unet=spatial_unet,
        text_encoder=split_encoder,
        num_frames=16,
        image_size=64,
        temporal_kernel=3,
        temporal_layers=2,
        freeze_spatial=True,
        use_temporal=True,
        enable_temporal_processing=enable_temporal
    )

    # Load experiment checkpoint
    exp_checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load experiment state
    video_unet.load_state_dict(exp_checkpoint['model_state_dict'])

    video_unet.eval()
    video_unet = video_unet.to(device)

    return video_unet, split_encoder

def evaluate_checkpoint(checkpoint_path, enable_temporal, video, token_ids, mask, num_timesteps=10):
    """Evaluate checkpoint on multiple timesteps."""
    device = torch.device('cpu')
    video_unet, split_encoder = load_model_from_checkpoint(checkpoint_path, enable_temporal)

    betas, alphas, alpha_bars = get_diffusion_schedule()

    losses = []

    for _ in range(num_timesteps):
        t = torch.randint(0, 1000, (1,)).to(device)
        xt, noise = add_noise(video, t, alpha_bars)

        with torch.no_grad():
            text_emb, text_mask = video_unet.text_encoder(token_ids, mask)
            output = video_unet(xt, t, text_emb, text_mask)
            loss = nn.functional.mse_loss(output, noise)
            losses.append(loss.item())

    return {
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses),
        'min_loss': np.min(losses),
        'max_loss': np.max(losses),
        'losses': losses
    }

def main():
    print("=" * 80)
    print("CONTROLLED SAME-INPUT LOSS COMPARISON")
    print("=" * 80)

    # Define checkpoints
    checkpoints_B = {
        25: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000025.pt",
        50: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000050.pt",
        75: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000075.pt",
        100: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_B/checkpoints/checkpoint_step_000100.pt"
    }

    checkpoints_C = {
        25: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000025.pt",
        50: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000050.pt",
        75: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000075.pt",
        100: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000100.pt"
    }

    # Load tokenizer
    tokenizer_path = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    # Create synthetic input
    device = torch.device('cpu')
    video = torch.randn(1, 3, 16, 64, 64).to(device)
    caption = "A red ball moves right"

    # Encode text
    token_ids_list, mask_list = tokenizer.encode(caption, max_length=64)
    token_ids = torch.tensor([token_ids_list]).to(device)
    mask = torch.tensor([mask_list], dtype=torch.float).to(device)

    print(f"\nInput shape: {video.shape}")
    print(f"Caption: {caption}")

    # Evaluate Experiment B checkpoints
    print("\n" + "=" * 80)
    print("EXPERIMENT B CHECKPOINT EVALUATION")
    print("=" * 80)

    results_B = {}
    for step, checkpoint_path in checkpoints_B.items():
        print(f"\nEvaluating checkpoint step {step}...")
        results = evaluate_checkpoint(checkpoint_path, False, video, token_ids, mask, 20)
        results_B[step] = results
        print(f"  Mean loss: {results['mean_loss']:.6f}")
        print(f"  Std loss: {results['std_loss']:.6f}")
        print(f"  Min loss: {results['min_loss']:.6f}")
        print(f"  Max loss: {results['max_loss']:.6f}")

    # Evaluate Experiment C checkpoints
    print("\n" + "=" * 80)
    print("EXPERIMENT C CHECKPOINT EVALUATION")
    print("=" * 80)

    results_C = {}
    for step, checkpoint_path in checkpoints_C.items():
        print(f"\nEvaluating checkpoint step {step}...")
        results = evaluate_checkpoint(checkpoint_path, True, video, token_ids, mask, 20)
        results_C[step] = results
        print(f"  Mean loss: {results['mean_loss']:.6f}")
        print(f"  Std loss: {results['std_loss']:.6f}")
        print(f"  Min loss: {results['min_loss']:.6f}")
        print(f"  Max loss: {results['max_loss']:.6f}")

    # Compare progression
    print("\n" + "=" * 80)
    print("TRAINING PROGRESSION COMPARISON")
    print("=" * 80)

    print("\nExperiment B progression:")
    for step in sorted(results_B.keys()):
        print(f"  Step {step}: {results_B[step]['mean_loss']:.6f}")

    print("\nExperiment C progression:")
    for step in sorted(results_C.keys()):
        print(f"  Step {step}: {results_C[step]['mean_loss']:.6f}")

    print("\nLoss difference (C - B) at each step:")
    for step in sorted(results_B.keys()):
        diff = results_C[step]['mean_loss'] - results_B[step]['mean_loss']
        ratio = results_C[step]['mean_loss'] / results_B[step]['mean_loss']
        print(f"  Step {step}: {diff:.6f} (ratio: {ratio:.2f}x)")

    # Analyze convergence
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    B_change = results_B[100]['mean_loss'] - results_B[25]['mean_loss']
    C_change = results_C[100]['mean_loss'] - results_C[25]['mean_loss']

    print(f"Experiment B loss change (step 25 to 100): {B_change:.6f}")
    print(f"Experiment C loss change (step 25 to 100): {C_change:.6f}")

    if B_change < 0:
        print("  -> Experiment B is converging (loss decreasing)")
    else:
        print("  -> Experiment B is diverging (loss increasing)")

    if C_change < 0:
        print("  -> Experiment C is converging (loss decreasing)")
    else:
        print("  -> Experiment C is diverging (loss increasing)")

    # Save results
    results = {
        'experiment_B': results_B,
        'experiment_C': results_C,
        'comparison': {
            'loss_difference_at_25': results_C[25]['mean_loss'] - results_B[25]['mean_loss'],
            'loss_difference_at_50': results_C[50]['mean_loss'] - results_B[50]['mean_loss'],
            'loss_difference_at_75': results_C[75]['mean_loss'] - results_B[75]['mean_loss'],
            'loss_difference_at_100': results_C[100]['mean_loss'] - results_B[100]['mean_loss'],
            'B_convergence': B_change,
            'C_convergence': C_change
        }
    }

    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/loss_investigation/CONTROLLED_LOSS_COMPARISON.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
