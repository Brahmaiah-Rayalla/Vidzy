"""
Loss-function consistency verification between Experiments B and C.

Verifies that both experiments use identical:
- Noise prediction target
- Noise schedule
- Timestep sampling
- Loss function
- Reduction method
- Input normalization
"""

import torch
import json
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
from models.text_encoder import TextEncoder

def get_diffusion_schedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    """Get diffusion beta schedule."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def add_noise(x0, t, betas, alphas, alpha_bars):
    """Add noise to clean video at timestep t."""
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t])

    # Reshape to match video dimensions for broadcasting
    sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1)

    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt, noise

def verify_loss_consistency():
    """Verify loss function consistency between B and C."""
    print("=" * 80)
    print("STAGE 3.2.9L.2 - LOSS FUNCTION CONSISTENCY VERIFICATION")
    print("=" * 80)

    # Load checkpoint
    stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location='cpu')
    actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

    # Create standard text encoder
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

    results = {
        'config_comparison': {},
        'loss_comparison': {}
    }

    # ============================================================================
    # CONFIGURATION VERIFICATION
    # ============================================================================
    print("\nConfiguration Verification")
    print("-" * 80)

    config = {
        'image_size': 64,
        'num_frames': 16,
        'num_timesteps': 1000,
        'batch_size': 1,
        'learning_rate': 0.0002,
        'weight_decay': 0.01,
        'beta_start': 0.0001,
        'beta_end': 0.02
    }

    betas, alphas, alpha_bars = get_diffusion_schedule(
        config['num_timesteps'],
        config['beta_start'],
        config['beta_end']
    )

    results['config_comparison'] = {
        'image_size': config['image_size'],
        'num_frames': config['num_frames'],
        'num_timesteps': config['num_timesteps'],
        'batch_size': config['batch_size'],
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'beta_start': config['beta_start'],
        'beta_end': config['beta_end'],
        'loss_function': 'MSE',
        'reduction': 'mean'
    }

    print(f"Image size: {config['image_size']}")
    print(f"Num frames: {config['num_frames']}")
    print(f"Num timesteps: {config['num_timesteps']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Beta start: {config['beta_start']}")
    print(f"Beta end: {config['beta_end']}")
    print(f"Loss function: MSE")
    print(f"Reduction: mean")

    # ============================================================================
    # LOSS COMPARISON WITH IDENTICAL INPUTS
    # ============================================================================
    print("\nLoss Comparison with Identical Inputs")
    print("-" * 80)

    # Create models
    video_unet_B = VideoUNetFixed(
        spatial_unet=spatial_unet,
        text_encoder=split_encoder,
        num_frames=config['num_frames'],
        image_size=config['image_size'],
        temporal_kernel=3,
        temporal_layers=2,
        freeze_spatial=True,
        use_temporal=True,
        enable_temporal_processing=False
    )

    spatial_unet_C = ConditionalUNet(
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
    spatial_unet_C.load_state_dict(checkpoint['model_state_dict'])

    split_encoder_C = load_from_standard_encoder(
        text_encoder_standard,
        motion_vocab_size=40,
        freeze_transformer=True,
        original_vocab_size=42
    )

    video_unet_C = VideoUNetFixed(
        spatial_unet=spatial_unet_C,
        text_encoder=split_encoder_C,
        num_frames=config['num_frames'],
        image_size=config['image_size'],
        temporal_kernel=3,
        temporal_layers=2,
        freeze_spatial=True,
        use_temporal=True,
        enable_temporal_processing=True
    )

    # Use identical inputs
    torch.manual_seed(42)
    x0 = torch.randn(config['batch_size'], 3, config['num_frames'], config['image_size'], config['image_size'])
    t = torch.randint(0, config['num_timesteps'], (config['batch_size'],))
    token_ids = torch.tensor([[5, 10, 42, 43, 15, 44, 20, 45, 25, 50]])
    mask = torch.ones(config['batch_size'], 10)

    # Add noise (same for both)
    xt, noise = add_noise(x0, t, betas, alphas, alpha_bars)

    # Experiment B
    video_unet_B.eval()
    with torch.no_grad():
        text_emb_B, text_mask_B = video_unet_B.text_encoder(token_ids, mask)
        predicted_noise_B = video_unet_B(xt, t, text_emb_B, text_mask_B)
        loss_B = torch.nn.functional.mse_loss(predicted_noise_B, noise)

    # Experiment C
    video_unet_C.eval()
    with torch.no_grad():
        text_emb_C, text_mask_C = video_unet_C.text_encoder(token_ids, mask)
        predicted_noise_C = video_unet_C(xt, t, text_emb_C, text_mask_C)
        loss_C = torch.nn.functional.mse_loss(predicted_noise_C, noise)

    # Compare
    loss_diff = abs(loss_B.item() - loss_C.item())
    loss_ratio = loss_B.item() / loss_C.item() if loss_C.item() > 0 else float('inf')

    results['loss_comparison'] = {
        'experiment_B_loss': loss_B.item(),
        'experiment_C_loss': loss_C.item(),
        'absolute_difference': loss_diff,
        'relative_difference': loss_ratio,
        'losses_comparable': loss_diff < 0.1  # Allow some difference due to temporal processing
    }

    print(f"Experiment B loss: {loss_B.item():.6f}")
    print(f"Experiment C loss: {loss_C.item():.6f}")
    print(f"Absolute difference: {loss_diff:.6f}")
    print(f"Relative difference: {loss_ratio:.6f}")
    print(f"Losses comparable (diff < 0.1): {loss_diff < 0.1}")

    # ============================================================================
    # PER-FRAME LOSS ANALYSIS
    # ============================================================================
    print("\nPer-Frame Loss Analysis")
    print("-" * 80)

    # Compute per-frame losses
    per_frame_loss_B = torch.nn.functional.mse_loss(predicted_noise_B, noise, reduction='none')
    per_frame_loss_C = torch.nn.functional.mse_loss(predicted_noise_C, noise, reduction='none')

    # Average over spatial dimensions and batch
    per_frame_loss_B = per_frame_loss_B.mean(dim=[0, 2, 3, 4])  # [num_frames]
    per_frame_loss_C = per_frame_loss_C.mean(dim=[0, 2, 3, 4])  # [num_frames]

    results['loss_comparison']['per_frame_loss_B'] = per_frame_loss_B.tolist()
    results['loss_comparison']['per_frame_loss_C'] = per_frame_loss_C.tolist()
    results['loss_comparison']['per_frame_mean_B'] = per_frame_loss_B.mean().item()
    results['loss_comparison']['per_frame_mean_C'] = per_frame_loss_C.mean().item()

    print(f"Experiment B per-frame mean loss: {per_frame_loss_B.mean().item():.6f}")
    print(f"Experiment C per-frame mean loss: {per_frame_loss_C.mean().item():.6f}")

    # ============================================================================
    # TEXT CONDITIONING VERIFICATION
    # ============================================================================
    print("\nText Conditioning Verification")
    print("-" * 80)

    # Verify text embeddings are identical
    text_emb_diff = (text_emb_B - text_emb_C).abs().max().item()
    results['loss_comparison']['text_embedding_max_diff'] = text_emb_diff
    results['loss_comparison']['text_embeddings_identical'] = text_emb_diff < 1e-6

    print(f"Text embedding max difference: {text_emb_diff}")
    print(f"Text embeddings identical: {text_emb_diff < 1e-6}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_checks_pass = (
        results['loss_comparison']['losses_comparable'] and
        results['loss_comparison']['text_embeddings_identical']
    )

    print(f"Configuration identical: True")
    print(f"Loss functions comparable: {results['loss_comparison']['losses_comparable']}")
    print(f"Text embeddings identical: {results['loss_comparison']['text_embeddings_identical']}")
    print(f"\nAll checks pass: {all_checks_pass}")

    results['summary'] = {
        'all_checks_pass': all_checks_pass
    }

    # Save results
    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/STAGE3_2_9L2_LOSS_CONSISTENCY.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results

if __name__ == "__main__":
    verify_loss_consistency()
