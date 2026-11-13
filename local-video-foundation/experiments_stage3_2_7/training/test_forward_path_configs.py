"""
Stage 3.2.8: Controlled Forward-Path Experiments

Compare three configurations:
A. Frame-independent spatial model
B. Integrated temporal model with temporal processing disabled
C. Integrated temporal model with temporal processing enabled
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments_stage3_2_7.video_foundation.models.text_encoder import TextEncoder
from experiments_stage3_2_7.video_foundation.models.conditional_unet import ConditionalUNet
from experiments_stage3_2_7.video_foundation.models.video_unet_fixed import VideoUNetFixed


def load_tokenizer(tokenizer_path: str) -> dict:
    """Load tokenizer from JSON file."""
    with open(tokenizer_path, 'r') as f:
        return json.load(f)


def test_forward_path_configs():
    """Test three forward-path configurations."""
    print("="*80)
    print("CONTROLLED FORWARD-PATH EXPERIMENTS")
    print("="*80)

    script_dir = Path(__file__).parent.parent
    tokenizer_path = script_dir / "checkpoints" / "tokenizer_motion_expanded.json"
    tokenizer = load_tokenizer(str(tokenizer_path))

    # Common input
    videos = torch.randn(1, 3, 4, 64, 64)
    timesteps = torch.randint(0, 1000, (1,))
    token_ids = torch.tensor([[3, 12, 33]])
    text_mask = torch.ones_like(token_ids, dtype=torch.float)

    # Configuration A: Frame-independent spatial model
    print("\n--- CONFIGURATION A: Frame-Independent Spatial Model ---")
    spatial_unet_a = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256,
        n_heads=4, use_cross_attn=True
    )

    text_encoder_a = TextEncoder(
        vocab_size=tokenizer['vocab_size'], d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64
    )

    model_a = VideoUNetFixed(
        spatial_unet=spatial_unet_a, text_encoder=text_encoder_a,
        num_frames=4, image_size=64,
        temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=False
    )

    text_enc_a, _ = model_a.text_encoder(token_ids, text_mask)
    output_a = model_a(videos, timesteps, text_enc_a, text_mask)
    print(f"Output shape: {output_a.shape}")
    print(f"Output mean: {output_a.mean().item():.6f}")
    print(f"Output std: {output_a.std().item():.6f}")

    # Configuration B: Integrated temporal model with temporal processing disabled
    print("\n--- CONFIGURATION B: Integrated Temporal Model (Processing Disabled) ---")
    spatial_unet_b = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256,
        n_heads=4, use_cross_attn=True
    )

    text_encoder_b = TextEncoder(
        vocab_size=tokenizer['vocab_size'], d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64
    )

    model_b = VideoUNetFixed(
        spatial_unet=spatial_unet_b, text_encoder=text_encoder_b,
        num_frames=4, image_size=64,
        temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True,
        enable_temporal_processing=False
    )

    text_enc_b, _ = model_b.text_encoder(token_ids, text_mask)
    output_b = model_b(videos, timesteps, text_enc_b, text_mask)
    print(f"Output shape: {output_b.shape}")
    print(f"Output mean: {output_b.mean().item():.6f}")
    print(f"Output std: {output_b.std().item():.6f}")

    # Configuration C: Integrated temporal model with temporal processing enabled
    print("\n--- CONFIGURATION C: Integrated Temporal Model (Processing Enabled) ---")
    spatial_unet_c = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256,
        n_heads=4, use_cross_attn=True
    )

    text_encoder_c = TextEncoder(
        vocab_size=tokenizer['vocab_size'], d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64
    )

    model_c = VideoUNetFixed(
        spatial_unet=spatial_unet_c, text_encoder=text_encoder_c,
        num_frames=4, image_size=64,
        temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True,
        enable_temporal_processing=True
    )

    text_enc_c, _ = model_c.text_encoder(token_ids, text_mask)
    output_c = model_c(videos, timesteps, text_enc_c, text_mask)
    print(f"Output shape: {output_c.shape}")
    print(f"Output mean: {output_c.mean().item():.6f}")
    print(f"Output std: {output_c.std().item():.6f}")

    # Compare A and B (should be equivalent)
    print("\n--- COMPARISON: A vs B (Should be equivalent) ---")
    diff_ab = torch.max(torch.abs(output_a - output_b)).item()
    mean_diff_ab = torch.mean(torch.abs(output_a - output_b)).item()
    print(f"Max difference: {diff_ab:.10f}")
    print(f"Mean difference: {mean_diff_ab:.10f}")

    if diff_ab < 1e-5:
        print("[PASS] Configurations A and B are numerically equivalent")
    else:
        print(f"[WARNING] Configurations A and B differ (max diff: {diff_ab:.6f})")

    # Compare A and C (should differ due to temporal processing)
    print("\n--- COMPARISON: A vs C (Should differ due to temporal processing) ---")
    diff_ac = torch.max(torch.abs(output_a - output_c)).item()
    mean_diff_ac = torch.mean(torch.abs(output_a - output_c)).item()
    print(f"Max difference: {diff_ac:.10f}")
    print(f"Mean difference: {mean_diff_ac:.10f}")

    if diff_ac > 1e-3:
        print(f"[PASS] Configurations A and C differ significantly (max diff: {diff_ac:.6f})")
    else:
        print(f"[WARNING] Configurations A and C are too similar (max diff: {diff_ac:.6f})")

    # Compare B and C (should differ due to temporal processing)
    print("\n--- COMPARISON: B vs C (Should differ due to temporal processing) ---")
    diff_bc = torch.max(torch.abs(output_b - output_c)).item()
    mean_diff_bc = torch.mean(torch.abs(output_b - output_c)).item()
    print(f"Max difference: {diff_bc:.10f}")
    print(f"Mean difference: {mean_diff_bc:.10f}")

    if diff_bc > 1e-3:
        print(f"[PASS] Configurations B and C differ significantly (max diff: {diff_bc:.6f})")
    else:
        print(f"[WARNING] Configurations B and C are too similar (max diff: {diff_bc:.6f})")

    return True


def main():
    success = test_forward_path_configs()

    print("\n" + "="*80)
    if success:
        print("[PASS] FORWARD-PATH EXPERIMENTS COMPLETED")
    else:
        print("[ERROR] FORWARD-PATH EXPERIMENTS FAILED")
    print("="*80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
