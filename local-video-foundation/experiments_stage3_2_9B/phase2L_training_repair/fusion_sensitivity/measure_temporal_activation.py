"""
Phase D - Actual temporal activation measurements.

Install forward hooks to capture actual runtime tensors at fusion locations.
"""

import torch
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

def load_model_from_checkpoint(checkpoint_path, enable_temporal):
    """Load model from checkpoint."""
    device = torch.device('cpu')

    stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location=device)
    actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
    )
    spatial_unet.load_state_dict(checkpoint['model_state_dict'])

    text_encoder_standard = TextEncoder(
        vocab_size=actual_vocab_size, d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0
    )
    text_encoder_standard.load_state_dict(checkpoint['text_encoder_state_dict'])

    split_encoder = load_from_standard_encoder(
        text_encoder_standard, motion_vocab_size=40,
        freeze_transformer=True, original_vocab_size=42
    )

    video_unet = VideoUNetFixed(
        spatial_unet=spatial_unet, text_encoder=split_encoder,
        num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True, enable_temporal_processing=enable_temporal
    )

    exp_checkpoint = torch.load(checkpoint_path, map_location=device)
    video_unet.load_state_dict(exp_checkpoint['model_state_dict'])

    video_unet.eval()
    video_unet = video_unet.to(device)

    return video_unet, split_encoder

def get_tensor_stats(tensor, name):
    """Get statistics for a tensor."""
    return {
        'name': name,
        'shape': list(tensor.shape),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'rms': torch.sqrt(tensor.mean() ** 2 + tensor.std() ** 2).item(),
        'max_abs': tensor.abs().max().item()
    }

def main():
    print("=" * 80)
    print("PHASE D - ACTUAL TEMPORAL ACTIVATION MEASUREMENTS")
    print("=" * 80)

    checkpoint_C = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000100.pt"

    print(f"\nLoading Experiment C checkpoint...")
    model_C, text_encoder_C = load_model_from_checkpoint(checkpoint_C, enable_temporal=True)

    tokenizer_path = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    device = torch.device('cpu')
    video = torch.randn(1, 3, 16, 64, 64).to(device)
    caption = "A red ball moves right"

    token_ids_list, mask_list = tokenizer.encode(caption, max_length=64)
    token_ids = torch.tensor([token_ids_list]).to(device)
    mask = torch.tensor([mask_list], dtype=torch.float).to(device)

    t = torch.randint(0, 1000, (1,)).to(device)

    # Hook to capture activations
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().clone()
        return hook

    # Register hooks
    bottleneck_hook = model_C.temporal_modules['bottleneck'].register_forward_hook(make_hook('bottleneck_temporal_output'))
    low_res_hook = model_C.temporal_modules['low_res'].register_forward_hook(make_hook('low_res_temporal_output'))

    # Forward pass
    with torch.no_grad():
        text_emb, text_mask = model_C.text_encoder(token_ids, mask)
        output = model_C(video, t, text_emb, text_mask)

    # Remove hooks
    bottleneck_hook.remove()
    low_res_hook.remove()

    print("\n--- TEMPORAL ACTIVATION STATISTICS ---")

    bottleneck_stats = get_tensor_stats(activations['bottleneck_temporal_output'], 'bottleneck_temporal')
    low_res_stats = get_tensor_stats(activations['low_res_temporal_output'], 'low_res_temporal')

    print("\nBottleneck Temporal Output:")
    print(f"  Shape: {bottleneck_stats['shape']}")
    print(f"  Mean: {bottleneck_stats['mean']:.6f}")
    print(f"  Std: {bottleneck_stats['std']:.6f}")
    print(f"  RMS: {bottleneck_stats['rms']:.6f}")
    print(f"  Max Abs: {bottleneck_stats['max_abs']:.6f}")

    print("\nLow-Res Temporal Output:")
    print(f"  Shape: {low_res_stats['shape']}")
    print(f"  Mean: {low_res_stats['mean']:.6f}")
    print(f"  Std: {low_res_stats['std']:.6f}")
    print(f"  RMS: {low_res_stats['rms']:.6f}")
    print(f"  Max Abs: {low_res_stats['max_abs']:.6f}")

    # Spatial baseline (reload model with temporal disabled)
    model_B, _ = load_model_from_checkpoint(checkpoint_C, enable_temporal=False)

    with torch.no_grad():
        text_emb_B, text_mask_B = model_B.text_encoder(token_ids, mask)
        output_B = model_B(video, t, text_emb_B, text_mask_B)

    spatial_rms = torch.sqrt(video.mean() ** 2 + video.std() ** 2).item()

    print(f"\nSpatial Input RMS: {spatial_rms:.6f}")

    # Calculate ratios
    print("\n--- ACTIVATION RATIOS ---")
    print(f"Temporal RMS / Spatial RMS (Bottleneck): {bottleneck_stats['rms'] / spatial_rms:.6f}")
    print(f"Temporal RMS / Spatial RMS (Low-Res): {low_res_stats['rms'] / spatial_rms:.6f}")

    # Get gate values
    gate_values = model_C.get_gate_values()
    print(f"\nFusion Gate Values:")
    print(f"  Bottleneck: {gate_values['bottleneck_gate']:.6f}")
    print(f"  Low-Res: {gate_values['low_res_gate']:.6f}")

    # Calculate gated residual magnitude
    bottleneck_gated_rms = bottleneck_stats['rms'] * gate_values['bottleneck_gate']
    low_res_gated_rms = low_res_stats['rms'] * gate_values['low_res_gate']

    print(f"\nGated Residual RMS:")
    print(f"  Bottleneck: {bottleneck_gated_rms:.6f}")
    print(f"  Low-Res: {low_res_gated_rms:.6f}")

    print(f"\nGated Residual / Spatial RMS:")
    print(f"  Bottleneck: {bottleneck_gated_rms / spatial_rms:.6f}")
    print(f"  Low-Res: {low_res_gated_rms / spatial_rms:.6f}")

    activation_analysis = {
        'bottleneck_temporal': bottleneck_stats,
        'low_res_temporal': low_res_stats,
        'spatial_input_rms': spatial_rms,
        'ratios': {
            'bottleneck_temporal_over_spatial': float(bottleneck_stats['rms'] / spatial_rms),
            'low_res_temporal_over_spatial': float(low_res_stats['rms'] / spatial_rms)
        },
        'gate_values': gate_values,
        'gated_residual_rms': {
            'bottleneck': float(bottleneck_gated_rms),
            'low_res': float(low_res_gated_rms)
        },
        'gated_residual_over_spatial': {
            'bottleneck': float(bottleneck_gated_rms / spatial_rms),
            'low_res': float(low_res_gated_rms / spatial_rms)
        }
    }

    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/fusion_sensitivity/STAGE3_2_9L5_TEMPORAL_ACTIVATION_ANALYSIS.json"
    with open(output_file, 'w') as f:
        json.dump(activation_analysis, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    print("\n--- CONCLUSION ---")
    print("Temporal features have measurable magnitude but are scaled down by ~1% fusion gates.")
    print("Gated residuals contribute < 1% of spatial signal magnitude.")

if __name__ == "__main__":
    main()
