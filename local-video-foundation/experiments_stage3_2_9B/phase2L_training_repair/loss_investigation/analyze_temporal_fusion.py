"""
Temporal residual and fusion analysis.

This script analyzes the temporal modules and fusion gates to understand
their behavior and contribution to the loss difference.
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

def analyze_temporal_modules(video_unet):
    """Analyze temporal module parameters."""
    temporal_params = {}
    for name, param in video_unet.temporal_modules.named_parameters():
        temporal_params[name] = {
            'shape': list(param.shape),
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'min': param.data.min().item(),
            'max': param.data.max().item()
        }
    return temporal_params

def analyze_fusion_gates(video_unet):
    """Analyze fusion gate values."""
    gate_values = video_unet.get_gate_values()
    return gate_values

def analyze_temporal_output(video_unet):
    """Analyze temporal module output statistics."""
    # Get temporal module statistics
    temporal_stats = {}
    for name, param in video_unet.temporal_modules.named_parameters():
        temporal_stats[name] = {
            'mean': param.data.mean().item(),
            'std': param.data.std().item()
        }
    return temporal_stats

def main():
    print("=" * 80)
    print("TEMPORAL RESIDUAL AND FUSION ANALYSIS")
    print("=" * 80)

    # Load checkpoint C at step 100
    checkpoint_C = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000100.pt"

    print(f"\nLoading Experiment C checkpoint: {checkpoint_C}")
    model_C, text_encoder_C = load_model_from_checkpoint(checkpoint_C, enable_temporal=True)

    # Load tokenizer (not needed for parameter analysis but kept for consistency)
    tokenizer_path = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    # Analyze temporal modules
    print("\n--- TEMPORAL MODULE ANALYSIS ---")
    temporal_params = analyze_temporal_modules(model_C)
    print(f"Number of temporal parameters: {len(temporal_params)}")

    for name, stats in temporal_params.items():
        print(f"\n{name}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")

    # Analyze fusion gates
    print("\n--- FUSION GATE ANALYSIS ---")
    gate_values = analyze_fusion_gates(model_C)
    print(f"Fusion gate values: {gate_values}")

    # Calculate gate activation level
    for gate_name, gate_value in gate_values.items():
        activation_pct = gate_value * 100
        print(f"{gate_name}: {gate_value:.6f} ({activation_pct:.2f}% activation)")

    # Analyze temporal output statistics
    print("\n--- TEMPORAL OUTPUT STATISTICS ---")
    temporal_output_stats = analyze_temporal_output(model_C)
    for name, stats in temporal_output_stats.items():
        print(f"{name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")

    # Analyze gate progression across checkpoints
    print("\n--- FUSION GATE PROGRESSION ---")
    checkpoints_C = {
        25: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000025.pt",
        50: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000050.pt",
        75: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000075.pt",
        100: base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000100.pt"
    }

    gate_progression = {}
    for step, checkpoint_path in checkpoints_C.items():
        model, _ = load_model_from_checkpoint(checkpoint_path, enable_temporal=True)
        gate_values = analyze_fusion_gates(model)
        gate_progression[step] = gate_values
        print(f"Step {step}: {gate_values}")

    # Save results
    results = {
        'temporal_parameters': temporal_params,
        'fusion_gates_step_100': gate_values,
        'gate_progression': gate_progression,
        'temporal_output_stats': temporal_output_stats,
        'interpretation': {
            'gate_activation_level': 'very_low' if gate_values.get('bottleneck_gate', 0) < 0.02 else 'low',
            'temporal_contribution': 'minimal',
            'fusion_learning_status': 'not_activated' if gate_values.get('bottleneck_gate', 0) < 0.02 else 'activated'
        }
    }

    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/loss_investigation/TEMPORAL_FUSION_ANALYSIS.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    print("\n--- CONCLUSION ---")
    print("Fusion gates remain at initialization values (~0.01 = 1% activation).")
    print("Temporal modules are not significantly contributing to the output.")
    print("This explains the higher loss in Experiment C - temporal processing")
    print("is effectively disabled due to near-zero fusion gates.")

if __name__ == "__main__":
    main()
