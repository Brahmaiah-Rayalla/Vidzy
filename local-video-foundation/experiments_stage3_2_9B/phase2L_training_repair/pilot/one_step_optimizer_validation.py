"""
One-step optimizer validation for Experiments B and C.

Verifies:
- Gradients exist for intended parameters
- Parameters actually change after optimizer step
- Frozen parameters remain unchanged
- Optimizer parameter groups are correct
"""

import torch
import torch.optim as optim
import json
from pathlib import Path
import sys
import copy

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

def get_param_hash(param):
    """Get a hash of parameter values for comparison."""
    return torch.hash(param).item()

def validate_one_step():
    """Validate one optimizer step for both experiments."""
    print("=" * 80)
    print("STAGE 3.2.9L.2 - ONE-STEP OPTIMIZER VALIDATION")
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
        'experiment_B': {},
        'experiment_C': {}
    }

    # ============================================================================
    # EXPERIMENT B: Motion embeddings only
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT B - Motion Embeddings Only")
    print("=" * 80)

    video_unet_B = VideoUNetFixed(
        spatial_unet=spatial_unet,
        text_encoder=split_encoder,
        num_frames=16,
        image_size=64,
        temporal_kernel=3,
        temporal_layers=2,
        freeze_spatial=True,
        use_temporal=True,
        enable_temporal_processing=False
    )

    # Setup optimizer
    motion_params = list(split_encoder.token_emb.motion_emb.parameters())
    optimizer_B = optim.AdamW(motion_params, lr=0.0002, weight_decay=0.01)

    # Record initial parameter values
    initial_motion_B = {id(p): p.clone() for p in motion_params}
    initial_spatial_B = {id(p): p.clone() for p in video_unet_B.spatial_unet.parameters()}
    initial_appearance_B = {id(p): p.clone() for p in split_encoder.token_emb.original_emb.parameters()}

    # Forward pass
    x = torch.randn(1, 3, 16, 64, 64)
    t = torch.randint(0, 1000, (1,))
    token_ids = torch.tensor([[5, 10, 42, 43, 15, 44, 20, 45, 25, 50]])
    mask = torch.ones(1, 10)

    video_unet_B.train()
    text_emb, text_mask = video_unet_B.text_encoder(token_ids, mask)
    predicted_noise = video_unet_B(x, t, text_emb, text_mask)
    loss = torch.nn.functional.mse_loss(predicted_noise, torch.randn_like(predicted_noise))

    # Backward pass
    optimizer_B.zero_grad()
    loss.backward()

    # Check gradients
    motion_grads_B = [p.grad for p in motion_params]
    motion_has_grad_B = [g is not None for g in motion_grads_B]
    motion_grad_norms_B = [g.norm().item() if g is not None else 0.0 for g in motion_grads_B]

    # Optimizer step
    optimizer_B.step()

    # Record final parameter values
    final_motion_B = {id(p): p.clone() for p in motion_params}
    final_spatial_B = {id(p): p.clone() for p in video_unet_B.spatial_unet.parameters()}
    final_appearance_B = {id(p): p.clone() for p in split_encoder.token_emb.original_emb.parameters()}

    # Compute changes
    motion_changes_B = [torch.norm(final_motion_B[id(p)] - initial_motion_B[id(p)]).item() for p in motion_params]
    spatial_changes_B = [torch.norm(final_spatial_B[id(p)] - initial_spatial_B[id(p)]).item() for p in video_unet_B.spatial_unet.parameters()]
    appearance_changes_B = [torch.norm(final_appearance_B[id(p)] - initial_appearance_B[id(p)]).item() for p in split_encoder.token_emb.original_emb.parameters()]

    results['experiment_B'] = {
        'motion_embedding': {
            'parameter_count': len(motion_params),
            'total_parameters': sum(p.numel() for p in motion_params),
            'all_have_grad': all(motion_has_grad_B),
            'gradient_norms': motion_grad_norms_B,
            'max_gradient_norm': max(motion_grad_norms_B) if motion_grad_norms_B else 0.0,
            'parameters_changed': [c > 1e-8 for c in motion_changes_B],
            'max_parameter_change': max(motion_changes_B) if motion_changes_B else 0.0,
            'mean_parameter_change': sum(motion_changes_B) / len(motion_changes_B) if motion_changes_B else 0.0
        },
        'spatial': {
            'parameter_count': len(list(video_unet_B.spatial_unet.parameters())),
            'max_change': max(spatial_changes_B) if spatial_changes_B else 0.0,
            'mean_change': sum(spatial_changes_B) / len(spatial_changes_B) if spatial_changes_B else 0.0,
            'any_changed': any(c > 1e-8 for c in spatial_changes_B)
        },
        'appearance': {
            'parameter_count': len(list(split_encoder.token_emb.original_emb.parameters())),
            'max_change': max(appearance_changes_B) if appearance_changes_B else 0.0,
            'mean_change': sum(appearance_changes_B) / len(appearance_changes_B) if appearance_changes_B else 0.0,
            'any_changed': any(c > 1e-8 for c in appearance_changes_B)
        },
        'optimizer': {
            'parameter_count': len(optimizer_B.param_groups[0]['params']),
            'all_gradients_finite': all(torch.isfinite(g).all() if g is not None else True for g in motion_grads_B)
        }
    }

    print(f"Motion embedding parameters: {len(motion_params)}")
    print(f"Motion has gradients: {all(motion_has_grad_B)}")
    print(f"Motion gradient norm: {motion_grad_norms_B[0]:.6f}")
    print(f"Motion parameters changed: {results['experiment_B']['motion_embedding']['parameters_changed']}")
    print(f"Motion max change: {results['experiment_B']['motion_embedding']['max_parameter_change']:.8f}")
    print(f"Spatial any changed: {results['experiment_B']['spatial']['any_changed']}")
    print(f"Appearance any changed: {results['experiment_B']['appearance']['any_changed']}")

    # ============================================================================
    # EXPERIMENT C: Motion + Temporal + Fusion
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT C - Motion + Temporal + Fusion")
    print("=" * 80)

    # Recreate models for Experiment C
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
        num_frames=16,
        image_size=64,
        temporal_kernel=3,
        temporal_layers=2,
        freeze_spatial=True,
        use_temporal=True,
        enable_temporal_processing=True
    )

    # Setup optimizer
    motion_params_C = list(split_encoder_C.token_emb.motion_emb.parameters())
    temporal_params_C = list(video_unet_C.temporal_modules.parameters())
    fusion_params_C = list(video_unet_C.fusion_modules.parameters())
    trainable_params_C = motion_params_C + temporal_params_C + fusion_params_C

    optimizer_C = optim.AdamW(trainable_params_C, lr=0.0002, weight_decay=0.01)

    # Record initial parameter values
    initial_motion_C = {id(p): p.clone() for p in motion_params_C}
    initial_temporal_C = {id(p): p.clone() for p in temporal_params_C}
    initial_fusion_C = {id(p): p.clone() for p in fusion_params_C}
    initial_spatial_C = {id(p): p.clone() for p in video_unet_C.spatial_unet.parameters()}
    initial_appearance_C = {id(p): p.clone() for p in split_encoder_C.token_emb.original_emb.parameters()}

    # Forward pass (same inputs for fairness)
    video_unet_C.train()
    text_emb_C, text_mask_C = video_unet_C.text_encoder(token_ids, mask)
    predicted_noise_C = video_unet_C(x, t, text_emb_C, text_mask_C)
    loss_C = torch.nn.functional.mse_loss(predicted_noise_C, torch.randn_like(predicted_noise_C))

    # Backward pass
    optimizer_C.zero_grad()
    loss_C.backward()

    # Check gradients
    motion_grads_C = [p.grad for p in motion_params_C]
    temporal_grads_C = [p.grad for p in temporal_params_C]
    fusion_grads_C = [p.grad for p in fusion_params_C]

    motion_has_grad_C = [g is not None for g in motion_grads_C]
    temporal_has_grad_C = [g is not None for g in temporal_grads_C]
    fusion_has_grad_C = [g is not None for g in fusion_grads_C]

    motion_grad_norms_C = [g.norm().item() if g is not None else 0.0 for g in motion_grads_C]
    temporal_grad_norms_C = [g.norm().item() if g is not None else 0.0 for g in temporal_grads_C]
    fusion_grad_norms_C = [g.norm().item() if g is not None else 0.0 for g in fusion_grads_C]

    # Optimizer step
    optimizer_C.step()

    # Record final parameter values
    final_motion_C = {id(p): p.clone() for p in motion_params_C}
    final_temporal_C = {id(p): p.clone() for p in temporal_params_C}
    final_fusion_C = {id(p): p.clone() for p in fusion_params_C}
    final_spatial_C = {id(p): p.clone() for p in video_unet_C.spatial_unet.parameters()}
    final_appearance_C = {id(p): p.clone() for p in split_encoder_C.token_emb.original_emb.parameters()}

    # Compute changes
    motion_changes_C = [torch.norm(final_motion_C[id(p)] - initial_motion_C[id(p)]).item() for p in motion_params_C]
    temporal_changes_C = [torch.norm(final_temporal_C[id(p)] - initial_temporal_C[id(p)]).item() for p in temporal_params_C]
    fusion_changes_C = [torch.norm(final_fusion_C[id(p)] - initial_fusion_C[id(p)]).item() for p in fusion_params_C]
    spatial_changes_C = [torch.norm(final_spatial_C[id(p)] - initial_spatial_C[id(p)]).item() for p in video_unet_C.spatial_unet.parameters()]
    appearance_changes_C = [torch.norm(final_appearance_C[id(p)] - initial_appearance_C[id(p)]).item() for p in split_encoder_C.token_emb.original_emb.parameters()]

    results['experiment_C'] = {
        'motion_embedding': {
            'parameter_count': len(motion_params_C),
            'total_parameters': sum(p.numel() for p in motion_params_C),
            'all_have_grad': all(motion_has_grad_C),
            'gradient_norms': motion_grad_norms_C,
            'max_gradient_norm': max(motion_grad_norms_C) if motion_grad_norms_C else 0.0,
            'parameters_changed': [c > 1e-8 for c in motion_changes_C],
            'max_parameter_change': max(motion_changes_C) if motion_changes_C else 0.0,
            'mean_parameter_change': sum(motion_changes_C) / len(motion_changes_C) if motion_changes_C else 0.0
        },
        'temporal_modules': {
            'parameter_count': len(temporal_params_C),
            'total_parameters': sum(p.numel() for p in temporal_params_C),
            'all_have_grad': all(temporal_has_grad_C),
            'gradient_norms': temporal_grad_norms_C,
            'max_gradient_norm': max(temporal_grad_norms_C) if temporal_grad_norms_C else 0.0,
            'parameters_changed': [c > 1e-8 for c in temporal_changes_C],
            'max_parameter_change': max(temporal_changes_C) if temporal_changes_C else 0.0,
            'mean_parameter_change': sum(temporal_changes_C) / len(temporal_changes_C) if temporal_changes_C else 0.0
        },
        'fusion_modules': {
            'parameter_count': len(fusion_params_C),
            'total_parameters': sum(p.numel() for p in fusion_params_C),
            'all_have_grad': all(fusion_has_grad_C),
            'gradient_norms': fusion_grad_norms_C,
            'max_gradient_norm': max(fusion_grad_norms_C) if fusion_grad_norms_C else 0.0,
            'parameters_changed': [c > 1e-8 for c in fusion_changes_C],
            'max_parameter_change': max(fusion_changes_C) if fusion_changes_C else 0.0,
            'mean_parameter_change': sum(fusion_changes_C) / len(fusion_changes_C) if fusion_changes_C else 0.0
        },
        'spatial': {
            'parameter_count': len(list(video_unet_C.spatial_unet.parameters())),
            'max_change': max(spatial_changes_C) if spatial_changes_C else 0.0,
            'mean_change': sum(spatial_changes_C) / len(spatial_changes_C) if spatial_changes_C else 0.0,
            'any_changed': any(c > 1e-8 for c in spatial_changes_C)
        },
        'appearance': {
            'parameter_count': len(list(split_encoder_C.token_emb.original_emb.parameters())),
            'max_change': max(appearance_changes_C) if appearance_changes_C else 0.0,
            'mean_change': sum(appearance_changes_C) / len(appearance_changes_C) if appearance_changes_C else 0.0,
            'any_changed': any(c > 1e-8 for c in appearance_changes_C)
        },
        'optimizer': {
            'parameter_count': len(optimizer_C.param_groups[0]['params']),
            'all_gradients_finite': all(
                torch.isfinite(g).all() if g is not None else True
                for g in motion_grads_C + temporal_grads_C + fusion_grads_C
            )
        }
    }

    print(f"Motion embedding parameters: {len(motion_params_C)}")
    print(f"Motion has gradients: {all(motion_has_grad_C)}")
    print(f"Motion gradient norm: {motion_grad_norms_C[0]:.6f}")
    print(f"Motion parameters changed: {results['experiment_C']['motion_embedding']['parameters_changed']}")
    print(f"Motion max change: {results['experiment_C']['motion_embedding']['max_parameter_change']:.8f}")
    print(f"\nTemporal parameters: {len(temporal_params_C)}")
    print(f"Temporal has gradients: {all(temporal_has_grad_C)}")
    print(f"Temporal max gradient norm: {results['experiment_C']['temporal_modules']['max_gradient_norm']:.6f}")
    print(f"Temporal parameters changed: {results['experiment_C']['temporal_modules']['parameters_changed']}")
    print(f"Temporal max change: {results['experiment_C']['temporal_modules']['max_parameter_change']:.8f}")
    print(f"\nFusion parameters: {len(fusion_params_C)}")
    print(f"Fusion has gradients: {all(fusion_has_grad_C)}")
    print(f"Fusion max gradient norm: {results['experiment_C']['fusion_modules']['max_gradient_norm']:.6f}")
    print(f"Fusion parameters changed: {results['experiment_C']['fusion_modules']['parameters_changed']}")
    print(f"Fusion max change: {results['experiment_C']['fusion_modules']['max_parameter_change']:.8f}")
    print(f"\nSpatial any changed: {results['experiment_C']['spatial']['any_changed']}")
    print(f"Appearance any changed: {results['experiment_C']['appearance']['any_changed']}")

    # Save results
    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/STAGE3_2_9L2_ONE_STEP_UPDATE_RESULTS.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"Saved to: {output_file}")

    return results

if __name__ == "__main__":
    validate_one_step()
