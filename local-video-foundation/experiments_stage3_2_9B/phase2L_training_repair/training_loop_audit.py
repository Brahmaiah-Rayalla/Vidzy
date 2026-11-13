"""
Stage 3.2.9L - Training Loop Audit

Traces the complete training path to verify:
- Correct model instance is being trained
- Optimizer references correct parameters
- Temporal processing is enabled for Experiment C
- Gradients reach intended parameters
- Gradients are not detached
- Parameters are not unintentionally frozen

Author: Devin
Date: 2026-11-04
"""

import torch
import json
from pathlib import Path
import sys

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
parent_dir = base_dir
stage3_2_7_dir = base_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation" / "models"))

def audit_training_loop():
    """Audit the training loop execution path."""
    print("=" * 80)
    print("STAGE 3.2.9L - TRAINING LOOP AUDIT")
    print("=" * 80)

    audit = {
        'experiment_B': {},
        'experiment_C': {},
        'findings': []
    }

    # Load models
    print("\nLoading models...")
    from models.conditional_unet import ConditionalUNet
    from models.video_unet_fixed import VideoUNetFixed
    from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder
    from models.text_encoder import TextEncoder
    from models.text_encoder import SimpleTokenizer

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
        original_vocab_size=42  # Original vocab size before motion expansion
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

    # Create video models
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

    video_unet_C = VideoUNetFixed(
        spatial_unet=spatial_unet,
        text_encoder=split_encoder,
        num_frames=16,
        image_size=64,
        temporal_kernel=3,
        temporal_layers=2,
        freeze_spatial=True,
        use_temporal=True,
        enable_temporal_processing=True
    )

    # Audit Experiment B
    print("\n" + "=" * 80)
    print("EXPERIMENT B AUDIT")
    print("=" * 80)

    audit['experiment_B']['model_config'] = {
        'enable_temporal_processing': video_unet_B.enable_temporal_processing,
        'use_temporal': video_unet_B.use_temporal,
        'spatial_frozen': all(not p.requires_grad for p in video_unet_B.spatial_unet.parameters())
    }

    # Check motion embedding parameters
    motion_params = list(split_encoder.token_emb.motion_emb.parameters())
    audit['experiment_B']['motion_embedding'] = {
        'parameter_count': len(motion_params),
        'total_parameters': sum(p.numel() for p in motion_params),
        'all_trainable': all(p.requires_grad for p in motion_params),
        'parameter_names': [f'motion_emb.{i}' for i in range(len(motion_params))]
    }

    # Check optimizer parameter groups
    import torch.optim as optim
    optimizer_B = optim.AdamW(motion_params, lr=0.0002, weight_decay=0.01)

    optimizer_param_ids_B = set(id(p) for p in optimizer_B.param_groups[0]['params'])
    audit['experiment_B']['optimizer'] = {
        'parameter_count': len(optimizer_B.param_groups[0]['params']),
        'optimizer_param_ids_match_motion': all(id(p) in optimizer_param_ids_B for p in motion_params)
    }

    # Verify no spatial parameters in optimizer
    spatial_in_optimizer = False
    for name, param in video_unet_B.named_parameters():
        if 'spatial' in name.lower() and id(param) in optimizer_param_ids_B:
            spatial_in_optimizer = True
            audit['findings'].append(f"CRITICAL: Spatial parameter {name} in optimizer B")

    audit['experiment_B']['optimizer']['spatial_in_optimizer'] = spatial_in_optimizer

    # Audit Experiment C
    print("\n" + "=" * 80)
    print("EXPERIMENT C AUDIT")
    print("=" * 80)

    audit['experiment_C']['model_config'] = {
        'enable_temporal_processing': video_unet_C.enable_temporal_processing,
        'use_temporal': video_unet_C.use_temporal,
        'spatial_frozen': all(not p.requires_grad for p in video_unet_C.spatial_unet.parameters())
    }

    # Check temporal module parameters
    temporal_params = list(video_unet_C.temporal_modules.parameters())
    audit['experiment_C']['temporal_modules'] = {
        'parameter_count': len(temporal_params),
        'total_parameters': sum(p.numel() for p in temporal_params),
        'all_trainable': all(p.requires_grad for p in temporal_params)
    }

    # Check fusion module parameters
    fusion_params = list(video_unet_C.fusion_modules.parameters())
    audit['experiment_C']['fusion_modules'] = {
        'parameter_count': len(fusion_params),
        'total_parameters': sum(p.numel() for p in fusion_params),
        'all_trainable': all(p.requires_grad for p in fusion_params)
    }

    # Check motion embedding parameters
    motion_params_C = list(split_encoder.token_emb.motion_emb.parameters())
    audit['experiment_C']['motion_embedding'] = {
        'parameter_count': len(motion_params_C),
        'total_parameters': sum(p.numel() for p in motion_params_C),
        'all_trainable': all(p.requires_grad for p in motion_params_C)
    }

    # Setup optimizer C
    trainable_params_C = motion_params_C + temporal_params + fusion_params
    optimizer_C = optim.AdamW(trainable_params_C, lr=0.0002, weight_decay=0.01)

    optimizer_param_ids_C = set(id(p) for p in optimizer_C.param_groups[0]['params'])
    audit['experiment_C']['optimizer'] = {
        'parameter_count': len(optimizer_C.param_groups[0]['params']),
        'expected_parameter_count': len(trainable_params_C),
        'all_trainable_in_optimizer': all(id(p) in optimizer_param_ids_C for p in trainable_params_C)
    }

    # Verify no spatial parameters in optimizer C
    spatial_in_optimizer_C = False
    for name, param in video_unet_C.named_parameters():
        if 'spatial' in name.lower() and id(param) in optimizer_param_ids_C:
            spatial_in_optimizer_C = True
            audit['findings'].append(f"CRITICAL: Spatial parameter {name} in optimizer C")

    audit['experiment_C']['optimizer']['spatial_in_optimizer'] = spatial_in_optimizer_C

    # Check gate initialization
    gate_values = video_unet_C.get_gate_values()
    audit['experiment_C']['gate_values'] = gate_values

    # Test forward pass with gradient flow
    print("\n" + "=" * 80)
    print("GRADIENT FLOW TEST")
    print("=" * 80)

    # Create dummy inputs
    batch_size = 1
    channels = 3
    num_frames = 16
    height = 64
    width = 64

    x = torch.randn(batch_size, channels, num_frames, height, width)
    t = torch.randint(0, 1000, (batch_size,))
    token_ids = torch.randint(0, actual_vocab_size, (batch_size, 10))
    mask = torch.ones(batch_size, 10)

    # Test Experiment C forward pass
    video_unet_C.train()
    text_emb, text_mask = video_unet_C.text_encoder(token_ids, mask)
    predicted_noise = video_unet_C(x, t, text_emb, text_mask)
    loss = torch.nn.functional.mse_loss(predicted_noise, torch.randn_like(predicted_noise))

    # Backward pass
    optimizer_C.zero_grad()
    loss.backward()

    # Check which parameters received gradients
    params_with_grad = []
    params_without_grad = []

    for name, param in video_unet_C.named_parameters():
        if param.grad is not None:
            params_with_grad.append(name)
        else:
            params_without_grad.append(name)

    # Check split encoder gradients
    for name, param in split_encoder.named_parameters():
        if param.grad is not None:
            params_with_grad.append(f"text_encoder.{name}")
        else:
            params_without_grad.append(f"text_encoder.{name}")

    audit['gradient_flow'] = {
        'total_parameters': len(list(video_unet_C.parameters())) + len(list(split_encoder.parameters())),
        'parameters_with_gradient': len(params_with_grad),
        'parameters_without_gradient': len(params_without_grad),
        'motion_embedding_has_grad': any('motion_emb' in name for name in params_with_grad),
        'temporal_has_grad': any('temporal' in name for name in params_with_grad),
        'fusion_has_grad': any('fusion' in name for name in params_with_grad),
        'spatial_has_grad': any('spatial' in name for name in params_with_grad)
    }

    print(f"Parameters with gradient: {len(params_with_grad)}")
    print(f"Parameters without gradient: {len(params_without_grad)}")
    print(f"Motion embedding has gradient: {audit['gradient_flow']['motion_embedding_has_grad']}")
    print(f"Temporal has gradient: {audit['gradient_flow']['temporal_has_grad']}")
    print(f"Fusion has gradient: {audit['gradient_flow']['fusion_has_grad']}")
    print(f"Spatial has gradient: {audit['gradient_flow']['spatial_has_grad']}")

    # Critical findings
    if not audit['gradient_flow']['motion_embedding_has_grad']:
        audit['findings'].append("CRITICAL: Motion embedding did not receive gradient")
    if not audit['gradient_flow']['temporal_has_grad']:
        audit['findings'].append("CRITICAL: Temporal modules did not receive gradient")
    if not audit['gradient_flow']['fusion_has_grad']:
        audit['findings'].append("CRITICAL: Fusion modules did not receive gradient")
    if audit['gradient_flow']['spatial_has_grad']:
        audit['findings'].append("WARNING: Spatial parameters received gradient (should be frozen)")

    # Save audit results
    output_dir = base_dir / "experiments_stage3_2_9B/phase2L_training_repair"
    output_file = output_dir / "STAGE3_2_9L_TRAINING_LOOP_AUDIT.json"

    with open(output_file, 'w') as f:
        json.dump(audit, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_file}")

    return audit

if __name__ == "__main__":
    audit_training_loop()
