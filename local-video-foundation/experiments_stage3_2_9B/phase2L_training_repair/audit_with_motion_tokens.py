"""
Stage 3.2.9L - Training Loop Audit with Motion Tokens

This version uses actual motion tokens to ensure motion embeddings are activated.
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

def audit_with_motion_tokens():
    """Audit with explicit motion tokens."""
    print("=" * 80)
    print("STAGE 3.2.9L - TRAINING LOOP AUDIT WITH MOTION TOKENS")
    print("=" * 80)

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
        freeze_transformer=True
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

    # Create video model C
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

    # Setup optimizer
    motion_params = list(split_encoder.token_emb.motion_emb.parameters())
    temporal_params = list(video_unet_C.temporal_modules.parameters())
    fusion_params = list(video_unet_C.fusion_modules.parameters())
    trainable_params_C = motion_params + temporal_params + fusion_params

    import torch.optim as optim
    optimizer_C = optim.AdamW(trainable_params_C, lr=0.0002, weight_decay=0.01)

    # Test forward pass with GRADIENT FLOW
    print("\n" + "=" * 80)
    print("GRADIENT FLOW TEST WITH MOTION TOKENS")
    print("=" * 80)

    # Create dummy inputs with MOTION TOKENS
    batch_size = 1
    channels = 3
    num_frames = 16
    height = 64
    width = 64

    x = torch.randn(batch_size, channels, num_frames, height, width)
    t = torch.randint(0, 1000, (batch_size,))

    # IMPORTANT: Use motion tokens (IDs >= 42, since original_vocab_size = 42)
    # Mix of original tokens (0-41) and motion tokens (42-81)
    token_ids = torch.tensor([
        [5, 10, 42, 43, 15, 44, 20, 45, 25, 50]  # Contains motion tokens: 42, 43, 44, 45, 50
    ])
    mask = torch.ones(batch_size, 10)

    print(f"Token IDs: {token_ids}")
    print(f"Motion tokens present: {(token_ids >= 42).any().item()}")
    print(f"Number of motion tokens: {(token_ids >= 42).sum().item()}")

    # Test forward pass
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

    # Check motion embedding specifically
    motion_emb_grad = split_encoder.token_emb.motion_emb.weight.grad
    print(f"\nMotion embedding gradient: {motion_emb_grad is not None}")
    if motion_emb_grad is not None:
        print(f"Motion embedding gradient norm: {motion_emb_grad.norm().item()}")
        print(f"Motion embedding gradient sum: {motion_emb_grad.sum().item()}")

    # Check original embedding (should be frozen)
    original_emb_grad = split_encoder.token_emb.original_emb.weight.grad
    print(f"Original embedding gradient: {original_emb_grad is not None}")

    # Summary
    print("\n" + "=" * 80)
    print("GRADIENT FLOW SUMMARY")
    print("=" * 80)
    print(f"Parameters with gradient: {len(params_with_grad)}")
    print(f"Parameters without gradient: {len(params_without_grad)}")
    print(f"Motion embedding has gradient: {motion_emb_grad is not None}")
    print(f"Original embedding has gradient: {original_emb_grad is not None}")
    print(f"Temporal has gradient: {any('temporal' in name for name in params_with_grad)}")
    print(f"Fusion has gradient: {any('fusion' in name for name in params_with_grad)}")
    print(f"Spatial has gradient: {any('spatial' in name for name in params_with_grad)}")

    # List all parameters with gradients
    print("\nParameters with gradients:")
    for name in params_with_grad:
        print(f"  - {name}")

    if not motion_emb_grad:
        print("\n" + "=" * 80)
        print("CRITICAL: Motion embedding did not receive gradient even with motion tokens!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("SUCCESS: Motion embedding received gradient")
        print("=" * 80)

if __name__ == "__main__":
    audit_with_motion_tokens()
