"""
Architecture audit for Stage 3.2.

Thoroughly inspects the video U-Net architecture, temporal modules,
gradient flow, and compatibility with Stage 2 checkpoints.
"""

import os
import sys
import torch
import torch.nn as nn
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def analyze_temporal_modules(video_unet):
    """Analyze temporal module structure and parameters."""
    print("\n" + "=" * 60)
    print("TEMPORAL MODULE ANALYSIS")
    print("=" * 60)

    if not video_unet.use_temporal:
        print("Temporal modules are DISABLED")
        return

    temporal_params = defaultdict(int)
    temporal_modules = defaultdict(list)

    for name, param in video_unet.named_parameters():
        if 'temporal' in name:
            temporal_params[name] = param.numel()
            module_name = name.split('.')[0]
            temporal_modules[module_name].append((name, param.numel()))

    print(f"\nTotal temporal parameters: {sum(temporal_params.values()):,}")
    print(f"Number of temporal parameter tensors: {len(temporal_params)}")

    print("\nBreakdown by module:")
    for module_name, params in temporal_modules.items():
        module_total = sum(p[1] for p in params)
        print(f"\n  {module_name}:")
        print(f"    Total parameters: {module_total:,}")
        print(f"    Parameter tensors: {len(params)}")
        for param_name, param_count in params:
            print(f"      {param_name}: {param_count:,}")

    return temporal_params


def check_gradient_flow(video_unet, device='cpu'):
    """Check gradient flow through temporal modules."""
    print("\n" + "=" * 60)
    print("GRADIENT FLOW ANALYSIS")
    print("=" * 60)

    video_unet.train()

    # Create dummy inputs
    batch_size = 1
    num_frames = 16
    image_size = 64
    shape = (batch_size, 3, num_frames, image_size, image_size)

    x = torch.randn(shape, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    text_emb = torch.randn(batch_size, 32, 256, device=device)
    text_mask = torch.ones(batch_size, 32, device=device)

    # Forward pass
    output = video_unet(x, t, text_emb, text_mask)

    # Compute loss
    loss = output.mean()

    # Backward pass
    loss.backward()

    # Check gradients
    temporal_params_with_grad = 0
    temporal_params_without_grad = 0
    temporal_tensors_with_grad = 0
    temporal_tensors_without_grad = 0

    gradient_norms = defaultdict(list)

    for name, param in video_unet.named_parameters():
        if 'temporal' in name:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms[name].append(grad_norm)
                temporal_params_with_grad += param.numel()
                temporal_tensors_with_grad += 1
            else:
                temporal_params_without_grad += param.numel()
                temporal_tensors_without_grad += 1

    total_temporal_params = temporal_params_with_grad + temporal_params_without_grad
    total_temporal_tensors = temporal_tensors_with_grad + temporal_tensors_without_grad

    print(f"\nTemporal parameter gradient status:")
    print(f"  Total temporal parameters: {total_temporal_params:,}")
    print(f"  Parameters with gradients: {temporal_params_with_grad:,} ({100*temporal_params_with_grad/total_temporal_params:.1f}%)")
    print(f"  Parameters without gradients: {temporal_params_without_grad:,} ({100*temporal_params_without_grad/total_temporal_params:.1f}%)")

    print(f"\nTemporal tensor gradient status:")
    print(f"  Total temporal parameter tensors: {total_temporal_tensors}")
    print(f"  Tensors with gradients: {temporal_tensors_with_grad} ({100*temporal_tensors_with_grad/total_temporal_tensors:.1f}%)")
    print(f"  Tensors without gradients: {temporal_tensors_without_grad} ({100*temporal_tensors_without_grad/total_temporal_tensors:.1f}%)")

    print(f"\nGradient norms by temporal block:")
    for name, norms in gradient_norms.items():
        block_name = name.split('.')[1] if len(name.split('.')) > 1 else 'unknown'
        avg_norm = np.mean(norms)
        print(f"  {name}: {avg_norm:.6f}")

    return {
        'total_params': total_temporal_params,
        'params_with_grad': temporal_params_with_grad,
        'params_without_grad': temporal_params_without_grad,
        'total_tensors': total_temporal_tensors,
        'tensors_with_grad': temporal_tensors_with_grad,
        'tensors_without_grad': temporal_tensors_without_grad,
        'gradient_norms': gradient_norms
    }


def check_spatial_compatibility(video_unet, checkpoint_path):
    """Check spatial backbone compatibility with Stage 2 checkpoint."""
    print("\n" + "=" * 60)
    print("SPATIAL BACKBONE COMPATIBILITY")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Get spatial parameter names
    spatial_params = set()
    for name, _ in video_unet.spatial_unet.named_parameters():
        spatial_params.add(name)

    # Get checkpoint keys
    checkpoint_keys = set(state_dict.keys())

    # Check for matches
    matching_keys = spatial_params & checkpoint_keys
    missing_in_checkpoint = spatial_params - checkpoint_keys
    extra_in_checkpoint = checkpoint_keys - spatial_params

    print(f"\nSpatial parameters in model: {len(spatial_params)}")
    print(f"Keys in checkpoint: {len(checkpoint_keys)}")
    print(f"Matching keys: {len(matching_keys)}")
    print(f"Missing in checkpoint: {len(missing_in_checkpoint)}")
    print(f"Extra in checkpoint: {len(extra_in_checkpoint)}")

    if missing_in_checkpoint:
        print(f"\nSample missing keys (first 5):")
        for key in list(missing_in_checkpoint)[:5]:
            print(f"  {key}")

    if extra_in_checkpoint:
        print(f"\nSample extra keys (first 5):")
        for key in list(extra_in_checkpoint)[:5]:
            print(f"  {key}")

    # Try loading
    print("\nAttempting to load spatial weights...")
    try:
        video_unet.spatial_unet.load_state_dict(state_dict, strict=False)
        print("✓ Spatial weights loaded successfully (non-strict mode)")
    except Exception as e:
        print(f"✗ Error loading spatial weights: {e}")


def check_text_conditioning(video_unet, device='cpu'):
    """Check text conditioning integration."""
    print("\n" + "=" * 60)
    print("TEXT CONDITIONING CHECK")
    print("=" * 60)

    # Create dummy inputs
    batch_size = 1
    num_frames = 16
    image_size = 64
    shape = (batch_size, 3, num_frames, image_size, image_size)

    x = torch.randn(shape, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)

    # Test with different text embeddings
    text_emb1 = torch.randn(batch_size, 32, 256, device=device)
    text_emb2 = torch.randn(batch_size, 32, 256, device=device)
    text_mask = torch.ones(batch_size, 32, device=device)

    video_unet.eval()
    with torch.no_grad():
        output1 = video_unet(x, t, text_emb1, text_mask)
        output2 = video_unet(x, t, text_emb2, text_mask)

    # Check if different text produces different outputs
    diff = (output1 - output2).abs().mean().item()
    print(f"\nText conditioning sensitivity: {diff:.6f}")
    print(f"Expected: > 0.001 (different text should produce different outputs)")

    if diff > 0.001:
        print("Text conditioning is working")
    else:
        print("Text conditioning may not be working properly")


def check_temporal_integration(video_unet, device='cpu'):
    """Check how temporal features are integrated."""
    print("\n" + "=" * 60)
    print("TEMPORAL INTEGRATION CHECK")
    print("=" * 60)

    # Check forward_with_temporal implementation
    print("\nChecking _forward_with_temporal implementation...")

    # Read the source code to verify integration
    import inspect
    source = inspect.getsource(video_unet._forward_with_temporal)

    checks = {
        'bottleneck_temporal': 'bottleneck' in source and 'temporal_modules' in source,
        'low_res_temporal': 'low_res' in source and 'temporal_modules' in source,
        'temporal_output_used': 'bottleneck_video' in source or 'low_res_video' in source,
        'residual_connection': '+' in source or 'residual' in source.lower(),
    }

    print("\nIntegration checks:")
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  {status} {check_name}: {result}")

    # Check if temporal modules are called in every forward pass
    print("\nTemporal module call pattern:")
    if 'self.temporal_modules[' in source:
        print("  Temporal modules are explicitly called")
        # Count calls
        call_count = source.count('self.temporal_modules[')
        print(f"  Number of temporal module calls: {call_count}")
    else:
        print("  Temporal modules may not be called")


def check_frame_order_sensitivity(video_unet, device='cpu'):
    """Check sensitivity to frame order."""
    print("\n" + "=" * 60)
    print("FRAME ORDER SENSITIVITY")
    print("=" * 60)

    batch_size = 1
    num_frames = 16
    image_size = 64
    shape = (batch_size, 3, num_frames, image_size, image_size)

    x = torch.randn(shape, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    text_emb = torch.randn(batch_size, 32, 256, device=device)
    text_mask = torch.ones(batch_size, 32, device=device)

    video_unet.eval()
    with torch.no_grad():
        # Original order
        output_original = video_unet(x, t, text_emb, text_mask)

        # Reversed order
        x_reversed = torch.flip(x, dims=[2])  # Reverse frame dimension
        output_reversed = video_unet(x_reversed, t, text_emb, text_mask)

        # Random permutation
        perm = torch.randperm(num_frames)
        x_permuted = x[:, :, perm, :, :]
        output_permuted = video_unet(x_permuted, t, text_emb, text_mask)

    # Compute differences
    diff_reversed = (output_original - output_reversed).abs().mean().item()
    diff_permuted = (output_original - output_permuted).abs().mean().item()

    print(f"\nOriginal vs reversed frame order: {diff_reversed:.6f}")
    print(f"Original vs permuted frame order: {diff_permuted:.6f}")

    if diff_reversed > 0.01:
        print("Model is sensitive to frame order (reversed)")
    else:
        print("Model may not be sensitive to frame order (reversed)")

    if diff_permuted > 0.01:
        print("Model is sensitive to frame order (permuted)")
    else:
        print("Model may not be sensitive to frame order (permuted)")


def run_full_audit(spatial_checkpoint_path=None):
    """Run complete architecture audit."""
    print("=" * 60)
    print("STAGE 3.2 ARCHITECTURE AUDIT")
    print("=" * 60)

    device = 'cpu'

    # Load tokenizer
    tokenizer_path = "data/text_image/tokenizer/tokenizer.json"
    with open(tokenizer_path, 'r') as f:
        data = json.load(f)

    idx_to_word = {}
    word_to_idx = {}
    for key, value in data['idx_to_word'].items():
        if key.startswith('<'):
            idx = int(value)
            token = key
            idx_to_word[idx] = token
            word_to_idx[token] = idx
        else:
            idx = int(key)
            token = value
            idx_to_word[idx] = token
            word_to_idx[token] = idx

    vocab_size = len(word_to_idx)

    # Create models
    print("\nCreating models...")
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

    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    video_unet = video_unet.to(device)

    # Create tokenizer
    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()

    # Parameter counts
    total, trainable, frozen = count_parameters(video_unet)
    print(f"\nOverall parameter counts:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {frozen:,}")

    # Analyze temporal modules
    temporal_params = analyze_temporal_modules(video_unet)

    # Check gradient flow
    gradient_info = check_gradient_flow(video_unet, device)

    # Check spatial compatibility
    if spatial_checkpoint_path:
        check_spatial_compatibility(video_unet, spatial_checkpoint_path)

    # Check text conditioning
    check_text_conditioning(video_unet, device)

    # Check temporal integration
    check_temporal_integration(video_unet)

    # Check frame order sensitivity
    check_frame_order_sensitivity(video_unet, device)

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    print(f"\nTemporal parameters: {sum(temporal_params.values()):,}")
    print(f"Temporal tensors with gradients: {gradient_info['tensors_with_grad']}/{gradient_info['total_tensors']}")
    print(f"Temporal parameters with gradients: {gradient_info['params_with_grad']:,}/{gradient_info['total_params']:,}")

    # Save audit results
    audit_results = {
        'total_parameters': int(total),
        'trainable_parameters': int(trainable),
        'frozen_parameters': int(frozen),
        'temporal_parameters': int(sum(temporal_params.values())),
        'temporal_tensors_with_grad': gradient_info['tensors_with_grad'],
        'temporal_tensors_without_grad': gradient_info['tensors_without_grad'],
        'temporal_params_with_grad': int(gradient_info['params_with_grad']),
        'temporal_params_without_grad': int(gradient_info['params_without_grad']),
        'gradient_norms': {k: [float(v) for v in vals] for k, vals in gradient_info['gradient_norms'].items()}
    }

    with open('architecture_audit_results.json', 'w') as f:
        json.dump(audit_results, f, indent=2)

    print("\nAudit results saved to architecture_audit_results.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Architecture audit for Stage 3.2")
    parser.add_argument("--spatial_checkpoint", type=str, default="checkpoints_conditional_minimal/checkpoint_step_000010.pt",
                        help="Path to Stage 2 spatial checkpoint")

    args = parser.parse_args()

    run_full_audit(spatial_checkpoint_path=args.spatial_checkpoint)
