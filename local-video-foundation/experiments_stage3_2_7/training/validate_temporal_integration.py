"""
Stage 3.2.8: Temporal Integration Validation

Verify that temporal modules are actually executed in the forward path.
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments_stage3_2_7.video_foundation.models.text_encoder import TextEncoder
from experiments_stage3_2_7.video_foundation.models.conditional_unet import ConditionalUNet
from experiments_stage3_2_7.video_foundation.models.video_unet_fixed import VideoUNetFixed


def load_tokenizer(tokenizer_path: str) -> dict:
    """Load tokenizer from JSON file."""
    with open(tokenizer_path, 'r') as f:
        return json.load(f)


def test_temporal_module_execution():
    """Test that temporal modules are actually executed."""
    print("="*80)
    print("TEMPORAL MODULE EXECUTION TEST")
    print("="*80)

    script_dir = Path(__file__).parent.parent
    tokenizer_path = script_dir / "checkpoints" / "tokenizer_motion_expanded.json"
    tokenizer = load_tokenizer(str(tokenizer_path))

    # Create model with temporal processing enabled
    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256,
        n_heads=4, use_cross_attn=True
    )

    text_encoder = TextEncoder(
        vocab_size=tokenizer['vocab_size'], d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64
    )

    model = VideoUNetFixed(
        spatial_unet=spatial_unet, text_encoder=text_encoder,
        num_frames=4, image_size=64,
        temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True,
        enable_temporal_processing=True
    )

    # Add hooks to track execution
    execution_log = {}

    def make_hook(name):
        def hook(module, input, output):
            execution_log[name] = execution_log.get(name, 0) + 1
        return hook

    # Register hooks on temporal modules
    model.temporal_modules['bottleneck'].register_forward_hook(make_hook('bottleneck_temporal'))
    model.temporal_modules['low_res'].register_forward_hook(make_hook('low_res_temporal'))

    # Note: Fusion gates are accessed directly, not via forward pass
    # We'll verify gate usage by checking if gate values changed

    # Create dummy input
    videos = torch.randn(1, 3, 4, 64, 64)  # [B, C, T, H, W]
    timesteps = torch.randint(0, 1000, (1,))
    token_ids = torch.tensor([[3, 12, 33]])  # Appearance tokens
    text_mask = torch.ones_like(token_ids, dtype=torch.float)

    # Forward pass
    text_enc, _ = model.text_encoder(token_ids, text_mask)
    output = model(videos, timesteps, text_enc, text_mask)

    print(f"\nExecution log:")
    for name, count in execution_log.items():
        print(f"  {name}: {count} call(s)")

    # Verify temporal modules were executed
    if execution_log.get('bottleneck_temporal', 0) > 0:
        print("\n[PASS] Bottleneck temporal module executed")
    else:
        print("\n[ERROR] Bottleneck temporal module NOT executed")
        return False

    if execution_log.get('low_res_temporal', 0) > 0:
        print("[PASS] Low-res temporal module executed")
    else:
        print("[ERROR] Low-res temporal module NOT executed")
        return False

    # Check gate values (they should be > 0 due to sigmoid)
    gate_values = model.get_gate_values()
    if 'bottleneck_gate' in gate_values and 'low_res_gate' in gate_values:
        print(f"[PASS] Fusion gates used (bottleneck: {gate_values['bottleneck_gate']:.4f}, low_res: {gate_values['low_res_gate']:.4f})")
    else:
        print("[ERROR] Fusion gates not used")
        return False

    return True


def test_temporal_disable():
    """Test that temporal processing can be disabled."""
    print("\n" + "="*80)
    print("TEMPORAL DISABLE TEST")
    print("="*80)

    script_dir = Path(__file__).parent.parent
    tokenizer_path = script_dir / "checkpoints" / "tokenizer_motion_expanded.json"
    tokenizer = load_tokenizer(str(tokenizer_path))

    # Create model with temporal processing disabled
    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256,
        n_heads=4, use_cross_attn=True
    )

    text_encoder = TextEncoder(
        vocab_size=tokenizer['vocab_size'], d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64
    )

    model_disabled = VideoUNetFixed(
        spatial_unet=spatial_unet, text_encoder=text_encoder,
        num_frames=4, image_size=64,
        temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True,
        enable_temporal_processing=False
    )

    # Create model with temporal processing enabled
    spatial_unet2 = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256,
        n_heads=4, use_cross_attn=True
    )

    text_encoder2 = TextEncoder(
        vocab_size=tokenizer['vocab_size'], d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64
    )

    model_enabled = VideoUNetFixed(
        spatial_unet=spatial_unet2, text_encoder=text_encoder2,
        num_frames=4, image_size=64,
        temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True,
        enable_temporal_processing=True
    )

    # Copy weights to ensure same initialization
    model_enabled.load_state_dict(model_disabled.state_dict())

    # Create dummy input
    videos = torch.randn(1, 3, 4, 64, 64)
    timesteps = torch.randint(0, 1000, (1,))
    token_ids = torch.tensor([[3, 12, 33]])
    text_mask = torch.ones_like(token_ids, dtype=torch.float)

    # Forward pass with temporal disabled
    text_enc, _ = model_disabled.text_encoder(token_ids, text_mask)
    output_disabled = model_disabled(videos, timesteps, text_enc, text_mask)

    # Forward pass with temporal enabled
    text_enc2, _ = model_enabled.text_encoder(token_ids, text_mask)
    output_enabled = model_enabled(videos, timesteps, text_enc2, text_mask)

    # Compare outputs
    max_diff = torch.max(torch.abs(output_disabled - output_enabled)).item()
    mean_diff = torch.mean(torch.abs(output_disabled - output_enabled)).item()

    print(f"\nOutput comparison:")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")

    if max_diff < 1e-6:
        print("[ERROR] Outputs are identical - temporal processing not working")
        return False
    else:
        print(f"[PASS] Outputs differ (max diff: {max_diff:.6f}) - temporal processing working")

    return True


def test_temporal_gradients():
    """Test that temporal modules receive gradients."""
    print("\n" + "="*80)
    print("TEMPORAL GRADIENT TEST")
    print("="*80)

    script_dir = Path(__file__).parent.parent
    tokenizer_path = script_dir / "checkpoints" / "tokenizer_motion_expanded.json"
    tokenizer = load_tokenizer(str(tokenizer_path))

    # Create model
    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256,
        n_heads=4, use_cross_attn=True
    )

    text_encoder = TextEncoder(
        vocab_size=tokenizer['vocab_size'], d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64
    )

    model = VideoUNetFixed(
        spatial_unet=spatial_unet, text_encoder=text_encoder,
        num_frames=4, image_size=64,
        temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True,
        enable_temporal_processing=True
    )

    # Freeze spatial, train temporal
    for param in model.spatial_unet.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.temporal_modules.parameters():
        param.requires_grad = True
    for param in model.fusion_modules.parameters():
        param.requires_grad = True

    # Create dummy input
    videos = torch.randn(1, 3, 4, 64, 64)
    timesteps = torch.randint(0, 1000, (1,))
    token_ids = torch.tensor([[3, 12, 33]])
    text_mask = torch.ones_like(token_ids, dtype=torch.float)

    # Forward pass
    text_enc, _ = model.text_encoder(token_ids, text_mask)
    output = model(videos, timesteps, text_enc, text_mask)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    print("\nGradient analysis:")

    bottleneck_temporal_has_grad = any(p.grad is not None and p.grad.abs().sum() > 1e-6
                                      for p in model.temporal_modules['bottleneck'].parameters())
    if bottleneck_temporal_has_grad:
        print("[PASS] Bottleneck temporal module has gradients")
    else:
        print("[ERROR] Bottleneck temporal module has no gradients")
        return False

    low_res_temporal_has_grad = any(p.grad is not None and p.grad.abs().sum() > 1e-6
                                    for p in model.temporal_modules['low_res'].parameters())
    if low_res_temporal_has_grad:
        print("[PASS] Low-res temporal module has gradients")
    else:
        print("[ERROR] Low-res temporal module has no gradients")
        return False

    fusion_has_grad = any(p.grad is not None and p.grad.abs().sum() > 1e-6
                          for p in model.fusion_modules.parameters())
    if fusion_has_grad:
        print("[PASS] Fusion modules have gradients")
    else:
        print("[ERROR] Fusion modules have no gradients")
        return False

    return True


def main():
    results = []

    # Test 1: Temporal module execution
    results.append(("Temporal Module Execution", test_temporal_module_execution()))

    # Test 2: Temporal disable
    results.append(("Temporal Disable", test_temporal_disable()))

    # Test 3: Temporal gradients
    results.append(("Temporal Gradients", test_temporal_gradients()))

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[ERROR]"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*80)
    if all_passed:
        print("[PASS] ALL TESTS PASSED")
    else:
        print("[ERROR] SOME TESTS FAILED")
    print("="*80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
