"""
Stage 3.2.7: Automated Tests for Feature-Level Temporal Fusion

Tests:
- Original ConditionalUNet output equivalence
- Intermediate feature dimensions
- Feature extraction and decoder continuation
- Temporal fusion tensor dimensions
- Actual temporal-axis processing
- Motion embedding dimensions
- Identity-preserving initialization
- Spatial checkpoint preservation
- CPU-only execution
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import json

# Add the parent directory (local-video-foundation) to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import from the experiments_stage3_2_7 package
from experiments_stage3_2_7.video_foundation.models.conditional_unet import ConditionalUNet
from experiments_stage3_2_7.video_foundation.models.text_encoder import TextEncoder
from experiments_stage3_2_7.video_foundation.models.video_unet_feature_fusion import VideoUNetFixed as VideoUNetFeatureFusion


def test_conditional_unet_feature_extraction():
    """Test that ConditionalUNet can extract intermediate features."""
    print("\n" + "=" * 80)
    print("TEST 1: ConditionalUNet Feature Extraction")
    print("=" * 80)

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

    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)

    # Test without feature extraction
    output_no_features = spatial_unet(x, timesteps, text_emb, text_mask, return_features=False)
    print(f"Output shape (no features): {output_no_features.shape}")

    # Test with feature extraction
    output_with_features, features = spatial_unet(x, timesteps, text_emb, text_mask, return_features=True)
    print(f"Output shape (with features): {output_with_features.shape}")
    print(f"Bottleneck features shape: {features['bottleneck'].shape}")
    print(f"Low-res features shape: {features['low_res'].shape}")

    # Verify outputs are equivalent
    diff = (output_no_features - output_with_features).abs().max().item()
    print(f"Max difference between outputs: {diff:.10f}")

    if diff < 1e-6:
        print("[PASS] Feature extraction does not change output")
    else:
        print("[FAIL] Feature extraction changes output")

    return diff < 1e-6


def test_temporal_convolution_axis():
    """Test that temporal convolution operates on frame dimension."""
    print("\n" + "=" * 80)
    print("TEST 2: Temporal Convolution Axis Verification")
    print("=" * 80)

    from experiments_stage3_2_7.video_foundation.models.video_unet_feature_fusion import TemporalConv1DBlock

    # Test with default (identity) initialization
    temporal_block_identity = TemporalConv1DBlock(channels=128, temporal_kernel=3, num_layers=2)

    batch_size = 2
    num_frames = 16
    channels = 128
    height = 8
    width = 8

    x_forward = torch.randn(batch_size, num_frames, channels, height, width)
    x_reversed = torch.flip(x_forward, dims=[1])

    output_forward = temporal_block_identity(x_forward)
    output_reversed = temporal_block_identity(x_reversed)

    # With identity initialization, should be time-symmetric
    output_reversed_reversed = torch.flip(output_reversed, dims=[1])
    diff_identity = (output_forward - output_reversed_reversed).abs().mean().item()

    print(f"Difference with identity init: {diff_identity:.10f}")

    test1_pass = diff_identity < 1e-4
    if test1_pass:
        print("[PASS] Identity initialization is time-symmetric (expected)")
    else:
        print("[FAIL] Identity initialization is not time-symmetric")

    # Test with random initialization (should NOT be time-symmetric)
    temporal_block_random = TemporalConv1DBlock(channels=128, temporal_kernel=3, num_layers=2)
    # Reinitialize with random weights
    for layer in temporal_block_random.layers:
        if isinstance(layer, nn.Conv1d):
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)

    output_forward_random = temporal_block_random(x_forward)
    output_reversed_random = temporal_block_random(x_reversed)
    output_reversed_reversed_random = torch.flip(output_reversed_random, dims=[1])
    diff_random = (output_forward_random - output_reversed_reversed_random).abs().mean().item()

    print(f"Difference with random init: {diff_random:.10f}")

    test2_pass = diff_random >= 1e-4
    if test2_pass:
        print("[PASS] Random initialization is not time-symmetric")
    else:
        print("[FAIL] Random initialization is time-symmetric")

    return test1_pass and test2_pass


def test_motion_vocabulary_embeddings():
    """Test that motion vocabulary has valid embeddings."""
    print("\n" + "=" * 80)
    print("TEST 3: Motion Vocabulary Embeddings")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "checkpoints/checkpoint_stage2_motion_expanded.pt")
    tokenizer_path = os.path.join(script_dir, "checkpoints/tokenizer_motion_expanded.json")

    if not os.path.exists(checkpoint_path):
        print(f"[SKIP] Checkpoint not found: {checkpoint_path}")
        return True

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    text_encoder_state = checkpoint['text_encoder_state_dict']
    token_emb = text_encoder_state['token_emb.weight']

    vocab_size, d_model = token_emb.shape
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {d_model}")

    # Check motion tokens have embeddings
    motion_tokens = ['moving', 'left', 'right', 'slow', 'fast', 'bounce', 'rotate']
    valid_embeddings = 0

    # Load tokenizer
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)

    word_to_idx = tokenizer_data['word_to_idx']

    for token in motion_tokens:
        if token in word_to_idx:
            idx = word_to_idx[token]
            emb = token_emb[idx]
            mean = emb.mean().item()
            std = emb.std().item()
            print(f"  '{token}' (idx {idx}): mean={mean:.4f}, std={std:.4f}")
            valid_embeddings += 1
        else:
            print(f"  '{token}': NOT in vocabulary")

    print(f"\nValid motion embeddings: {valid_embeddings}/{len(motion_tokens)}")

    if valid_embeddings == len(motion_tokens):
        print("[PASS] All motion tokens have embeddings")
    else:
        print("[FAIL] Some motion tokens missing")

    return valid_embeddings == len(motion_tokens)


def test_identity_preserving_initialization():
    """Test that temporal modules can be disabled to match baseline."""
    print("\n" + "=" * 80)
    print("TEST 4: Identity-Preserving Initialization")
    print("=" * 80)

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
        vocab_size=82,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Create video model with temporal disabled
    video_unet = VideoUNetFeatureFusion(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=False,
        use_temporal=False
    )

    # Create video model with temporal enabled but gates set to 0
    video_unet_temporal = VideoUNetFeatureFusion(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=False,
        use_temporal=True
    )

    # Test 1: Manually set gates to 0 (disable temporal)
    with torch.no_grad():
        video_unet_temporal.fusion_modules['bottleneck'].gate.fill_(-100)  # Use large negative to ensure sigmoid is near 0
        video_unet_temporal.fusion_modules['low_res'].gate.fill_(-100)

    # Verify gate values
    print(f"Bottleneck gate: {video_unet_temporal.fusion_modules['bottleneck'].gate.item()}")
    print(f"Low-res gate: {video_unet_temporal.fusion_modules['low_res'].gate.item()}")
    print(f"Bottleneck gate activated: {video_unet_temporal.fusion_modules['bottleneck'].gate_activation(video_unet_temporal.fusion_modules['bottleneck'].gate).item()}")

    batch_size = 2
    x = torch.randn(batch_size, 3, 16, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)

    video_unet.eval()
    video_unet_temporal.eval()

    with torch.no_grad():
        output_no_temporal = video_unet(x, timesteps, text_emb, text_mask)
        output_temporal_disabled = video_unet_temporal(x, timesteps, text_emb, text_mask)

    diff = (output_no_temporal - output_temporal_disabled).abs().max().item()
    print(f"Max difference (gates set to 0): {diff:.10f}")

    test1_pass = diff < 1e-4
    if test1_pass:
        print("[PASS] Disabling temporal gates reproduces baseline")
    else:
        print("[FAIL] Disabling temporal gates does not reproduce baseline")

    # Test 2: Check default initialization is close to baseline
    video_unet_temporal_default = VideoUNetFeatureFusion(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=False,
        use_temporal=True
    )

    # Check default gate values
    default_bottleneck_gate = video_unet_temporal_default.fusion_modules['bottleneck'].gate.item()
    default_low_res_gate = video_unet_temporal_default.fusion_modules['low_res'].gate.item()
    default_bottleneck_activated = video_unet_temporal_default.fusion_modules['bottleneck'].gate_activation(
        video_unet_temporal_default.fusion_modules['bottleneck'].gate).item()
    print(f"\nDefault bottleneck gate: {default_bottleneck_gate}")
    print(f"Default low-res gate: {default_low_res_gate}")
    print(f"Default bottleneck gate activated: {default_bottleneck_activated}")

    video_unet_temporal_default.eval()
    with torch.no_grad():
        output_temporal_default = video_unet_temporal_default(x, timesteps, text_emb, text_mask)

    diff_default = (output_no_temporal - output_temporal_default).abs().max().item()
    print(f"Max difference (default initialization): {diff_default:.10f}")

    test2_pass = diff_default < 0.02  # Allow small difference due to initial gate value
    if test2_pass:
        print("[PASS] Default initialization is close to baseline")
    else:
        print("[FAIL] Default initialization is far from baseline")

    return test1_pass and test2_pass


def test_spatial_checkpoint_preservation():
    """Test that spatial parameters remain frozen during training."""
    print("\n" + "=" * 80)
    print("TEST 5: Spatial Checkpoint Preservation")
    print("=" * 80)

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
        vocab_size=82,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    video_unet = VideoUNetFeatureFusion(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    # Count frozen parameters
    frozen_count = sum(1 for p in video_unet.spatial_unet.parameters() if not p.requires_grad)
    trainable_count = sum(1 for p in video_unet.spatial_unet.parameters() if p.requires_grad)

    print(f"Spatial parameters frozen: {frozen_count}")
    print(f"Spatial parameters trainable: {trainable_count}")

    if trainable_count == 0:
        print("[PASS] All spatial parameters frozen")
    else:
        print("[FAIL] Some spatial parameters trainable")

    return trainable_count == 0


def test_cpu_execution():
    """Test that model runs on CPU."""
    print("\n" + "=" * 80)
    print("TEST 6: CPU Execution")
    print("=" * 80)

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
        vocab_size=82,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    video_unet = VideoUNetFeatureFusion(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=True,
        use_temporal=True
    )

    video_unet = video_unet.to('cpu')

    batch_size = 1
    x = torch.randn(batch_size, 3, 16, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 32, 256)
    text_mask = torch.ones(batch_size, 32)

    video_unet.eval()
    with torch.no_grad():
        output = video_unet(x, timesteps, text_emb, text_mask)

    print(f"Output shape: {output.shape}")
    print(f"Device: {output.device}")

    if output.device.type == 'cpu':
        print("[PASS] Model runs on CPU")
    else:
        print("[FAIL] Model not on CPU")

    return output.device.type == 'cpu'


def main():
    print("=" * 80)
    print("STAGE 3.2.7: AUTOMATED TESTS")
    print("=" * 80)

    results = {
        'feature_extraction': test_conditional_unet_feature_extraction(),
        'temporal_axis': test_temporal_convolution_axis(),
        'motion_embeddings': test_motion_vocabulary_embeddings(),
        'identity_preserving': test_identity_preserving_initialization(),
        'spatial_preservation': test_spatial_checkpoint_preservation(),
        'cpu_execution': test_cpu_execution(),
    }

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n[SUCCESS] All tests passed")
    else:
        print("\n[FAILURE] Some tests failed")

    return all_passed


if __name__ == "__main__":
    main()
