"""
Stage 3.2.7C: Split Embedding Validation Script

Comprehensive validation of split embedding conversion including:
1. Token ID mapping verification
2. Embedding dimension verification
3. Round-trip conversion test
4. Gradient flow verification
5. Checkpoint compatibility
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments_stage3_2_7.video_foundation.models.text_encoder import TextEncoder
from experiments_stage3_2_7.training.text_encoder_split import (
    SplitTextEmbedding,
    convert_text_encoder_to_split
)


def load_tokenizer(tokenizer_path: str) -> dict:
    """Load tokenizer from JSON file."""
    with open(tokenizer_path, 'r') as f:
        return json.load(f)


def test_token_id_mapping(tokenizer: dict):
    """Test 1: Verify token ID mappings are correct."""
    print("\n" + "="*80)
    print("TEST 1: Token ID Mapping Verification")
    print("="*80)

    vocab_size = tokenizer['vocab_size']
    word_to_idx = tokenizer['word_to_idx']

    print(f"\nVocabulary size: {vocab_size}")

    # Special tokens
    special_tokens = ['<PAD>', '<UNK>', '<EOS>']
    print(f"\nSpecial tokens:")
    for token in special_tokens:
        idx = word_to_idx[token]
        print(f"  {token}: {idx}")

    # Appearance tokens (should be 0-41)
    print(f"\nAppearance tokens (0-41):")
    appearance_keywords = ['red', 'blue', 'circle', 'square', 'small', 'large', 'left', 'right']
    for word in appearance_keywords:
        if word in word_to_idx:
            idx = word_to_idx[word]
            print(f"  {word}: {idx}")
            if idx >= 42:
                print(f"    [ERROR] Appearance token {word} has index {idx} >= 42")
                return False

    # Motion tokens (should be 42-81)
    print(f"\nMotion tokens (42-81):")
    motion_keywords = ['moving', 'stationary', 'up', 'down', 'leftward', 'rightward', 'slow', 'fast']
    for word in motion_keywords:
        if word in word_to_idx:
            idx = word_to_idx[word]
            print(f"  {word}: {idx}")
            if idx < 42 or idx >= 82:
                print(f"    [ERROR] ERROR: Motion token {word} has index {idx} (expected 42-81)")
                return False

    print("\n[PASS] Token ID mappings are correct")
    return True


def test_embedding_dimensions(split_emb: SplitTextEmbedding, d_model: int):
    """Test 2: Verify embedding dimensions."""
    print("\n" + "="*80)
    print("TEST 2: Embedding Dimension Verification")
    print("="*80)

    print(f"\nExpected d_model: {d_model}")
    print(f"Appearance embedding shape: {split_emb.appearance_emb.weight.shape}")
    print(f"Motion embedding shape: {split_emb.motion_emb.weight.shape}")

    if split_emb.appearance_emb.weight.shape != (42, d_model):
        print(f"[ERROR] ERROR: Appearance embedding shape mismatch")
        return False

    if split_emb.motion_emb.weight.shape != (40, d_model):
        print(f"[ERROR] ERROR: Motion embedding shape mismatch")
        return False

    print("\n[PASS] Embedding dimensions are correct")
    return True


def test_round_trip_conversion(text_encoder: TextEncoder):
    """Test 3: Round-trip conversion test."""
    print("\n" + "="*80)
    print("TEST 3: Round-Trip Conversion Test")
    print("="*80)

    # Save original embeddings
    original_emb = text_encoder.token_emb.weight.clone()
    print(f"Original embedding shape: {original_emb.shape}")

    # Convert to split
    print("\nConverting to split embeddings...")
    convert_text_encoder_to_split(text_encoder, appearance_vocab_size=42, motion_vocab_size=40)

    split_emb = text_encoder.token_emb
    print(f"Split appearance shape: {split_emb.appearance_emb.weight.shape}")
    print(f"Split motion shape: {split_emb.motion_emb.weight.shape}")

    # Merge back to unified
    print("\nMerging back to unified embeddings...")
    unified_emb = split_emb.merge_to_unified()
    print(f"Reconstructed embedding shape: {unified_emb.weight.shape}")

    # Compare
    print("\nComparing original vs reconstructed...")
    max_diff = torch.max(torch.abs(original_emb - unified_emb.weight)).item()
    mean_diff = torch.mean(torch.abs(original_emb - unified_emb.weight)).item()

    print(f"Max difference: {max_diff:.10f}")
    print(f"Mean difference: {mean_diff:.10f}")

    if max_diff > 1e-6:
        print(f"[ERROR] ERROR: Round-trip conversion failed (max diff {max_diff})")
        return False

    print("\n[PASS] Round-trip conversion successful")
    return True


def test_forward_pass(split_emb: SplitTextEmbedding):
    """Test 4: Forward pass with mixed tokens."""
    print("\n" + "="*80)
    print("TEST 4: Forward Pass with Mixed Tokens")
    print("="*80)

    # Create mixed token sequence
    token_ids = torch.tensor([
        [0, 1, 2, 3, 12, 33],      # Special + appearance
        [42, 43, 47, 48, 54, 55],  # Motion tokens
        [3, 12, 33, 42, 47, 54]    # Mixed
    ])

    print(f"\nInput token IDs:\n{token_ids}")

    # Forward pass
    embeddings = split_emb(token_ids)
    print(f"\nOutput embeddings shape: {embeddings.shape}")
    print(f"Expected shape: (3, 6, 256)")

    if embeddings.shape != (3, 6, 256):
        print(f"[ERROR] ERROR: Output shape mismatch")
        return False

    # Verify no NaN or Inf
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        print(f"[ERROR] ERROR: Output contains NaN or Inf")
        return False

    print("\n[PASS] Forward pass successful")
    return True


def test_gradient_flow(text_encoder: TextEncoder):
    """Test 5: Gradient flow through split embeddings."""
    print("\n" + "="*80)
    print("TEST 5: Gradient Flow Verification")
    print("="*80)

    # Convert to split if needed
    if not hasattr(text_encoder.token_emb, 'motion_emb'):
        convert_text_encoder_to_split(text_encoder, appearance_vocab_size=42, motion_vocab_size=40)

    split_emb = text_encoder.token_emb

    # Freeze appearance, train motion
    split_emb.freeze_appearance(freeze=True)
    split_emb.freeze_motion(freeze=False)

    print(f"\nAppearance trainable: {split_emb.appearance_emb.weight.requires_grad}")
    print(f"Motion trainable: {split_emb.motion_emb.weight.requires_grad}")

    # Create sample input
    token_ids = torch.tensor([[42, 43, 47, 48]])
    embeddings = split_emb(token_ids)

    # Create dummy loss
    loss = embeddings.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    appearance_grad = split_emb.appearance_emb.weight.grad
    motion_grad = split_emb.motion_emb.weight.grad

    print(f"\nAppearance gradient exists: {appearance_grad is not None}")
    print(f"Motion gradient exists: {motion_grad is not None}")

    if appearance_grad is not None and appearance_grad.abs().sum() > 1e-6:
        print(f"[ERROR] ERROR: Appearance embeddings have gradients (should be frozen)")
        return False

    if motion_grad is None or motion_grad.abs().sum() < 1e-6:
        print(f"[ERROR] ERROR: Motion embeddings have no gradients (should be trainable)")
        return False

    print(f"Motion gradient norm: {motion_grad.abs().sum().item():.6f}")

    print("\n[PASS] Gradient flow correct")
    return True


def test_checkpoint_save_load(text_encoder: TextEncoder, save_path: str):
    """Test 6: Checkpoint save/load."""
    print("\n" + "="*80)
    print("TEST 6: Checkpoint Save/Load")
    print("="*80)

    # Convert to split if needed
    if not hasattr(text_encoder.token_emb, 'motion_emb'):
        convert_text_encoder_to_split(text_encoder, appearance_vocab_size=42, motion_vocab_size=40)

    split_emb = text_encoder.token_emb

    # Save split embedding weights before conversion
    appearance_before = split_emb.appearance_emb.weight.clone()
    motion_before = split_emb.motion_emb.weight.clone()

    # Merge to unified for saving (standard checkpoint format)
    print("\nMerging to unified for checkpoint save...")
    unified_emb = split_emb.merge_to_unified()
    text_encoder.token_emb = unified_emb

    # Save checkpoint
    print(f"Saving checkpoint to {save_path}...")
    torch.save({
        'model_state_dict': text_encoder.state_dict(),
        'vocab_size': text_encoder.vocab_size,
        'd_model': text_encoder.d_model,
    }, save_path)

    # Load checkpoint
    print(f"Loading checkpoint from {save_path}...")
    checkpoint = torch.load(save_path, map_location='cpu')

    # Create new text encoder
    new_text_encoder = TextEncoder(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['d_model'],
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Load state
    new_text_encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Convert to split embeddings
    print("Converting to split embeddings after load...")
    convert_text_encoder_to_split(new_text_encoder, appearance_vocab_size=42, motion_vocab_size=40)

    new_split_emb = new_text_encoder.token_emb

    # Verify weights are preserved
    appearance_after = new_split_emb.appearance_emb.weight
    motion_after = new_split_emb.motion_emb.weight

    appearance_diff = torch.max(torch.abs(appearance_before - appearance_after)).item()
    motion_diff = torch.max(torch.abs(motion_before - motion_after)).item()

    print(f"Appearance embedding max diff: {appearance_diff:.10f}")
    print(f"Motion embedding max diff: {motion_diff:.10f}")

    if appearance_diff > 1e-6 or motion_diff > 1e-6:
        print(f"[ERROR] ERROR: Weights not preserved after save/load")
        return False

    # Clean up
    Path(save_path).unlink()

    print("\n[PASS] Checkpoint save/load successful (merge -> save -> load -> split)")
    return True


def main():
    print("="*80)
    print("STAGE 3.2.7C: SPLIT EMBEDDING VALIDATION")
    print("="*80)

    # Load tokenizer
    script_dir = Path(__file__).parent.parent
    tokenizer_path = script_dir / "checkpoints" / "tokenizer_motion_expanded.json"
    tokenizer = load_tokenizer(str(tokenizer_path))

    # Create text encoder
    text_encoder = TextEncoder(
        vocab_size=tokenizer['vocab_size'],
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Run tests
    results = []

    # Test 1: Token ID mapping
    results.append(("Token ID Mapping", test_token_id_mapping(tokenizer)))

    # Test 2: Embedding dimensions
    split_emb = SplitTextEmbedding(appearance_vocab_size=42, motion_vocab_size=40, d_model=256)
    results.append(("Embedding Dimensions", test_embedding_dimensions(split_emb, 256)))

    # Test 3: Round-trip conversion
    results.append(("Round-Trip Conversion", test_round_trip_conversion(text_encoder)))

    # Test 4: Forward pass
    results.append(("Forward Pass", test_forward_pass(split_emb)))

    # Test 5: Gradient flow
    results.append(("Gradient Flow", test_gradient_flow(text_encoder)))

    # Test 6: Checkpoint save/load
    checkpoint_path = script_dir / "checkpoints" / "test_split_checkpoint.pt"
    results.append(("Checkpoint Save/Load", test_checkpoint_save_load(text_encoder, str(checkpoint_path))))

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "[PASS] PASS" if passed else "[ERROR] FAIL"
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
