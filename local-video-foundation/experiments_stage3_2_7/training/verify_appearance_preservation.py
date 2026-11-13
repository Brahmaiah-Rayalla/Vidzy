"""
Stage 3.2.7C: Appearance Knowledge Preservation Verification

Verify that appearance embeddings are preserved when:
1. Loading Stage 2.1 checkpoint
2. Converting to split embeddings
3. Comparing text encoder outputs before/after conversion
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments_stage3_2_7.video_foundation.models.text_encoder import TextEncoder
from experiments_stage3_2_7.training.text_encoder_split import convert_text_encoder_to_split


def load_tokenizer(tokenizer_path: str) -> dict:
    """Load tokenizer from JSON file."""
    with open(tokenizer_path, 'r') as f:
        return json.load(f)


def test_appearance_preservation(checkpoint_path: str, tokenizer_path: str):
    """Test that appearance knowledge is preserved."""
    print("="*80)
    print("APPEARANCE KNOWLEDGE PRESERVATION TEST")
    print("="*80)

    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Vocabulary size: {tokenizer['vocab_size']}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Get text encoder state dict
    if 'text_encoder_state_dict' in checkpoint:
        text_state = checkpoint['text_encoder_state_dict']
    else:
        text_state = {k.replace('text_encoder.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('text_encoder.')}

    # Infer configuration from state dict
    vocab_size = tokenizer['vocab_size']
    d_model = text_state['token_emb.weight'].shape[1]

    # Create text encoder with inferred configuration
    print("\nCreating text encoder from checkpoint...")
    print(f"  Vocab size: {vocab_size}")
    print(f"  d_model: {d_model}")

    text_encoder_original = TextEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )
    text_encoder_original.load_state_dict(text_state, strict=False)

    # Save original appearance embeddings
    original_emb = text_encoder_original.token_emb.weight.clone()
    print(f"Original embedding shape: {original_emb.shape}")

    # Create a copy for split conversion
    text_encoder_split = TextEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )
    text_encoder_split.load_state_dict(text_state, strict=False)

    # Create test token sequence (appearance-only)
    appearance_tokens = [3, 12, 33, 38, 15, 24]  # a, circle, red, blue, small, background
    token_ids = torch.tensor([appearance_tokens])
    attention_mask = torch.ones_like(token_ids, dtype=torch.float)

    print(f"\nTest token IDs (appearance only): {token_ids.tolist()}")

    # Get original output
    print("\nComputing original text encoder output...")
    text_encoder_original.eval()
    with torch.no_grad():
        original_output, _ = text_encoder_original(token_ids, attention_mask)

    print(f"Original output shape: {original_output.shape}")
    print(f"Original output mean: {original_output.mean().item():.6f}")
    print(f"Original output std: {original_output.std().item():.6f}")

    # Disable dropout for consistent comparison
    print("\nDisabling dropout for consistent comparison...")
    for module in text_encoder_original.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    # Convert copy to split embeddings
    print("\nConverting copy to split embeddings...")
    convert_text_encoder_to_split(text_encoder_split, appearance_vocab_size=42, motion_vocab_size=40)

    # Disable dropout on split as well
    for module in text_encoder_split.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    split_emb = text_encoder_split.token_emb
    print(f"Split appearance shape: {split_emb.appearance_emb.weight.shape}")
    print(f"Split motion shape: {split_emb.motion_emb.weight.shape}")

    # Verify appearance embeddings are preserved
    print("\nVerifying appearance embeddings...")
    appearance_split = split_emb.appearance_emb.weight
    appearance_diff = torch.max(torch.abs(original_emb[:42] - appearance_split)).item()
    appearance_mean_diff = torch.mean(torch.abs(original_emb[:42] - appearance_split)).item()

    print(f"Appearance embedding max diff: {appearance_diff:.10f}")
    print(f"Appearance embedding mean diff: {appearance_mean_diff:.10f}")

    if appearance_diff > 1e-6:
        print("[ERROR] Appearance embeddings changed after conversion!")
        return False

    print("[PASS] Appearance embeddings preserved")

    # Get split output
    print("\nComputing split text encoder output...")
    text_encoder_split.eval()
    with torch.no_grad():
        split_output, _ = text_encoder_split(token_ids, attention_mask)

    print(f"Split output shape: {split_output.shape}")
    print(f"Split output mean: {split_output.mean().item():.6f}")
    print(f"Split output std: {split_output.std().item():.6f}")

    # Compare outputs
    print("\nComparing original vs split outputs...")
    output_diff = torch.max(torch.abs(original_output - split_output)).item()
    output_mean_diff = torch.mean(torch.abs(original_output - split_output)).item()

    print(f"Output max diff: {output_diff:.10f}")
    print(f"Output mean diff: {output_mean_diff:.10f}")

    if output_diff > 1e-6:
        print("[ERROR] Text encoder outputs changed after conversion!")
        return False

    print("[PASS] Text encoder outputs preserved")

    # Test with mixed tokens
    print("\nTesting with mixed appearance/motion tokens...")
    mixed_tokens = [3, 12, 33, 42, 47, 54]  # a, circle, red, moving, leftward, slow
    token_ids_mixed = torch.tensor([mixed_tokens])

    with torch.no_grad():
        mixed_output, _ = text_encoder_split(token_ids_mixed, attention_mask)

    print(f"Mixed output shape: {mixed_output.shape}")
    print(f"Mixed output mean: {mixed_output.mean().item():.6f}")
    print(f"Mixed output std: {mixed_output.std().item():.6f}")

    if torch.isnan(mixed_output).any() or torch.isinf(mixed_output).any():
        print("[ERROR] Mixed output contains NaN or Inf!")
        return False

    print("[PASS] Mixed token processing works")

    return True


def main():
    script_dir = Path(__file__).parent.parent
    checkpoint_path = script_dir / "checkpoints" / "checkpoint_stage2_motion_expanded.pt"
    tokenizer_path = script_dir / "checkpoints" / "tokenizer_motion_expanded.json"

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return 1

    if not tokenizer_path.exists():
        print(f"[ERROR] Tokenizer not found: {tokenizer_path}")
        return 1

    success = test_appearance_preservation(str(checkpoint_path), str(tokenizer_path))

    print("\n" + "="*80)
    if success:
        print("[PASS] APPEARANCE KNOWLEDGE PRESERVATION TEST PASSED")
    else:
        print("[ERROR] APPEARANCE KNOWLEDGE PRESERVATION TEST FAILED")
    print("="*80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
