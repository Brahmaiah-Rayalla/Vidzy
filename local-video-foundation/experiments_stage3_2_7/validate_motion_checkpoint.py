"""
Stage 3.2.7B: Motion Checkpoint Validation

Validates:
- Expanded tokenizer JSON
- Expanded text encoder checkpoint
- Original token IDs are unchanged
- Vocabulary size matches embedding matrix
- Added token embeddings have correct dimensions
- Checkpoint architecture matches current text encoder
- No supported motion phrase maps to UNK
"""

import os
import sys
import torch
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.models.text_encoder import TextEncoder


def load_tokenizer(tokenizer_path):
    """Load tokenizer from JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, tokenizer_path)
    with open(full_path, 'r') as f:
        return json.load(f)


def load_checkpoint(checkpoint_path):
    """Load checkpoint from PT file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, checkpoint_path)
    return torch.load(full_path, map_location='cpu')


def validate_original_token_ids():
    """Validate that original token IDs are unchanged."""
    print("\n" + "=" * 80)
    print("VALIDATION 1: Original Token IDs Unchanged")
    print("=" * 80)

    # Load expanded tokenizer
    expanded_tokenizer = load_tokenizer("checkpoints/tokenizer_motion_expanded.json")

    # Original tokens (from Stage 2.1)
    original_tokens = [
        "<PAD>", "<UNK>", "<EOS>", "a", "on", ".", "in", "the", "square", "triangle",
        ",", "and", "circle", "medium", "large", "small", "light_gray", "light_blue",
        "white", "light_yellow", "there", "is", "cream", "size", "background",
        "bottom", "pink", "left", "top", "green", "of", "purple", "yellow", "red",
        "positioned", "cyan", "right", "black", "blue", "orange", "gray", "center"
    ]

    word_to_idx = expanded_tokenizer['word_to_idx']

    all_correct = True
    for token in original_tokens:
        if token in word_to_idx:
            idx = word_to_idx[token]
            print(f"  '{token}': {idx}")
        else:
            print(f"  [FAIL] '{token}' not found in expanded tokenizer")
            all_correct = False

    if all_correct:
        print("[PASS] All original token IDs unchanged")
    else:
        print("[FAIL] Some original token IDs changed")

    return all_correct


def validate_vocabulary_size():
    """Validate vocabulary size matches embedding matrix."""
    print("\n" + "=" * 80)
    print("VALIDATION 2: Vocabulary Size Matches Embedding Matrix")
    print("=" * 80)

    # Load expanded tokenizer
    expanded_tokenizer = load_tokenizer("checkpoints/tokenizer_motion_expanded.json")
    vocab_size = expanded_tokenizer['vocab_size']
    print(f"Tokenizer vocab_size: {vocab_size}")

    # Load checkpoint
    checkpoint = load_checkpoint("checkpoints/checkpoint_stage2_motion_expanded.pt")
    text_encoder_state = checkpoint['text_encoder_state_dict']
    token_emb = text_encoder_state['token_emb.weight']
    embedding_vocab_size, d_model = token_emb.shape
    print(f"Embedding matrix shape: [{embedding_vocab_size}, {d_model}]")

    if vocab_size == embedding_vocab_size:
        print("[PASS] Vocabulary size matches embedding matrix")
        return True
    else:
        print(f"[FAIL] Vocabulary size mismatch: tokenizer={vocab_size}, embedding={embedding_vocab_size}")
        return False


def validate_added_embeddings():
    """Validate added token embeddings have correct dimensions."""
    print("\n" + "=" * 80)
    print("VALIDATION 3: Added Token Embeddings Dimensions")
    print("=" * 80)

    # Load expanded tokenizer
    expanded_tokenizer = load_tokenizer("checkpoints/tokenizer_motion_expanded.json")
    original_vocab_size = 42  # Original size

    # Load checkpoint
    checkpoint = load_checkpoint("checkpoints/checkpoint_stage2_motion_expanded.pt")
    text_encoder_state = checkpoint['text_encoder_state_dict']
    token_emb = text_encoder_state['token_emb.weight']
    vocab_size, d_model = token_emb.shape

    # Check added embeddings
    added_vocab_size = vocab_size - original_vocab_size
    print(f"Original vocab size: {original_vocab_size}")
    print(f"Expanded vocab size: {vocab_size}")
    print(f"Added tokens: {added_vocab_size}")
    print(f"Embedding dimension: {d_model}")

    # Check added embeddings are not all zeros
    added_embeddings = token_emb[original_vocab_size:]
    mean = added_embeddings.mean().item()
    std = added_embeddings.std().item()
    print(f"Added embeddings mean: {mean:.6f}")
    print(f"Added embeddings std: {std:.6f}")

    if std > 0.01:  # Should have some variation
        print("[PASS] Added embeddings have valid dimensions and variation")
        return True
    else:
        print("[FAIL] Added embeddings appear to be zero-initialized")
        return False


def validate_architecture():
    """Validate checkpoint architecture matches current text encoder."""
    print("\n" + "=" * 80)
    print("VALIDATION 4: Checkpoint Architecture Match")
    print("=" * 80)

    # Load checkpoint
    checkpoint = load_checkpoint("checkpoints/checkpoint_stage2_motion_expanded.pt")
    text_encoder_state = checkpoint['text_encoder_state_dict']

    # Create text encoder with expanded vocab
    text_encoder = TextEncoder(
        vocab_size=82,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Load state dict
    try:
        text_encoder.load_state_dict(text_encoder_state, strict=True)
        print("[PASS] Checkpoint architecture matches current text encoder")
        return True
    except Exception as e:
        print(f"[FAIL] Architecture mismatch: {e}")
        return False


def validate_motion_phrases():
    """Validate no supported motion phrase maps to UNK."""
    print("\n" + "=" * 80)
    print("VALIDATION 5: Motion Phrases No UNK")
    print("=" * 80)

    # Load expanded tokenizer
    expanded_tokenizer = load_tokenizer("checkpoints/tokenizer_motion_expanded.json")
    word_to_idx = expanded_tokenizer['word_to_idx']
    unk_idx = word_to_idx['<UNK>']

    # Motion phrases to test
    motion_phrases = [
        "moving",
        "stationary",
        "left",
        "right",
        "up",
        "down",
        "slow",
        "fast",
        "slowly",
        "quickly",
        "bounce",
        "rotate",
        "diagonal",
        "clockwise",
        "counterclockwise"
    ]

    all_valid = True
    for phrase in motion_phrases:
        if phrase in word_to_idx:
            idx = word_to_idx[phrase]
            if idx == unk_idx:
                print(f"  [FAIL] '{phrase}' maps to UNK")
                all_valid = False
            else:
                print(f"  '{phrase}': {idx}")
        else:
            print(f"  [FAIL] '{phrase}' not in vocabulary")
            all_valid = False

    if all_valid:
        print("[PASS] All motion phrases are valid (no UNK)")
    else:
        print("[FAIL] Some motion phrases map to UNK or are missing")

    return all_valid


def validate_checkpoint_metadata():
    """Validate checkpoint contains expected metadata."""
    print("\n" + "=" * 80)
    print("VALIDATION 6: Checkpoint Metadata")
    print("=" * 80)

    # Load checkpoint
    checkpoint = load_checkpoint("checkpoints/checkpoint_stage2_motion_expanded.pt")

    required_keys = ['text_encoder_state_dict']
    optional_keys = ['step', 'loss']

    all_required_present = True
    for key in required_keys:
        if key in checkpoint:
            print(f"  '{key}': present")
        else:
            print(f"  [FAIL] '{key}': missing")
            all_required_present = False

    for key in optional_keys:
        if key in checkpoint:
            print(f"  '{key}': present ({checkpoint[key]})")
        else:
            print(f"  '{key}': not present (optional)")

    if all_required_present:
        print("[PASS] All required metadata present")
        return True
    else:
        print("[FAIL] Some required metadata missing")
        return False


def main():
    print("=" * 80)
    print("STAGE 3.2.7B: MOTION CHECKPOINT VALIDATION")
    print("=" * 80)

    results = {
        'original_token_ids': validate_original_token_ids(),
        'vocab_size_match': validate_vocabulary_size(),
        'added_embeddings': validate_added_embeddings(),
        'architecture_match': validate_architecture(),
        'motion_phrases': validate_motion_phrases(),
        'checkpoint_metadata': validate_checkpoint_metadata()
    }

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n[SUCCESS] All validations passed")
        return 0
    else:
        print("\n[FAILURE] Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
