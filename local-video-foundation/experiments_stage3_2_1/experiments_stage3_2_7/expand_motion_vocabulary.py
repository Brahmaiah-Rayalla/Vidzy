"""
Stage 3.2.7: Motion-Aware Vocabulary Expansion

Extends the existing tokenizer with motion vocabulary while preserving
all existing token IDs and embeddings.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer


MOTION_VOCABULARY = [
    # Basic motion states
    'moving', 'stationary', 'static',

    # Directions
    'left', 'right', 'up', 'down',
    'leftward', 'rightward', 'upward', 'downward',
    'horizontal', 'vertical', 'diagonal',

    # Speed
    'slow', 'fast', 'slowly', 'quickly',
    'accelerating', 'decelerating',

    # Motion types
    'bounce', 'bouncing', 'rotate', 'rotating', 'rotation',
    'oscillate', 'oscillating', 'swing', 'swinging',
    'slide', 'sliding', 'spin', 'spinning',

    # Additional descriptors
    'forward', 'backward',
    'motion', 'movement', 'velocity', 'speed',

    # Common words in motion phrases
    'object', 'clockwise', 'counterclockwise',
]


def expand_tokenizer(original_tokenizer_path, output_tokenizer_path):
    """
    Expand tokenizer with motion vocabulary while preserving existing tokens.
    """
    print("Loading original tokenizer...")
    with open(original_tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)

    # Parse the mixed format
    word_to_idx = {}
    idx_to_word = {}
    for key, value in tokenizer_data['word_to_idx'].items():
        if key.startswith('<'):
            word_to_idx[key] = value
            idx_to_word[value] = key
        elif str(key).isdigit():
            # Numeric key mapping to a word
            word_to_idx[value] = int(key)
            idx_to_word[int(key)] = value
        else:
            # Word key mapping to an index
            word_to_idx[key] = value
            idx_to_word[value] = key

    print(f"Original vocabulary size: {len(word_to_idx)}")

    # Add motion vocabulary
    # Convert keys to int for finding max
    int_keys = [int(k) for k in idx_to_word.keys() if isinstance(k, str) and k.isdigit()] + [k for k in idx_to_word.keys() if isinstance(k, int)]
    next_idx = max(int_keys) + 1
    added_tokens = []

    for word in MOTION_VOCABULARY:
        if word not in word_to_idx:
            word_to_idx[word] = next_idx
            idx_to_word[next_idx] = word
            added_tokens.append((word, next_idx))
            next_idx += 1
            print(f"  Added: '{word}' -> idx {next_idx - 1}")
        else:
            print(f"  Skipped (already exists): '{word}'")

    print(f"\nNew vocabulary size: {len(word_to_idx)}")
    print(f"Added {len(added_tokens)} motion tokens")

    # Save expanded tokenizer
    expanded_data = {
        'vocab_size': len(word_to_idx),
        'min_freq': tokenizer_data['min_freq'],
        'word_to_idx': word_to_idx,
        'idx_to_word': {str(k): v for k, v in idx_to_word.items()},
        'word_freq': tokenizer_data['word_freq']
    }

    with open(output_tokenizer_path, 'w') as f:
        json.dump(expanded_data, f, indent=2)

    print(f"\nExpanded tokenizer saved to: {output_tokenizer_path}")

    return word_to_idx, idx_to_word, added_tokens


def expand_text_encoder_embeddings(original_checkpoint_path, new_tokenizer_path, output_checkpoint_path):
    """
    Expand text encoder embeddings for new motion tokens.
    Initialize new embeddings from existing similar tokens or random.
    """
    print("\n" + "=" * 80)
    print("EXPANDING TEXT ENCODER EMBEDDINGS")
    print("=" * 80)

    # Load original checkpoint
    print(f"\nLoading original checkpoint: {original_checkpoint_path}")
    checkpoint = torch.load(original_checkpoint_path, map_location='cpu')

    # Load new tokenizer to get vocabulary size
    with open(new_tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)

    new_vocab_size = tokenizer_data['vocab_size']
    print(f"New vocabulary size: {new_vocab_size}")

    # Get original text encoder state
    if 'text_encoder_state_dict' in checkpoint:
        text_encoder_state = checkpoint['text_encoder_state_dict']
    else:
        raise ValueError("No text_encoder_state_dict in checkpoint")

    # Get original embedding matrix
    original_emb = text_encoder_state['token_emb.weight']
    original_vocab_size, d_model = original_emb.shape

    print(f"Original embedding shape: {original_emb.shape}")

    # Create new embedding matrix
    new_emb = torch.zeros(new_vocab_size, d_model)

    # Copy existing embeddings
    new_emb[:original_vocab_size] = original_emb

    # Initialize new embeddings
    # Strategy: Initialize from mean of existing embeddings + small random noise
    mean_emb = original_emb.mean(dim=0)
    std_emb = original_emb.std(dim=0)

    for new_idx in range(original_vocab_size, new_vocab_size):
        # Initialize with mean + small random perturbation
        new_emb[new_idx] = mean_emb + torch.randn(d_model) * std_emb * 0.1

    print(f"New embedding shape: {new_emb.shape}")
    print(f"Initialized {new_vocab_size - original_vocab_size} new embeddings")

    # Update checkpoint
    text_encoder_state['token_emb.weight'] = new_emb
    checkpoint['text_encoder_state_dict'] = text_encoder_state
    checkpoint['vocab_size'] = new_vocab_size

    # Save expanded checkpoint
    torch.save(checkpoint, output_checkpoint_path)
    print(f"\nExpanded checkpoint saved to: {output_checkpoint_path}")

    return new_emb


def test_motion_tokenization(new_tokenizer_path):
    """
    Test that motion phrases are correctly tokenized without UNK tokens.
    """
    print("\n" + "=" * 80)
    print("MOTION TOKENIZATION TEST")
    print("=" * 80)

    # Load tokenizer with proper parsing (same as expand_tokenizer)
    with open(new_tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)

    word_to_idx = {}
    for key, value in tokenizer_data['word_to_idx'].items():
        if key.startswith('<'):
            word_to_idx[key] = value
        elif str(key).isdigit():
            word_to_idx[value] = int(key)
        else:
            word_to_idx[key] = value

    # Test motion phrases
    test_phrases = [
        "moving left",
        "moving right",
        "moving slowly",
        "moving quickly",
        "stationary object",
        "bouncing up and down",
        "rotating clockwise",
        "diagonal movement",
        "accelerating forward",
    ]

    unk_token = word_to_idx.get('<UNK>', 1)

    print("\nTokenization results:")
    all_passed = True
    for phrase in test_phrases:
        tokens = phrase.lower().split()
        indices = []
        for token in tokens:
            if token in word_to_idx:
                indices.append(word_to_idx[token])
            else:
                indices.append(unk_token)
        unk_count = indices.count(unk_token)

        status = "[PASS]" if unk_count == 0 else "[FAIL]"
        print(f"  {status} '{phrase}': {unk_count} UNK tokens")

        if unk_count > 0:
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All motion phrases tokenized without UNK tokens")
    else:
        print("\n[FAILURE] Some motion phrases still have UNK tokens")

    return all_passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3.2.7: Motion Vocabulary Expansion")
    parser.add_argument("--original_tokenizer", type=str, default="../data/text_image/tokenizer/tokenizer.json", help="Original tokenizer path")
    parser.add_argument("--original_checkpoint", type=str, default="../checkpoints_conditional/checkpoint_final.pt", help="Original Stage 2 checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_tokenizer_path = os.path.join(args.output_dir, "tokenizer_motion_expanded.json")
    output_checkpoint_path = os.path.join(args.output_dir, "checkpoint_stage2_motion_expanded.pt")

    # Expand tokenizer
    word_to_idx, idx_to_word, added_tokens = expand_tokenizer(
        args.original_tokenizer,
        output_tokenizer_path
    )

    # Expand text encoder embeddings
    new_emb = expand_text_encoder_embeddings(
        args.original_checkpoint,
        output_tokenizer_path,
        output_checkpoint_path
    )

    # Test motion tokenization
    test_passed = test_motion_tokenization(output_tokenizer_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Added {len(added_tokens)} motion tokens")
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Tokenization test: {'PASSED' if test_passed else 'FAILED'}")

    if test_passed:
        print("\n[SUCCESS] Motion vocabulary expansion complete")
    else:
        print("\n[WARNING] Some issues detected in tokenization")


if __name__ == "__main__":
    main()
