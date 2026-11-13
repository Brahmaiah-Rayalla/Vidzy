"""
Stage 3.2.6 Task 2: Investigate Stage 2 Text Encoder for Motion Conditioning Capability

Audits the tokenizer vocabulary and learned embeddings to determine whether
the text encoder can represent motion-related concepts.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def audit_tokenizer_vocabulary(tokenizer):
    """
    Audit the tokenizer vocabulary for motion-related tokens.
    """
    print_section("TOKENIZER VOCABULARY AUDIT")

    print(f"\nVocabulary size: {len(tokenizer.word_to_idx)}")
    print(f"Special tokens: {tokenizer.SPECIAL_TOKENS}")

    # Motion-related tokens to search for
    motion_keywords = [
        'moving', 'motion', 'movement',
        'slow', 'fast', 'quickly', 'slowly',
        'bounce', 'rotate', 'rotation',
        'diagonal', 'direction', 'trajectory',
        'velocity', 'speed', 'acceleration',
        'leftward', 'rightward', 'upward', 'downward',
        'oscillate', 'swing', 'slide',
        'spin', 'turn', 'twist'
    ]

    print("\n" + "-" * 80)
    print("MOTION KEYWORD SEARCH")
    print("-" * 80)

    found_motion = []
    for keyword in motion_keywords:
        if keyword in tokenizer.word_to_idx:
            idx = tokenizer.word_to_idx[keyword]
            found_motion.append((keyword, idx))
            print(f"  [FOUND] '{keyword}' -> idx {idx}")
        else:
            print(f"  [NOT FOUND] '{keyword}'")

    print(f"\nTotal motion keywords found: {len(found_motion)}/{len(motion_keywords)}")

    # Print all regular tokens (non-special)
    print("\n" + "-" * 80)
    print("ALL VOCABULARY TOKENS (excluding special)")
    print("-" * 80)

    regular_tokens = [(word, idx) for word, idx in tokenizer.word_to_idx.items()
                      if word not in tokenizer.SPECIAL_TOKENS]
    regular_tokens.sort(key=lambda x: x[1])

    for word, idx in regular_tokens:
        freq = tokenizer.word_freq.get(word, 0)
        print(f"  [{idx}] '{word}' (freq: {freq})")

    return found_motion


def test_motion_tokenization(tokenizer):
    """
    Test how motion-related phrases are tokenized.
    """
    print_section("MOTION PHRASE TOKENIZATION TEST")

    motion_phrases = [
        "moving left",
        "moving right",
        "moving slowly",
        "moving quickly",
        "bouncing up and down",
        "rotating clockwise",
        "diagonal movement",
        "fast motion",
        "slow motion",
        "stationary object",
        "static position"
    ]

    print("\nTokenization results:")
    for phrase in motion_phrases:
        indices, mask = tokenizer.encode(phrase, max_length=32)
        decoded = tokenizer.decode(indices)

        # Count unknown tokens
        unk_count = indices.count(tokenizer.SPECIAL_TOKENS['<UNK>'])

        print(f"\n  Phrase: '{phrase}'")
        print(f"  Token indices: {indices}")
        print(f"  <UNK> count: {unk_count}")
        print(f"  Decoded: '{decoded}'")

        if unk_count > 0:
            print(f"  [WARNING] {unk_count} unknown tokens")


def compare_embedding_similarity(text_encoder, tokenizer, phrases):
    """
    Compare embeddings for contrasting phrases to see if the encoder
    can distinguish different concepts.
    """
    print_section("EMBEDDING SIMILARITY ANALYSIS")

    # Static vs motion phrases
    test_cases = [
        ("static position", "moving left"),
        ("stationary", "moving quickly"),
        ("left", "right"),
        ("top", "bottom"),
        ("small", "large"),
        ("red circle", "blue circle"),
        ("moving slowly", "moving quickly"),
        ("moving left", "moving right"),
    ]

    text_encoder.eval()

    with torch.no_grad():
        for phrase1, phrase2 in test_cases:
            # Encode both phrases
            indices1, mask1 = tokenizer.encode(phrase1, max_length=32)
            indices2, mask2 = tokenizer.encode(phrase2, max_length=32)

            token_ids1 = torch.tensor([indices1])
            token_ids2 = torch.tensor([indices2])
            mask1_tensor = torch.tensor([mask1], dtype=torch.float)
            mask2_tensor = torch.tensor([mask2], dtype=torch.float)

            # Get embeddings
            emb1, _ = text_encoder(token_ids1, mask1_tensor)
            emb2, _ = text_encoder(token_ids2, mask2_tensor)

            # Use mean pooling over sequence
            emb1_pooled = emb1.mean(dim=1).squeeze().numpy()
            emb2_pooled = emb2.mean(dim=1).squeeze().numpy()

            # Compute cosine similarity
            similarity = 1 - cosine(emb1_pooled, emb2_pooled)

            print(f"\n  '{phrase1}' vs '{phrase2}'")
            print(f"  Cosine similarity: {similarity:.4f}")

            if similarity > 0.95:
                print(f"  [WARNING] Very high similarity (may not be distinguishable)")
            elif similarity < 0.5:
                print(f"  [OK] Good distinction")


def analyze_token_embeddings(text_encoder, tokenizer):
    """
    Analyze individual token embeddings to see if motion-related concepts
    have meaningful representations.
    """
    print_section("INDIVIDUAL TOKEN EMBEDDING ANALYSIS")

    # Get embedding layer
    token_emb = text_encoder.token_emb.weight.data  # [vocab_size, d_model]

    # Analyze specific tokens
    tokens_to_analyze = [
        'left', 'right', 'top', 'bottom', 'center',
        'small', 'medium', 'large',
        'moving', 'slow', 'fast'  # These may not exist
    ]

    print("\nToken embedding statistics:")
    for token in tokens_to_analyze:
        if token in tokenizer.word_to_idx:
            idx = tokenizer.word_to_idx[token]
            emb = token_emb[idx].numpy()

            print(f"\n  Token: '{token}' (idx {idx})")
            print(f"  Mean: {emb.mean():.6f}")
            print(f"  Std: {emb.std():.6f}")
            print(f"  Norm: {np.linalg.norm(emb):.6f}")
        else:
            print(f"\n  Token: '{token}' - NOT IN VOCABULARY")

    # Compute pairwise similarities between position tokens
    position_tokens = ['left', 'right', 'top', 'bottom', 'center']
    valid_positions = [(t, tokenizer.word_to_idx[t]) for t in position_tokens if t in tokenizer.word_to_idx]

    if len(valid_positions) >= 2:
        print("\n" + "-" * 80)
        print("POSITION TOKEN SIMILARITIES")
        print("-" * 80)

        for i, (token1, idx1) in enumerate(valid_positions):
            for j, (token2, idx2) in enumerate(valid_positions):
                if i < j:
                    emb1 = token_emb[idx1].numpy()
                    emb2 = token_emb[idx2].numpy()
                    sim = 1 - cosine(emb1, emb2)
                    print(f"  '{token1}' vs '{token2}': {sim:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3.2.6 Task 2: Audit Text Encoder Motion Conditioning")
    parser.add_argument("--stage2_checkpoint", type=str, default="../checkpoints_conditional/checkpoint_final.pt", help="Stage 2 checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="../data/text_image/tokenizer/tokenizer.json", help="Tokenizer path")
    args = parser.parse_args()

    print_section("STAGE 3.2.6 TASK 2: TEXT ENCODER MOTION CONDITIONING AUDIT")

    # Load tokenizer
    print(f"\n[1] Loading tokenizer from {args.tokenizer_path}")
    with open(args.tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)

    # Manually construct tokenizer to handle the specific format
    tokenizer = SimpleTokenizer(vocab_size=tokenizer_data['vocab_size'], min_freq=tokenizer_data['min_freq'])

    # Handle the mixed key format (some keys are strings, some are ints)
    word_to_idx = {}
    idx_to_word = {}
    for key, value in tokenizer_data['word_to_idx'].items():
        if key.startswith('<'):
            # Special token
            word_to_idx[key] = value
            idx_to_word[value] = key
        else:
            # Regular token - key might be string or int
            idx = int(key) if isinstance(key, str) and key.isdigit() else key
            word = value
            word_to_idx[word] = idx
            idx_to_word[idx] = word

    tokenizer.word_to_idx = word_to_idx
    tokenizer.idx_to_word = idx_to_word
    tokenizer.word_freq = tokenizer_data['word_freq']

    print(f"Tokenizer loaded: {len(tokenizer.word_to_idx)} tokens")

    # Audit vocabulary
    found_motion = audit_tokenizer_vocabulary(tokenizer)

    # Test motion phrase tokenization
    test_motion_tokenization(tokenizer)

    # Load text encoder
    print(f"\n[2] Loading text encoder from {args.stage2_checkpoint}")
    checkpoint = torch.load(args.stage2_checkpoint, map_location='cpu')

    vocab_size = checkpoint.get('text_encoder_state_dict', {}).get('token_emb.weight', torch.zeros(42, 256)).shape[0]
    print(f"Detected vocab size from checkpoint: {vocab_size}")

    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    if 'text_encoder_state_dict' in checkpoint:
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        print("Text encoder state loaded")
    else:
        print("WARNING: No text_encoder_state_dict in checkpoint")

    # Compare embedding similarities
    compare_embedding_similarity(text_encoder, tokenizer, [])

    # Analyze individual token embeddings
    analyze_token_embeddings(text_encoder, tokenizer)

    print_section("SUMMARY")

    print("\nKEY FINDINGS:")
    print(f"  1. Motion keywords in vocabulary: {len(found_motion)}")
    print(f"  2. Total vocabulary size: {len(tokenizer.word_to_idx)}")
    print(f"  3. Vocabulary built from: Static image captions only")

    if len(found_motion) == 0:
        print("\n  [CRITICAL] No motion-related vocabulary exists")
        print("  -> The text encoder cannot represent motion concepts")
        print("  -> Motion conditioning is impossible with current tokenizer")
        print("  -> RECOMMENDATION: Motion-specific vocabulary expansion required")
    else:
        print(f"\n  [OK] Found {len(found_motion)} motion-related tokens")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
