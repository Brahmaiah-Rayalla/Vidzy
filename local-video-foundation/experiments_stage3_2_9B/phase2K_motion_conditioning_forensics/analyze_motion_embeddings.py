"""
Stage 3.2.9K - Motion Embedding Analysis

Analyzes motion tokenizer vocabulary, embeddings, and tokenization behavior.

Author: Devin
Date: 2026-11-04
"""

import torch
import json
from pathlib import Path
import sys
from collections import Counter
from typing import Tuple

# Add paths
base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "video_foundation"))

def analyze_tokenizer():
    """Analyze motion tokenizer vocabulary."""
    tokenizer_path = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"

    print("=" * 80)
    print("ANALYZING MOTION TOKENIZER")
    print("=" * 80)

    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)

    analysis = {
        'tokenizer_path': str(tokenizer_path),
        'vocabulary_size': len(tokenizer_data),
        'vocab_details': {},
        'motion_tokens': [],
        'non_motion_tokens': []
    }

    print(f"\nVocabulary size: {len(tokenizer_data)}")

    # Categorize tokens
    for token_id, token in tokenizer_data.items():
        token_str = str(token)
        if 'motion' in token_str.lower():
            analysis['motion_tokens'].append({'id': token_id, 'token': token_str})
        else:
            analysis['non_motion_tokens'].append({'id': token_id, 'token': token_str})

    print(f"Motion tokens: {len(analysis['motion_tokens'])}")
    print(f"Non-motion tokens: {len(analysis['non_motion_tokens'])}")

    # Show motion tokens
    print("\nMotion tokens:")
    for item in sorted(analysis['motion_tokens'], key=lambda x: str(x['id'])):
        print(f"  ID {item['id']}: {item['token']}")

    # Show some non-motion tokens
    print("\nSample non-motion tokens (first 10):")
    for item in sorted(analysis['non_motion_tokens'], key=lambda x: str(x['id']))[:10]:
        print(f"  ID {item['id']}: {item['token']}")

    analysis['vocab_details']['motion_count'] = len(analysis['motion_tokens'])
    analysis['vocab_details']['non_motion_count'] = len(analysis['non_motion_tokens'])

    return analysis

def analyze_text_encoder_embeddings():
    """Analyze text encoder motion embeddings."""
    checkpoint_path = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"

    print("\n" + "=" * 80)
    print("ANALYZING TEXT ENCODER EMBEDDINGS")
    print("=" * 80)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    analysis = {
        'checkpoint_path': str(checkpoint_path),
        'embedding_keys': [],
        'motion_embedding_analysis': {},
        'token_embedding_analysis': {}
    }

    # Find embedding layers
    for key in checkpoint.keys():
        if 'emb' in key.lower() or 'embedding' in key.lower():
            analysis['embedding_keys'].append(key)
            print(f"\nFound embedding: {key}")
            print(f"  Shape: {checkpoint[key].shape}")

            # Analyze motion embedding specifically
            if 'motion' in key.lower():
                analysis['motion_embedding_analysis'][key] = {
                    'shape': list(checkpoint[key].shape),
                    'mean': checkpoint[key].mean().item(),
                    'std': checkpoint[key].std().item(),
                    'min': checkpoint[key].min().item(),
                    'max': checkpoint[key].max().item(),
                    'norm': checkpoint[key].norm().item()
                }

                print(f"  Mean: {analysis['motion_embedding_analysis'][key]['mean']:.6f}")
                print(f"  Std: {analysis['motion_embedding_analysis'][key]['std']:.6f}")
                print(f"  Norm: {analysis['motion_embedding_analysis'][key]['norm']:.6f}")

            # Analyze token embedding
            if 'token' in key.lower():
                analysis['token_embedding_analysis'][key] = {
                    'shape': list(checkpoint[key].shape),
                    'mean': checkpoint[key].mean().item(),
                    'std': checkpoint[key].std().item(),
                    'min': checkpoint[key].min().item(),
                    'max': checkpoint[key].max().item(),
                    'norm': checkpoint[key].norm().item()
                }

                print(f"  Mean: {analysis['token_embedding_analysis'][key]['mean']:.6f}")
                print(f"  Std: {analysis['token_embedding_analysis'][key]['std']:.6f}")
                print(f"  Norm: {analysis['token_embedding_analysis'][key]['norm']:.6f}")

    return analysis

def test_tokenization():
    """Test motion tokenization for different motion descriptions."""
    print("\n" + "=" * 80)
    print("TESTING MOTION TOKENIZATION")
    print("=" * 80)

    from models.text_encoder import SimpleTokenizer

    # Load tokenizer
    tokenizer_path = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"

    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)

    # Create SimpleTokenizer from saved data
    tokenizer = SimpleTokenizer(
        vocab_size=tokenizer_data['vocab_size'],
        min_freq=tokenizer_data['min_freq']
    )
    tokenizer.word_to_idx = tokenizer_data['word_to_idx']
    tokenizer.idx_to_word = tokenizer_data['idx_to_word']
    tokenizer.word_freq = Counter(tokenizer_data['word_freq'])

    # Test prompts
    test_prompts = [
        "moving right",
        "moving left",
        "moving diagonal",
        "stationary",
        "pan right",
        "pan left",
        "zoom in",
        "zoom out"
    ]

    tokenization_results = {}

    for prompt in test_prompts:
        token_ids, attention_mask = tokenizer.encode(prompt)

        # Get actual tokens from IDs
        tokens = [tokenizer.idx_to_word.get(str(idx), '<UNK>') for idx in token_ids]

        # Check for motion-related tokens
        motion_tokens = [t for t in tokens if any(m in t.lower() for m in ['moving', 'stationary', 'right', 'left', 'up', 'down', 'diagonal'])]

        tokenization_results[prompt] = {
            'tokens': tokens,
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'unique_tokens': len(set(tokens)),
            'motion_tokens': motion_tokens
        }

        print(f"\nPrompt: '{prompt}'")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Unique tokens: {len(set(tokens))}")
        print(f"  Motion-related tokens: {motion_tokens}")

    return tokenization_results

def main():
    """Run motion embedding analysis."""
    print("=" * 80)
    print("STAGE 3.2.9K - MOTION EMBEDDING ANALYSIS")
    print("=" * 80)

    tokenizer_analysis = analyze_tokenizer()
    embedding_analysis = analyze_text_encoder_embeddings()
    tokenization_results = test_tokenization()

    results = {
        'tokenizer_analysis': tokenizer_analysis,
        'embedding_analysis': embedding_analysis,
        'tokenization_results': tokenization_results,
        'conclusion': {}
    }

    # Determine conclusions
    results['conclusion']['motion_token_count'] = len(tokenizer_analysis['motion_tokens'])
    results['conclusion']['has_motion_embedding'] = len(embedding_analysis['motion_embedding_analysis']) > 0
    results['conclusion']['motion_embedding_stats'] = embedding_analysis['motion_embedding_analysis'] if embedding_analysis['motion_embedding_analysis'] else None

    # Save results
    output_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2K_motion_conditioning_forensics")

    with open(output_dir / "STAGE3_2_9K_MOTION_EMBEDDING_ANALYSIS.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_dir}")

    return results

if __name__ == "__main__":
    main()
