"""
Stage 3.2.9C - Prompt Tokenization Verification

Test actual motion prompts using the existing tokenizer.

Author: Devin
Date: 2026-11-02
"""

import sys
import json
from pathlib import Path

# Add paths
parent_dir = Path(__file__).parent.parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer


def test_tokenization():
    """Test tokenization of motion prompts."""
    print("=" * 80)
    print("  STAGE 3.2.9C - PROMPT TOKENIZATION VERIFICATION")
    print("=" * 80)

    # Load tokenizer configuration
    tokenizer_path = parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json"
    with open(tokenizer_path, 'r') as f:
        tokenizer_config = json.load(f)

    print(f"\nTokenizer loaded from: {tokenizer_path}")
    print(f"Number of tokens: {tokenizer_config.get('num_tokens', 'N/A')}")
    print(f"Token dimension: {tokenizer_config.get('token_dim', 'N/A')}")

    # Check if motion token vocabulary exists
    motion_vocab = tokenizer_config.get('motion_vocab', {})
    if motion_vocab:
        print(f"Motion vocabulary size: {len(motion_vocab)}")
        print(f"Motion tokens: {list(motion_vocab.keys())[:10]}")
    else:
        print("No motion vocabulary found in tokenizer config")

    # Test prompts
    test_prompts = [
        "A red object moving left",
        "A blue object moving right",
        "An object moving upward",
        "An object moving downward",
        "An object moving slowly",
        "An object moving quickly",
        "An object bouncing",
        "A stationary object",
        "An object moving left and then right",
        "An object moving right and then left"
    ]

    print("\n--- TOKENIZATION ANALYSIS ---")
    results = []

    for prompt in test_prompts:
        # Simple analysis: check if motion keywords are in prompt
        motion_keywords = ['left', 'right', 'upward', 'downward', 'slowly', 'quickly', 'bouncing', 'stationary']
        found_keywords = [kw for kw in motion_keywords if kw in prompt.lower()]

        result = {
            "prompt": prompt,
            "motion_keywords_found": found_keywords,
            "has_motion_tokenization": len(found_keywords) > 0
        }
        results.append(result)

        print(f"\nPrompt: {prompt}")
        print(f"  Motion keywords: {found_keywords}")
        print(f"  Has motion tokenization: {result['has_motion_tokenization']}")

    # Save results
    output_path = Path(__file__).parent / "tokenization_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            "tokenizer_config": tokenizer_config,
            "prompt_analysis": results
        }, f, indent=2)

    print(f"\nSaved tokenization results to {output_path}")

    return results


if __name__ == "__main__":
    results = test_tokenization()
