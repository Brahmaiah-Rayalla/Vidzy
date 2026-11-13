"""
Stage 3.2.9H - Text Encoding Validation Tests

Test A: Determinism
Test B: Prompt sensitivity
Test C: Token coverage
Test D: Seed independence

Author: Devin
Date: 2026-11-03
"""

import torch
import numpy as np
from pathlib import Path
import json
from corrected_inference_adapter import create_corrected_adapter


def test_determinism(adapter, prompt: str) -> dict:
    """Test A: Same prompt produces identical embeddings."""
    print(f"\nTest A - Determinism: '{prompt}'")
    
    adapter.load()
    
    # Encode prompt multiple times
    embeddings = []
    for i in range(3):
        text_emb, text_mask = adapter.encode_prompt(prompt)
        embeddings.append(text_emb.clone())
    
    # Check if all embeddings are identical
    is_deterministic = torch.allclose(embeddings[0], embeddings[1], atol=1e-6) and \
                      torch.allclose(embeddings[1], embeddings[2], atol=1e-6)
    
    print(f"  Deterministic: {is_deterministic}")
    
    return {
        'prompt': prompt,
        'deterministic': bool(is_deterministic),
        'embedding_shape': list(embeddings[0].shape),
        'embedding_mean': float(embeddings[0].mean()),
        'embedding_std': float(embeddings[0].std())
    }


def test_prompt_sensitivity(adapter, prompt1: str, prompt2: str) -> dict:
    """Test B: Different prompts produce different embeddings."""
    print(f"\nTest B - Prompt Sensitivity")
    print(f"  Prompt 1: '{prompt1}'")
    print(f"  Prompt 2: '{prompt2}'")
    
    adapter.load()
    
    # Encode both prompts
    text_emb1, mask1 = adapter.encode_prompt(prompt1)
    text_emb2, mask2 = adapter.encode_prompt(prompt2)
    
    # Compute differences
    diff_norm = torch.norm(text_emb1 - text_emb2).item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        text_emb1.flatten().unsqueeze(0),
        text_emb2.flatten().unsqueeze(0)
    ).item()
    mean_abs_diff = torch.abs(text_emb1 - text_emb2).mean().item()
    max_abs_diff = torch.abs(text_emb1 - text_emb2).max().item()
    
    # Get token IDs
    token_ids1, _ = adapter.tokenizer.encode(prompt1, max_length=64)
    token_ids2, _ = adapter.tokenizer.encode(prompt2, max_length=64)
    
    print(f"  Token IDs 1: {token_ids1[:10]}...")
    print(f"  Token IDs 2: {token_ids2[:10]}...")
    print(f"  Embedding Difference Norm: {diff_norm:.4f}")
    print(f"  Cosine Similarity: {cosine_sim:.4f}")
    print(f"  Mean Absolute Difference: {mean_abs_diff:.4f}")
    print(f"  Max Absolute Difference: {max_abs_diff:.4f}")
    
    return {
        'prompt1': prompt1,
        'prompt2': prompt2,
        'token_ids_1': token_ids1,
        'token_ids_2': token_ids2,
        'cosine_similarity': cosine_sim,
        'difference_norm': diff_norm,
        'mean_abs_diff': mean_abs_diff,
        'max_abs_diff': max_abs_diff
    }


def test_token_coverage(adapter, prompts: list) -> dict:
    """Test C: Motion tokens are correctly covered."""
    print(f"\nTest C - Token Coverage")
    
    adapter.load()
    
    results = {}
    motion_words = ['left', 'right', 'up', 'down', 'upward', 'downward', 'moving', 'slowly', 'quickly', 'stationary']
    
    for prompt in prompts:
        token_ids, mask = adapter.tokenizer.encode(prompt, max_length=64)
        tokens = [adapter.tokenizer.idx_to_word.get(idx, '<UNK>') for idx in token_ids]
        
        unk_count = sum(1 for idx in token_ids if idx == adapter.tokenizer.SPECIAL_TOKENS['<UNK>'])
        motion_tokens_present = [word for word in motion_words if word in tokens]
        
        results[prompt] = {
            'tokens': tokens,
            'token_ids': token_ids,
            'unk_count': unk_count,
            'motion_tokens_present': motion_tokens_present,
            'effective_length': sum(mask)
        }
        
        print(f"  '{prompt}': {len(motion_tokens_present)} motion tokens, {unk_count} UNK")
    
    return results


def test_seed_independence(adapter, prompt: str) -> dict:
    """Test D: Text encoding is independent of diffusion seed."""
    print(f"\nTest D - Seed Independence: '{prompt}'")
    
    adapter.load()
    
    # Encode with different random seeds
    embeddings = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        text_emb, text_mask = adapter.encode_prompt(prompt)
        embeddings.append(text_emb.clone())
    
    # Check if all embeddings are identical (should be true for deterministic text encoding)
    is_seed_independent = torch.allclose(embeddings[0], embeddings[1], atol=1e-6) and \
                          torch.allclose(embeddings[1], embeddings[2], atol=1e-6)
    
    print(f"  Seed Independent: {is_seed_independent}")
    
    return {
        'prompt': prompt,
        'seed_independent': bool(is_seed_independent)
    }


def main():
    """Run all text encoding validation tests."""
    
    print("="*80)
    print("STAGE 3.2.9H - TEXT ENCODING VALIDATION TESTS")
    print("="*80)
    
    base_dir = Path(__file__).parent.parent.parent
    checkpoint_path = str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt")
    
    adapter = create_corrected_adapter(checkpoint_path, 'C')
    
    # Test prompts
    prompts = [
        "A red object moving left",
        "A red object moving right",
        "A blue object moving upward",
        "A blue object moving downward",
        "A red object remaining stationary",
        "A red object moving slowly",
        "A red object moving quickly"
    ]
    
    results = {}
    
    # Test A: Determinism
    results['test_a_determinism'] = test_determinism(adapter, prompts[0])
    
    # Test B: Prompt sensitivity
    results['test_b_sensitivity'] = {}
    results['test_b_sensitivity']['left_vs_right'] = test_prompt_sensitivity(adapter, prompts[0], prompts[1])
    results['test_b_sensitivity']['up_vs_down'] = test_prompt_sensitivity(adapter, prompts[2], prompts[3])
    results['test_b_sensitivity']['left_vs_stationary'] = test_prompt_sensitivity(adapter, prompts[0], prompts[4])
    
    # Test C: Token coverage
    results['test_c_coverage'] = test_token_coverage(adapter, prompts)
    
    # Test D: Seed independence
    results['test_d_seed_independence'] = test_seed_independence(adapter, prompts[0])
    
    # Save results
    output_dir = Path(__file__).parent
    results_path = output_dir / "STAGE3_2_9H_TEXT_ENCODING_VALIDATION.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to {results_path}")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Determinism: {'PASS' if results['test_a_determinism']['deterministic'] else 'FAIL'}")
    print(f"  Seed Independence: {'PASS' if results['test_d_seed_independence']['seed_independent'] else 'FAIL'}")
    print(f"  Left vs Right Similarity: {results['test_b_sensitivity']['left_vs_right']['cosine_similarity']:.4f}")
    print(f"  Up vs Down Similarity: {results['test_b_sensitivity']['up_vs_down']['cosine_similarity']:.4f}")
    print(f"  Left vs Stationary Similarity: {results['test_b_sensitivity']['left_vs_stationary']['cosine_similarity']:.4f}")


if __name__ == "__main__":
    main()
