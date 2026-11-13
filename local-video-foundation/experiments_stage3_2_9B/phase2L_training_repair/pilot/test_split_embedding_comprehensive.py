"""
Comprehensive tests for SplitEmbedding gradient-preserving fix.

Tests:
- Original appearance tokens
- Motion tokens
- Mixed appearance and motion tokens
- Padding tokens
- Multiple motion tokens in one sequence
- Gradient flow verification
- Comparison with original standard encoder
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import sys

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
stage3_2_7_dir = base_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation" / "models"))

from models.text_encoder_split import SplitEmbedding, TextEncoderSplit, load_from_standard_encoder
from models.text_encoder import TextEncoder

def test_split_embedding():
    """Comprehensive SplitEmbedding tests."""
    print("=" * 80)
    print("STAGE 3.2.9L.2 - SPLIT EMBEDDING COMPREHENSIVE TESTS")
    print("=" * 80)

    # Load checkpoint
    stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location='cpu')

    original_vocab_size = 42
    motion_vocab_size = 40
    d_model = 256

    # Create standard encoder for comparison
    text_encoder_standard = TextEncoder(
        vocab_size=82,  # Expanded vocab
        d_model=d_model,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64,
        dropout=0.0
    )
    text_encoder_standard.load_state_dict(checkpoint['text_encoder_state_dict'])

    # Create split embedding
    split_emb = SplitEmbedding(
        original_vocab_size=original_vocab_size,
        motion_vocab_size=motion_vocab_size,
        d_model=d_model,
        original_embeddings=checkpoint['text_encoder_state_dict']['token_emb.weight'][:original_vocab_size].data.clone()
    )

    # Ensure proper requires_grad
    split_emb.motion_emb.weight.requires_grad = True
    split_emb.original_emb.weight.requires_grad = False

    results = {
        'tests': {}
    }

    # Test 1: Original appearance tokens only
    print("\nTest 1: Original appearance tokens only")
    print("-" * 80)
    token_ids_1 = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40]])
    print(f"Token IDs: {token_ids_1.tolist()}")

    split_emb.zero_grad()
    embeddings_1 = split_emb(token_ids_1)
    loss_1 = embeddings_1.sum()
    loss_1.backward()

    motion_grad_1 = split_emb.motion_emb.weight.grad
    original_grad_1 = split_emb.original_emb.weight.grad

    results['tests']['test1_original_only'] = {
        'tokens': token_ids_1.tolist(),
        'motion_tokens_present': False,
        'output_requires_grad': embeddings_1.requires_grad,
        'motion_has_grad': motion_grad_1 is not None,
        'original_has_grad': original_grad_1 is not None,
        'output_shape': list(embeddings_1.shape)
    }

    print(f"Output requires_grad: {embeddings_1.requires_grad}")
    print(f"Motion embedding has gradient: {motion_grad_1 is not None}")
    print(f"Original embedding has gradient: {original_grad_1 is not None}")

    # Test 2: Motion tokens only
    print("\nTest 2: Motion tokens only")
    print("-" * 80)
    token_ids_2 = torch.tensor([[42, 43, 44, 45, 50, 55, 60, 65]])
    print(f"Token IDs: {token_ids_2.tolist()}")

    split_emb.zero_grad()
    embeddings_2 = split_emb(token_ids_2)
    loss_2 = embeddings_2.sum()
    loss_2.backward()

    motion_grad_2 = split_emb.motion_emb.weight.grad
    original_grad_2 = split_emb.original_emb.weight.grad

    results['tests']['test2_motion_only'] = {
        'tokens': token_ids_2.tolist(),
        'motion_tokens_present': True,
        'output_requires_grad': embeddings_2.requires_grad,
        'motion_has_grad': motion_grad_2 is not None,
        'original_has_grad': original_grad_2 is not None,
        'motion_grad_norm': motion_grad_2.norm().item() if motion_grad_2 is not None else None,
        'output_shape': list(embeddings_2.shape)
    }

    print(f"Output requires_grad: {embeddings_2.requires_grad}")
    print(f"Motion embedding has gradient: {motion_grad_2 is not None}")
    print(f"Motion gradient norm: {motion_grad_2.norm().item() if motion_grad_2 is not None else 'N/A'}")
    print(f"Original embedding has gradient: {original_grad_2 is not None}")

    # Test 3: Mixed appearance and motion tokens
    print("\nTest 3: Mixed appearance and motion tokens")
    print("-" * 80)
    token_ids_3 = torch.tensor([[5, 10, 42, 43, 15, 44, 20, 45, 25, 50]])
    print(f"Token IDs: {token_ids_3.tolist()}")

    split_emb.zero_grad()
    embeddings_3 = split_emb(token_ids_3)
    loss_3 = embeddings_3.sum()
    loss_3.backward()

    motion_grad_3 = split_emb.motion_emb.weight.grad
    original_grad_3 = split_emb.original_emb.weight.grad

    results['tests']['test3_mixed'] = {
        'tokens': token_ids_3.tolist(),
        'motion_tokens_present': True,
        'num_motion_tokens': (token_ids_3 >= original_vocab_size).sum().item(),
        'output_requires_grad': embeddings_3.requires_grad,
        'motion_has_grad': motion_grad_3 is not None,
        'original_has_grad': original_grad_3 is not None,
        'motion_grad_norm': motion_grad_3.norm().item() if motion_grad_3 is not None else None,
        'output_shape': list(embeddings_3.shape)
    }

    print(f"Output requires_grad: {embeddings_3.requires_grad}")
    print(f"Motion embedding has gradient: {motion_grad_3 is not None}")
    print(f"Motion gradient norm: {motion_grad_3.norm().item() if motion_grad_3 is not None else 'N/A'}")
    print(f"Original embedding has gradient: {original_grad_3 is not None}")

    # Test 4: Padding tokens (ID 0)
    print("\nTest 4: Padding tokens")
    print("-" * 80)
    token_ids_4 = torch.tensor([[0, 0, 5, 10, 42, 43, 0, 0]])
    print(f"Token IDs: {token_ids_4.tolist()}")

    split_emb.zero_grad()
    embeddings_4 = split_emb(token_ids_4)
    loss_4 = embeddings_4.sum()
    loss_4.backward()

    motion_grad_4 = split_emb.motion_emb.weight.grad
    original_grad_4 = split_emb.original_emb.weight.grad

    results['tests']['test4_padding'] = {
        'tokens': token_ids_4.tolist(),
        'motion_tokens_present': True,
        'num_padding_tokens': (token_ids_4 == 0).sum().item(),
        'output_requires_grad': embeddings_4.requires_grad,
        'motion_has_grad': motion_grad_4 is not None,
        'original_has_grad': original_grad_4 is not None,
        'motion_grad_norm': motion_grad_4.norm().item() if motion_grad_4 is not None else None,
        'output_shape': list(embeddings_4.shape)
    }

    print(f"Output requires_grad: {embeddings_4.requires_grad}")
    print(f"Motion embedding has gradient: {motion_grad_4 is not None}")
    print(f"Motion gradient norm: {motion_grad_4.norm().item() if motion_grad_4 is not None else 'N/A'}")

    # Test 5: Multiple motion tokens in one sequence
    print("\nTest 5: Multiple motion tokens in one sequence")
    print("-" * 80)
    token_ids_5 = torch.tensor([[42, 42, 43, 43, 44, 44, 45, 45]])
    print(f"Token IDs: {token_ids_5.tolist()}")

    split_emb.zero_grad()
    embeddings_5 = split_emb(token_ids_5)
    loss_5 = embeddings_5.sum()
    loss_5.backward()

    motion_grad_5 = split_emb.motion_emb.weight.grad
    original_grad_5 = split_emb.original_emb.weight.grad

    results['tests']['test5_repeated_motion'] = {
        'tokens': token_ids_5.tolist(),
        'motion_tokens_present': True,
        'unique_motion_tokens': len(set(token_ids_5[0].tolist())),
        'output_requires_grad': embeddings_5.requires_grad,
        'motion_has_grad': motion_grad_5 is not None,
        'original_has_grad': original_grad_5 is not None,
        'motion_grad_norm': motion_grad_5.norm().item() if motion_grad_5 is not None else None,
        'output_shape': list(embeddings_5.shape)
    }

    print(f"Output requires_grad: {embeddings_5.requires_grad}")
    print(f"Motion embedding has gradient: {motion_grad_5 is not None}")
    print(f"Motion gradient norm: {motion_grad_5.norm().item() if motion_grad_5 is not None else 'N/A'}")

    # Test 6: Comparison with original standard encoder
    print("\nTest 6: Comparison with original standard encoder")
    print("-" * 80)
    token_ids_6 = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40]])

    # Standard encoder output
    standard_embs = text_encoder_standard.token_emb(token_ids_6)

    # Split encoder output (original tokens only)
    split_embs = split_emb(token_ids_6)

    # Compare first 42 embeddings
    max_diff = (standard_embs - split_embs).abs().max().item()
    mean_diff = (standard_embs - split_embs).abs().mean().item()

    results['tests']['test6_standard_comparison'] = {
        'tokens': token_ids_6.tolist(),
        'max_absolute_difference': max_diff,
        'mean_absolute_difference': mean_diff,
        'equivalent_within_tolerance': max_diff < 1e-6
    }

    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    print(f"Equivalent within tolerance (1e-6): {max_diff < 1e-6}")

    # Test 7: Check for NaN or infinite gradients
    print("\nTest 7: Check for NaN or infinite gradients")
    print("-" * 80)
    token_ids_7 = torch.tensor([[5, 10, 42, 43, 15, 44, 20, 45, 25, 50]])

    split_emb.zero_grad()
    embeddings_7 = split_emb(token_ids_7)
    loss_7 = embeddings_7.sum()
    loss_7.backward()

    motion_grad_7 = split_emb.motion_emb.weight.grad

    has_nan = torch.isnan(motion_grad_7).any().item() if motion_grad_7 is not None else False
    has_inf = torch.isinf(motion_grad_7).any().item() if motion_grad_7 is not None else False

    results['tests']['test7_gradient_health'] = {
        'has_nan': has_nan,
        'has_inf': has_inf,
        'gradient_finite': not (has_nan or has_inf)
    }

    print(f"Gradient has NaN: {has_nan}")
    print(f"Gradient has Inf: {has_inf}")
    print(f"Gradient is finite: {not (has_nan or has_inf)}")

    # Test 8: Output ordering verification
    print("\nTest 8: Output ordering verification")
    print("-" * 80)
    token_ids_8 = torch.tensor([[5, 42, 10, 43, 15, 44, 20, 45]])

    split_emb.zero_grad()
    embeddings_8 = split_emb(token_ids_8)

    # Manually verify ordering by checking positions
    # Position 0 should be original token 5
    pos0_original = embeddings_8[0, 0]
    # Position 1 should be motion token 42
    pos1_motion = embeddings_8[0, 1]

    # Verify by direct lookup
    expected_pos0 = split_emb.original_emb(torch.tensor([5]))
    expected_pos1 = split_emb.motion_emb(torch.tensor([0]))  # 42 - 42 = 0

    pos0_match = torch.allclose(pos0_original, expected_pos0, atol=1e-6)
    pos1_match = torch.allclose(pos1_motion, expected_pos1, atol=1e-6)

    results['tests']['test8_ordering'] = {
        'tokens': token_ids_8.tolist(),
        'position_0_matches': pos0_match,
        'position_1_matches': pos1_match,
        'ordering_correct': pos0_match and pos1_match
    }

    print(f"Position 0 (original token 5) matches direct lookup: {pos0_match}")
    print(f"Position 1 (motion token 42) matches direct lookup: {pos1_match}")
    print(f"Output ordering correct: {pos0_match and pos1_match}")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = 0
    failed = 0

    for test_name, test_result in results['tests'].items():
        # Determine pass/fail based on test-specific criteria
        if 'original_only' in test_name:
            # Motion embeddings can have gradient due to torch.where evaluating both branches
            # This is acceptable - they just won't be used if no motion tokens
            status = test_result['original_has_grad'] == False
        elif 'motion_only' in test_name or 'mixed' in test_name or 'padding' in test_name or 'repeated_motion' in test_name:
            status = test_result['motion_has_grad'] == True and test_result['original_has_grad'] == False
        elif 'standard_comparison' in test_name:
            status = test_result['equivalent_within_tolerance']
        elif 'gradient_health' in test_name:
            status = test_result['gradient_finite']
        elif 'ordering' in test_name:
            status = test_result['ordering_correct']
        else:
            status = True

        if status:
            passed += 1
            print(f"[PASS] {test_name}")
        else:
            failed += 1
            print(f"[FAIL] {test_name}")

    print(f"\nTotal: {passed} passed, {failed} failed")

    results['summary'] = {
        'total_tests': len(results['tests']),
        'passed': passed,
        'failed': failed,
        'all_passed': failed == 0
    }

    # Save results
    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/STAGE3_2_9L2_SPLIT_EMBEDDING_TESTS.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results

if __name__ == "__main__":
    test_split_embedding()
