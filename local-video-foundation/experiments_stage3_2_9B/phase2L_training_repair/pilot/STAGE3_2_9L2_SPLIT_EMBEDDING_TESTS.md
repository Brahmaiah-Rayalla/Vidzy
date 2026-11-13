# Stage 3.2.9L.2 - SplitEmbedding Tests Report

**Date:** 2026-11-05
**Status:** ✅ PASS - All 8 tests passed

## Objective

Verify that the gradient-preserving fix in `SplitEmbedding.forward()` works correctly across various token patterns and preserves autograd connectivity.

## Test Suite

### Test 1: Original Appearance Tokens Only
**Tokens:** `[5, 10, 15, 20, 25, 30, 35, 40]` (all < 42)

**Results:**
- Output requires_grad: ✅ True
- Motion embedding has gradient: ✅ True (expected due to torch.where evaluating both branches)
- Original embedding has gradient: ✅ False (correctly frozen)

**Interpretation:** Motion embeddings can receive gradients even when not used, which is acceptable with torch.where. Original embeddings remain frozen as expected.

### Test 2: Motion Tokens Only
**Tokens:** `[42, 43, 44, 45, 50, 55, 60, 65]` (all ≥ 42)

**Results:**
- Output requires_grad: ✅ True
- Motion embedding has gradient: ✅ True
- Motion gradient norm: 45.25
- Original embedding has gradient: ✅ False

**Interpretation:** Motion embeddings correctly receive gradients when used exclusively.

### Test 3: Mixed Appearance and Motion Tokens
**Tokens:** `[5, 10, 42, 43, 15, 44, 20, 45, 25, 50]` (mixed)

**Results:**
- Output requires_grad: ✅ True
- Motion embedding has gradient: ✅ True
- Motion gradient norm: 35.78
- Original embedding has gradient: ✅ False
- Number of motion tokens: 5

**Interpretation:** Mixed token sequences work correctly, with gradients flowing to motion embeddings only.

### Test 4: Padding Tokens
**Tokens:** `[0, 0, 5, 10, 42, 43, 0, 0]` (includes padding)

**Results:**
- Output requires_grad: ✅ True
- Motion embedding has gradient: ✅ True
- Motion gradient norm: 22.63
- Number of padding tokens: 4

**Interpretation:** Padding tokens (ID 0) are handled correctly without breaking gradient flow.

### Test 5: Multiple Motion Tokens in One Sequence
**Tokens:** `[42, 42, 43, 43, 44, 44, 45, 45]` (repeated motion tokens)

**Results:**
- Output requires_grad: ✅ True
- Motion embedding has gradient: ✅ True
- Motion gradient norm: 64.0
- Unique motion tokens: 4

**Interpretation:** Repeated motion tokens accumulate gradients correctly (higher norm = stronger signal).

### Test 6: Comparison with Original Standard Encoder
**Tokens:** `[5, 10, 15, 20, 25, 30, 35, 40]` (original tokens only)

**Results:**
- Max absolute difference: 0.0
- Mean absolute difference: 0.0
- Equivalent within tolerance (1e-6): ✅ True

**Interpretation:** Original token embeddings are exactly equivalent to the standard encoder, confirming backward compatibility.

### Test 7: Gradient Health Check
**Tokens:** `[5, 10, 42, 43, 15, 44, 20, 45, 25, 50]` (mixed)

**Results:**
- Gradient has NaN: ✅ False
- Gradient has Inf: ✅ False
- Gradient is finite: ✅ True

**Interpretation:** No numerical instability in gradients.

### Test 8: Output Ordering Verification
**Tokens:** `[5, 42, 10, 43, 15, 44, 20, 45]` (alternating)

**Results:**
- Position 0 (original token 5) matches direct lookup: ✅ True
- Position 1 (motion token 42) matches direct lookup: ✅ True
- Output ordering correct: ✅ True

**Interpretation:** Embedding ordering matches input token sequence exactly.

## Summary

| Test | Status | Key Finding |
|------|--------|-------------|
| Original tokens only | ✅ PASS | Original embeddings frozen, motion embeddings can receive gradients |
| Motion tokens only | ✅ PASS | Motion embeddings receive gradients correctly |
| Mixed tokens | ✅ PASS | Gradients flow to motion embeddings only |
| Padding tokens | ✅ PASS | Padding handled without gradient issues |
| Repeated motion tokens | ✅ PASS | Gradient accumulation works correctly |
| Standard encoder comparison | ✅ PASS | Exact equivalence for original tokens |
| Gradient health | ✅ PASS | No NaN or Inf gradients |
| Output ordering | ✅ PASS | Embedding ordering matches token sequence |

**Total:** 8/8 tests passed

## Conclusion

The gradient-preserving fix using `torch.where` works correctly across all tested scenarios:
- Motion embeddings receive gradients when motion tokens are present
- Original embeddings remain frozen
- Output ordering is preserved
- No numerical instability
- Backward compatibility with standard encoder is maintained

The SplitEmbedding implementation is ready for training.
