# Stage 3.2.9L - Training Loop Repair Report

**Date:** 2026-11-05
**Status:** ✅ FIXED - Motion embeddings now receive gradients

## Problem Summary

Motion embeddings were not receiving gradients during training, even though they were marked as trainable and included in the optimizer. This prevented the model from learning motion conditioning.

## Root Cause Analysis

### Issue 1: Gradient Breaking in SplitEmbedding.forward()

**Location:** `experiments_stage3_2_7/video_foundation/models/text_encoder_split.py`, lines 73-88

**Problem:** The `SplitEmbedding.forward()` method used in-place assignment (`embeddings[mask] = values`) to combine original and motion embeddings. In-place assignment breaks the gradient computation graph, preventing gradients from flowing back to the motion embeddings.

**Original Code:**
```python
embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=device)
embeddings[original_mask] = original_embs[original_mask]  # In-place assignment
embeddings[motion_mask] = motion_embs[motion_mask]        # In-place assignment
```

**Fix:** Use `torch.where` for gradient-preserving conditional assignment:
```python
embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=device)
embeddings = torch.where(original_mask.unsqueeze(-1), original_embs, embeddings)
embeddings = torch.where(motion_mask.unsqueeze(-1), motion_embs, embeddings)
```

### Issue 2: Incorrect Vocab Size in load_from_standard_encoder()

**Location:** `experiments_stage3_2_7/video_foundation/models/text_encoder_split.py`, line 229

**Problem:** The `load_from_standard_encoder()` function used `standard_encoder.vocab_size` (82, after motion expansion) as the original vocab size, but it should use the original vocab size before expansion (42). This caused motion tokens to be incorrectly classified as original tokens.

**Original Code:**
```python
original_vocab_size = standard_encoder.vocab_size  # 82 (incorrect)
original_embeddings = standard_encoder.token_emb.weight.data.clone()
```

**Fix:** Add `original_vocab_size` parameter and extract only the original embeddings:
```python
if original_vocab_size is None:
    original_vocab_size = standard_encoder.vocab_size

if standard_encoder.vocab_size > original_vocab_size:
    original_embeddings = standard_encoder.token_emb.weight.data[:original_vocab_size].clone()
else:
    original_embeddings = standard_encoder.token_emb.weight.data.clone()
```

### Issue 3: Missing original_vocab_size in Training Script

**Location:** `stage3_2_9b_experiment.py`, line 200

**Problem:** The training script didn't pass the `original_vocab_size` parameter to `load_from_standard_encoder()`.

**Fix:** Add the parameter:
```python
split_encoder = load_from_standard_encoder(
    text_encoder,
    motion_vocab_size=40,
    freeze_transformer=True,
    original_vocab_size=42  # Original vocab size before motion expansion
)
```

## Verification

### Before Fix
- Motion embedding has gradient: **False**
- Temporal has gradient: True
- Fusion has gradient: True
- Spatial has gradient: False

### After Fix
- Motion embedding has gradient: **True** ✅
- Temporal has gradient: True ✅
- Fusion has gradient: True ✅
- Spatial has gradient: False ✅

## Impact

This fix enables:
1. Motion embeddings to learn from training data
2. Motion conditioning to actually affect the model's behavior
3. Effective training of Experiment B (motion embeddings only)
4. Effective training of Experiment C (motion + temporal + fusion)

## Files Modified

1. `experiments_stage3_2_7/video_foundation/models/text_encoder_split.py`
   - Fixed `SplitEmbedding.forward()` to use `torch.where`
   - Updated `load_from_standard_encoder()` to accept `original_vocab_size` parameter

2. `stage3_2_9b_experiment.py`
   - Added `original_vocab_size=42` to `load_from_standard_encoder()` call

3. `experiments_stage3_2_9B/phase2L_training_repair/training_loop_audit.py`
   - Added `original_vocab_size=42` to `load_from_standard_encoder()` call

## Next Steps

1. Run controlled pilot training with the fixed code
2. Verify that motion embeddings actually update during training
3. Evaluate motion conditioning quality after training
4. Confirm that the training pipeline is now functional

## Conclusion

The training loop repair is complete. The root cause was a combination of gradient-breaking in-place operations and incorrect vocab size handling. Both issues have been fixed, and gradient flow to motion embeddings is now confirmed.
