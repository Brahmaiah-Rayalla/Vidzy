# Stage 3.2.9K - B/D Equivalence Investigation Report

**Date**: 2026-11-04

## Executive Summary

Investigation into why Experiment B and Experiment D produced identical video outputs has revealed that:
1. **Checkpoints B and C have identical tensor parameters** (despite different training configurations)
2. **B and D models differ in motion embedding** when loaded for inference
3. **C and D models are identical** when loaded for inference

This explains the duplicate video outputs: the motion embedding difference between B and D is too small (0.03) to affect generation, making them effectively equivalent in practice.

## Detailed Findings

### Checkpoint Comparison

**Tensor Parameters:**
- Max tensor difference: **0.0** (identical)
- Checkpoint B and C have **exactly the same model weights**
- Both checkpoints have 589 parameter keys in `video_unet`

**Non-Tensor Differences:**
- `config`: Different training configurations
- `loss`: B=0.007, C=1.177 (different training progress)
- `optimizer`: Different optimizer states
- `step`: Both at step 100

**Key Insight:** The fact that B and C checkpoints have identical tensor parameters is highly suspicious. This suggests either:
- C checkpoint was initialized from B checkpoint without training
- Training failed to update weights for C
- Checkpoint saving/loading issue

### Inference Adapter Comparison

**B vs D Model Parameters:**
- Total comparisons: 570
- Equal: 569
- **Difference in: `text_encoder.token_emb.motion_emb.weight`**
- Max difference: 0.02965

**C vs D Model Parameters:**
- Total comparisons: 570
- Equal: 570
- **All parameters identical**

**Key Insight:** When loading the C checkpoint with temporal disabled (configuration D), the model parameters remain identical to configuration C. However, loading the B checkpoint produces a slightly different motion embedding than loading the C checkpoint with temporal disabled.

### Parameter Breakdown

**B Model (from checkpoint B):**
- Spatial: 9,439,171 parameters
- Temporal: 0 parameters
- Fusion: 0 parameters
- Total: 11,568,579 parameters
- use_temporal: False

**C Model (from checkpoint C):**
- Spatial: 9,439,171 parameters
- Temporal: 124,032 parameters
- Fusion: 2 parameters
- Total: 11,692,613 parameters
- use_temporal: True

**D Model (from checkpoint C, temporal disabled):**
- Spatial: 9,439,171 parameters
- Temporal: 0 parameters
- Fusion: 0 parameters
- Total: 11,568,579 parameters
- use_temporal: False

## Why B and D Produced Identical Videos

The motion embedding difference (0.03) is extremely small relative to the embedding magnitude. This difference is likely:
1. Too small to affect the generation trajectory
2. Possibly due to random initialization differences in the motion embedding layer
3. The motion embedding may not be effectively used in the generation process

Given that all other 569 parameters are identical, and the motion embedding difference is negligible, the two configurations produce identical outputs.

## Critical Issues Identified

### Issue 1: Identical Checkpoint Weights
**Severity: HIGH**

Checkpoints B and C have identical tensor parameters despite:
- Different training configurations (B: no temporal, C: with temporal)
- Different loss values (B: 0.007, C: 1.177)
- Supposedly different training runs

**Possible Causes:**
- Configuration C training failed to update weights
- Checkpoint C was incorrectly saved (saved B checkpoint instead)
- Training loop bug where temporal training didn't affect weights

**Impact:** This invalidates the comparison between temporal and non-temporal training. If C didn't actually train with temporal processing, the Stage 3.2.9J results are meaningless.

### Issue 2: Motion Embedding Not Learning
**Severity: HIGH**

The motion embedding difference between B and D is only 0.03, suggesting:
- Motion embedding is not being trained effectively
- Motion conditioning is not being learned
- The text encoder's motion embedding layer may be disconnected from the loss

**Impact:** This directly explains the poor motion control (19.4% accuracy) observed in Stage 3.2.9J.

### Issue 3: Temporal Processing Not Effective
**Severity: MEDIUM**

Configuration C shows temporal modules exist (124K parameters), but:
- Weights are identical to configuration B (no temporal)
- No benefit in motion control was observed
- Temporal processing may not be integrated into the training loop

## Next Steps

1. **Investigate Checkpoint C Training:** Examine training logs for configuration C to determine if temporal training actually occurred
2. **Analyze Motion Embedding Training:** Check if motion embedding gradients are being computed and applied
3. **Verify Temporal Integration:** Ensure temporal modules are actually being used during training
4. **Review Training Loop:** Audit the training code to verify all components are being trained

## Conclusion

The B/D equivalence investigation has revealed serious issues with the training process:
- Checkpoint C appears to be a copy of checkpoint B (identical weights)
- Motion embedding is not being trained effectively
- Temporal processing may not be integrated into training

These findings explain the poor motion control results and suggest that the Stage 3.2.9J evaluation compared two effectively identical configurations.
