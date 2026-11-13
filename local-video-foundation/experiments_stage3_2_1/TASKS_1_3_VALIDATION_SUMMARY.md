# Stage 3.2.3 Tasks 1-3 Validation Summary

**Date:** October 31, 2026
**Status:** Tasks 1-3 Complete

---

## Task 1: Verify Gate Parameterization ✅

### Findings

**Gate Architecture:**
- Type: `LearnableTemporalFusion` with sigmoid activation
- Raw parameter: Scalar (not channel-wise)
- Initial value: 0.01 (maps to 0.5025 via sigmoid)
- Activation: Sigmoid constrains output to [0, 1]

**Optimizer Inclusion:**
- Both gates included in optimizer: [PASS]
- Fusion parameters: 2 (bottleneck + low-res)
- Learning rate: 1e-4 (equal to temporal params)

**Gradient Flow:**
- Gradients non-zero: [PASS]
- Bottleneck gate gradient: ~0.18 average
- Low-res gate gradient: ~0.15 average

**Parameter Updates:**
- Numerically measurable: [PASS]
- Bottleneck gate change: -0.00049 over 5 steps
- Low-res gate change: -0.00049 over 5 steps
- Activated gate change: -0.00012 over 5 steps

**Effective Temporal Contribution:**
- Bottleneck temporal features: Mean ~0.77, Std ~1.27
- Low-res temporal features: Mean ~0.38, Std ~0.86
- Combined effective scaling: ~1.005 (0.5025 + 0.5025)

**Conclusion:** Gates are correctly parameterized, receiving gradients, and updating meaningfully.

---

## Task 2: Verify Video Normalization Pipeline ✅

### Findings

**Video Loading Pipeline:**
1. Load frames as PIL Images
2. Resize to (64, 64)
3. Convert to numpy array: `np.array(frame).astype(np.float32)`
4. Normalize: `/ 127.5 - 1.0` (maps [0, 255] → [-1, 1])
5. Stack frames: [num_frames, h, w, c]
6. Transpose: [c, num_frames, h, w]

**Video Statistics (10 samples):**
- Mean of means: 0.165784
- Mean of stds: 0.800192
- Overall min: -1.000000
- Overall max: 1.000000
- Mean range: 2.000000

**Normalization Verification:**
- All values in [-1, 1]: [PASS]
- Tensor dimensions correct: [PASS] ([batch, channels, frames, height, width])
- Values are float32: [PASS]
- Range matches expected: [PASS]

**Tensor Dimensions:**
- Shape: [1, 3, 16, 64, 64]
- Correct: [batch=1, channels=3, frames=16, height=64, width=64]

**RGB Channel Ordering:**
- Standard PIL Image to numpy conversion (RGB)
- Channel means vary by color content

**Frame Ordering:**
- 16 frames as expected
- Temporal dimension at index 2 (after channels)

**Conclusion:** Video normalization is correct and consistent with expected [-1, 1] range.

---

## Task 3: Validate Diffusion Objective ⚠️

### Findings

**Zero-Prediction Baseline:**
- Target noise mean: -0.000923
- Target noise std: 0.997432
- Zero-prediction MSE: 0.994867 (expected for unit-variance noise)

**Frame-Independent Model:**
- Predicted noise mean: -0.181410
- Predicted noise std: 0.281626
- Frame-independent MSE: 1.129481
- Improvement over zero: **-13.53%** (WORSE)
- Prediction-target correlation: -0.040453 (essentially uncorrelated)

**Temporal Model:**
- Predicted noise mean: -0.389170
- Predicted noise std: 0.493296
- Temporal MSE: 1.408291
- Improvement over zero: **-41.56%** (WORSE)
- Improvement over frame-independent: **-24.68%** (WORSE)
- Prediction-target correlation: -0.019663 (essentially uncorrelated)

**Loss Across Timestep Ranges:**
- Low (0-250): 1.500350
- Mid-low (250-500): 1.453109
- Mid-high (500-750): 1.490111
- High (750-1000): 1.518333

**Loss Discrepancy Investigation:**
- Previous reported losses: > 1.0
- Latest 10-step experiment loss: 0.0922
- Current measurements: ~1.4

**Possible explanations for discrepancy:**
1. Different evaluation conditions (batch size, dataset)
2. Different loss reduction (mean vs sum)
3. Different normalization
4. Different checkpoint (trained vs untrained)
5. Random variation in timestep sampling

**Critical Finding:** 
- **Both models perform WORSE than zero-prediction baseline**
- **Temporal model performs WORSE than frame-independent model**
- **Predictions are essentially uncorrelated with target noise**
- **This indicates a fundamental issue with the model or training**

**Conclusion:** The model is not learning to predict noise correctly. This is a critical issue that must be resolved before proceeding to extended training.

---

## Overall Assessment

### Passed Validations:
1. ✅ Gate parameterization is correct
2. ✅ Video normalization is correct

### Critical Issues:
1. ⚠️ Model performs worse than zero-prediction baseline
2. ⚠️ Temporal model performs worse than frame-independent model
3. ⚠️ Predictions are uncorrelated with target noise

### Next Steps:
- Task 4: Temporal ablation to isolate the issue
- Investigate why model predictions are uncorrelated with target noise
- Do NOT proceed to extended training until this is resolved
