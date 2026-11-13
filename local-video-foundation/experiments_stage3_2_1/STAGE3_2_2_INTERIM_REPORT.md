# Stage 3.2.2: Interim Report - Critical Findings

## Executive Summary

Stage 3.2.2 has revealed **critical issues** with the current temporal model implementation. The trained temporal model is performing **worse** than both the frame-independent baseline and the zero-prediction baseline, indicating fundamental problems with the training objective or architecture.

## Critical Findings

### 1. Diffusion Objective Validation - FAILED

**Objective:** Verify that the model learns better than predicting zero noise.

**Results:**
- **Video Normalization:** FAIL
  - Mean: -0.0389 (expected: ~0.0)
  - Std: 0.3957 (expected: ~0.5 for range [-1, 1])
  - Issue: Videos are not properly normalized to [-1, 1] with zero mean

- **Noise Distribution:** PASS
  - Mean: 0.0005 (expected: ~0.0)
  - Std: 0.9997 (expected: ~1.0)
  - RMS: 0.9997
  - Noise addition is correct

- **Zero-Prediction Baseline:**
  - Mean: 0.9994
  - Std: 0.0030
  - Theoretical noise variance: 0.9994 ✓

- **Model Prediction:** FAIL
  - Mean: 1.2431
  - Std: 0.0216
  - **Improvement over zero baseline: -0.2437 (-24.4%)**
  - **Model is 24% WORSE than predicting zero noise**

**Conclusion:** The model is actively hurting performance compared to a zero-prediction baseline. This is a critical failure.

### 2. Baseline Comparison - FAILED

**Objective:** Compare frame-independent vs temporal models.

**Results:**
- **Frame-Independent (use_temporal=False):** 1.1763 +/- 0.0385
- **Untrained Temporal:** 1.4047 +/- 0.0532
- **Trained Temporal:** 1.2290 +/- 0.0336

**Improvements:**
- Untrained vs Frame-Independent: +0.2284 (19% worse)
- Trained vs Frame-Independent: +0.0527 (4.5% worse)
- Trained vs Untrained: -0.1756 (12.5% improvement over untrained)

**Conclusion:**
- Frame-independent baseline is the BEST performing model
- Adding temporal modules (even untrained) makes performance worse
- Training temporal modules provides some improvement over untrained, but still worse than frame-independent
- **Temporal modules are currently detrimental to performance**

### 3. Temporal Feature Contribution Diagnosis

**Objective:** Measure temporal feature contribution and gate values.

**Results:**

**Bottleneck Level:**
- Spatial RMS: 471.55 +/- 8.57
- Temporal RMS: 548.25 +/- 6.45
- Residual RMS: 217.75 +/- 0.45
- Fused RMS: 366.43 +/- 9.05
- Gate value: 0.502500 +/- 0.000000 (CONSTANT - NOT LEARNING)
- Temporal grad norm: 0.1331 +/- 0.0065
- Fusion grad norm: 0.1639 +/- 0.0095
- Temporal/Spatial ratio: 1.1627
- Residual/Spatial ratio: 0.4618

**Low-Res Level:**
- Spatial RMS: 1456.17 +/- 27.79
- Temporal RMS: 1988.63 +/- 19.37
- Residual RMS: 1217.10 +/- 5.78
- Fused RMS: 223.29 +/- 4.04
- Gate value: 0.502500 +/- 0.000000 (CONSTANT - NOT LEARNING)
- Temporal grad norm: 0.0868 +/- 0.0153
- Fusion grad norm: 0.0552 +/- 0.0041
- Temporal/Spatial ratio: 1.3657
- Residual/Spatial ratio: 0.8358

**Key Observations:**
1. **Gate values are constant at 0.5025** - they are not learning despite receiving gradients
2. Temporal features have higher RMS than spatial features (16-37% higher)
3. Residuals are substantial (46-84% of spatial RMS)
4. Gradients are active and non-zero
5. Fused features are lower RMS than raw temporal features

**Conclusion:** Temporal modules are generating features, but the learnable gates are not adapting (likely due to sigmoid activation constraining them or insufficient learning rate).

## Root Cause Analysis

### Primary Issues

1. **Gate Values Not Learning**
   - Gates initialized to 0.01 → sigmoid(0.01) ≈ 0.5025
   - Gates remain at exactly 0.5025 throughout training
   - Possible causes:
     - Sigmoid activation constraining gates to [0, 1]
     - Learning rate for fusion parameters too low (5e-6)
     - Gate gradients too small to overcome sigmoid derivative
     - Optimal gate value may actually be ~0.5 for this architecture

2. **Video Normalization Issue**
   - Video mean: -0.0389 (should be ~0.0)
   - Video std: 0.3957 (should be ~0.5)
   - This suggests images are not uniformly distributed in [-1, 1]
   - Gray background (128, 128, 128) normalized to (0, 0, 0) may be causing bias

3. **Temporal Modules Degrading Performance**
   - Adding temporal modules increases loss by 19% (untrained) or 4.5% (trained)
   - Temporal contribution may be adding noise rather than useful signal
   - Current fusion mechanism (projecting to noise space and adding) may be suboptimal

### Secondary Issues

4. **Training Loss Above 1.0**
   - Loss ~1.2 is higher than zero baseline ~1.0
   - This is unusual for diffusion models (should approach noise variance)
   - Suggests loss calculation or target may be incorrect

5. **Residual Magnitude**
   - Residuals are 46-84% of spatial RMS
   - This is quite large - temporal features are changing spatial features significantly
   - May be overwhelming the spatial signal

## Recommendations

### Immediate Actions Required

1. **Fix Video Normalization**
   - Ensure images are properly normalized to [-1, 1] with zero mean
   - Consider using white background or adjusting normalization for gray background
   - Verify normalization in dataset loading

2. **Fix Gate Learning**
   - Remove sigmoid activation from gates (use unconstrained gates)
   - Or use different initialization (try 0.0, 0.1, 1.0)
   - Increase fusion learning rate to match temporal learning rate
   - Consider using bias term in gate computation

3. **Verify Loss Calculation**
   - Double-check that loss is computed correctly
   - Verify that prediction target is the actual noise
   - Check tensor dimensions are correct
   - Compare with known working diffusion implementations

4. **Reconsider Temporal Fusion Architecture**
   - Current approach: project temporal features to noise space and add
   - Alternative: inject temporal features directly into spatial U-Net feature maps
   - Alternative: use skip connections for temporal features
   - Alternative: multi-scale temporal fusion at multiple resolutions

### Next Steps

Given the critical failures identified, **do not proceed with**:
- Expanding dataset to 256-512 videos
- Running additional training experiments
- Video generation evaluation
- CPU benchmarking

**Instead, focus on fixing the fundamental issues:**
1. Fix video normalization
2. Fix gate learning mechanism
3. Verify loss calculation
4. Re-evaluate temporal fusion architecture
5. Re-run validation to confirm fixes

Once these issues are resolved and the model outperforms the frame-independent baseline, then proceed with the remaining Stage 3.2.2 objectives.

## Acceptance Criteria Status

- [FAIL] Noise-prediction objective verified against zero-prediction baseline
- [PARTIAL] Both temporal blocks contribute measurable feature changes (yes, but detrimental)
- [FAIL] Trained temporal model outperforms frame-independent baseline
- [SKIPPED] Generated trajectories respond to movement prompts
- [SKIPPED] Stationary prompts produce stable outputs
- [SKIPPED] Results generalize to held-out trajectories
- [SKIPPED] CPU-only inference works
- [SKIPPED] Performance and memory measured
- [SKIPPED] All automated tests pass

## Conclusion

Stage 3.2.2 has identified critical issues that prevent meaningful temporal learning. The current implementation:

1. Performs worse than predicting zero noise
2. Performs worse than frame-independent baseline
3. Has non-learning gate values
4. Has video normalization issues

**Recommendation:** Pause Stage 3.2.2 and address these fundamental issues before proceeding. The temporal fusion architecture and training objective need to be re-examined and fixed.

## Files Generated

1. `validate_diffusion_objective.py` - Script to validate diffusion objective
2. `diffusion_objective_validation.json` - Validation results
3. `compare_baselines.py` - Script to compare model configurations
4. `baseline_comparison.json` - Baseline comparison results
5. `diagnose_temporal_contribution.py` - Script to diagnose temporal contribution
6. `temporal_contribution_diagnosis.json` - Temporal contribution statistics
7. `STAGE3_2_2_INTERIM_REPORT.md` - This report

## Data Changes

1. Modified `generate_controlled_dataset.py` to use gray background instead of white
2. Modified `synthetic_video_dataset.py` to add gray color
3. Regenerated controlled dataset with gray background
