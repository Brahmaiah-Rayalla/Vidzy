# Stage 3.2.9C - Loss Analysis Report

**Date:** 2026-11-02
**Purpose:** Investigate loss discrepancy between Experiment B and Experiment C

## Executive Summary

Experiment B and Experiment C exhibit **dramatically different loss scales** (133x difference), indicating that the loss function or prediction target differs between the two experiments. This makes direct loss comparison invalid. The loss scale difference is not due to parameter count alone and requires investigation of the training implementation.

## Loss Statistics

### Experiment B (Motion Embeddings Only)
- **Initial loss:** 0.009629
- **Final loss:** 0.006957
- **Loss range:** 0.005096 - 0.219391
- **Mean loss:** 0.019940
- **Std loss:** 0.032128
- **Loss reduction:** 0.002672 (27.74%)
- **Coefficient of variation:** 1.6112

### Experiment C (Motion + Temporal Learning)
- **Initial loss:** 1.286032
- **Final loss:** 1.177136
- **Loss range:** 0.922432 - 1.666774
- **Mean loss:** 1.221151
- **Std loss:** 0.114933
- **Loss reduction:** 0.108896 (8.47%)
- **Coefficient of variation:** 0.0941

## Loss Scale Discrepancy

### Magnitude of Difference
- **Initial loss ratio (C/B):** 133.56x
- **Final loss ratio (C/B):** 169.19x
- **Mean loss ratio (C/B):** 61.23x

### Parameter Count Comparison
- **Experiment B parameters:** 10,240
- **Experiment C parameters:** 134,274
- **Parameter ratio (C/B):** 13.11x

### Analysis
The loss scale difference (133x) is **10x larger** than the parameter ratio (13x). This indicates the loss scale difference is **not explained by parameter count alone**.

## Potential Causes of Loss Scale Difference

### 1. Different Loss Functions
Experiment C may use a different loss function that includes:
- Temporal consistency loss
- Fusion gate regularization
- Additional penalty terms

### 2. Different Prediction Targets
The models may predict different quantities:
- Experiment B: Predicts noise for single frames
- Experiment C: Predicts noise for temporal sequences with different normalization

### 3. Different Input Scales
- Experiment B: Uses motion tokens with specific scaling
- Experiment C: Uses motion tokens + temporal features with different scaling

### 4. Different Loss Reduction Methods
- Experiment B: May use mean reduction
- Experiment C: May use sum reduction or different weighting

## Loss Stability Analysis

### Experiment B
- **Coefficient of variation:** 1.6112
- **Interpretation:** High volatility, frequent spikes
- **Cause:** Likely due to sparse motion token gradients

### Experiment C
- **Coefficient of variation:** 0.0941
- **Interpretation:** Much more stable
- **Cause:** More parameters provide smoother gradients

### Comparison
Experiment C's loss is **17x more stable** than Experiment B (1.6112 / 0.0941 = 17.1). This suggests:
- Temporal processing provides smoother optimization landscape
- Additional parameters act as regularization
- Fusion gates contribute to stability

## Loss Reduction Comparison

### Experiment B
- **Absolute reduction:** 0.002672
- **Relative reduction:** 27.74%
- **Rate:** Good convergence relative to scale

### Experiment C
- **Absolute reduction:** 0.108896
- **Relative reduction:** 8.47%
- **Rate:** Slower convergence relative to scale

### Analysis
Experiment B achieved better relative loss reduction (27.74% vs 8.47%), but this is **not comparable** due to different loss scales. The absolute reduction in Experiment C (0.108896) is much larger, but this is meaningless without knowing the loss scale.

## Implementation Investigation

### Training Script Analysis
The Phase 2C training script (`phase2c_pilot_training.py`) was examined:
- **Loss function:** Not explicitly defined in the script
- **Likely source:** Loss is computed within the model's forward pass
- **Missing information:** Cannot determine loss definition from script alone

### Required Investigation
To fully understand the loss discrepancy, we need to examine:
1. The `ConditionalUNet` forward pass implementation
2. How loss is computed (MSE, L1, custom)
3. Whether temporal fusion adds additional loss terms
4. Input normalization for both experiments
5. Noise prediction target (epsilon, v-prediction, etc.)

## Conclusion

### Key Findings
1. **Loss scales are incompatible:** Experiment C losses are 133x larger than Experiment B
2. **Not explained by parameters:** Parameter ratio (13x) << loss ratio (133x)
3. **Different stability:** Experiment C is 17x more stable
4. **Implementation unclear:** Loss function not visible in training script

### Implications
1. **Direct loss comparison is invalid:** Cannot conclude which model trained better based on loss alone
2. **Generative evaluation required:** Must compare actual video outputs
3. **Loss investigation incomplete:** Need to examine model implementation to determine root cause

### Recommendations
1. **Do not use loss for comparison:** Loss values are not comparable between experiments
2. **Focus on generative quality:** Evaluate actual video outputs
3. **Investigate model implementation:** Examine ConditionalUNet to understand loss computation
4. **Document loss scaling:** Ensure future experiments use comparable loss scales

### Status
**Investigation Status:** ⚠️ INCOMPLETE
**Reason:** Loss function implementation not accessible from training script alone
**Required:** Model implementation inspection or direct loss measurement with identical inputs

---

**Analysis Status:** ✅ STATISTICS COMPLETE
**Root Cause:** ⚠️ REQUIRES MODEL IMPLEMENTATION INSPECTION
