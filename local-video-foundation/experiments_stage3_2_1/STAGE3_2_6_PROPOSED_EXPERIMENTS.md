# Stage 3.2.6: Proposed Controlled Architecture Experiments

## Overview

Based on the root-cause analysis, this document proposes minimal, controlled experiments to validate fixes before proceeding with large-scale retraining.

## Root Causes Identified

1. **Text Encoder Cannot Represent Motion**: 0/26 motion keywords in vocabulary; all motion phrases map to `<UNK>` tokens
2. **Fusion Mechanism is Noise-Level**: Temporal features projected to noise space, bypassing spatial decoder
3. **Objective Does Not Encourage Temporal Learning**: Standard noise prediction MSE has no temporal consistency term
4. **Gates Learned to Suppress Temporal**: Both gates < 0.5, indicating temporal features were harmful

## Proposed Experiments

### Experiment A: Motion Vocabulary Expansion (Priority: CRITICAL)

**Objective**: Enable text encoder to represent motion concepts

**Changes**:
1. Expand tokenizer vocabulary with motion keywords:
   - moving, motion, movement
   - slow, fast, quickly, slowly
   - bounce, rotate, rotation
   - leftward, rightward, upward, downward
   - oscillate, swing, slide
2. Re-train text encoder embeddings for new tokens (or initialize from similar tokens)
3. Update video captions to include motion descriptions

**Expected Outcome**: Text encoder can distinguish between different motion types

**Validation**:
- Tokenize motion phrases → verify no `<UNK>` tokens
- Compare embeddings for "moving left" vs "moving right" → verify distinguishable (cosine similarity < 0.9)

**CPU Impact**: Minimal (vocabulary expansion, small embedding retraining)

---

### Experiment B: Feature-Level Fusion (Priority: CRITICAL)

**Objective**: Use temporal features at feature level instead of noise level

**Changes**:
1. Modify `_forward_with_temporal` to use `LearnableTemporalFusion` class
2. Inject temporal features back into spatial U-Net decoder
3. Remove projection to noise space before fusion

**Implementation**:
```python
# Current (broken):
bottleneck_noise = self.bottleneck_proj(bottleneck_temporal_flat)
temporal_noise = bottleneck_gate * bottleneck_noise_upsampled
predicted_noise_spatial = predicted_noise_spatial + temporal_noise

# Proposed (fixed):
bottleneck_fused = self.bottleneck_fusion(
    bottleneck_temporal,  # [batch, frames, channels, h, w]
    bottleneck_features   # [batch, frames, channels, h, w]
)
# Inject fused features back into spatial U-Net path
```

**Expected Outcome**: Temporal features benefit from spatial decoder processing

**Validation**:
- Compare gate values → should increase > 0.5 if temporal is useful
- Compare temporal vs frame-independent MSE → should improve

**CPU Impact**: Minimal (same computation, different fusion point)

---

### Experiment C: Temporal Consistency Loss (Priority: HIGH)

**Objective**: Add temporal consistency term to training objective

**Changes**:
1. Add frame-to-frame smoothness loss:
   ```python
   temporal_loss = F.mse_loss(
       predicted_noise[:, :, 1:, :, :, :],
       predicted_noise[:, :, :-1, :, :, :]
   )
   total_loss = noise_mse_loss + lambda_temporal * temporal_loss
   ```
2. Start with `lambda_temporal = 0.1`

**Expected Outcome**: Model incentivized to maintain temporal coherence

**Validation**:
- Monitor temporal_loss during training → should decrease
- Compare generated video smoothness → should improve

**CPU Impact**: Minimal (additional loss computation)

---

### Experiment D: Separate Motion Encoder (Priority: MEDIUM)

**Objective**: Decouple motion learning from noise prediction

**Changes**:
1. Create small motion encoder that predicts frame T+1 from frame T
2. Train motion encoder separately with motion prediction loss
3. Use motion encoder outputs to condition diffusion sampling

**Expected Outcome**: Motion patterns learned explicitly

**Validation**:
- Motion encoder prediction accuracy > frame-independent baseline
- Generated videos show coherent motion

**CPU Impact**: Moderate (additional model, but small)

---

### Experiment E: Gate Initialization Change (Priority: LOW)

**Objective**: Test if gate initialization affects learning

**Changes**:
1. Initialize gates at 0.1 instead of 0.01 (via sigmoid → ~0.525)
2. Allow gates to grow if temporal is useful

**Expected Outcome**: Gates may start higher, giving temporal features a chance

**Validation**:
- Monitor gate evolution during training
- Compare final gate values with baseline

**CPU Impact**: None (hyperparameter change only)

---

## Experiment Prioritization

### Phase 1: Critical Fixes (Run Sequentially)
1. **Experiment A**: Motion vocabulary expansion
2. **Experiment B**: Feature-level fusion

### Phase 2: Objective Enhancement
3. **Experiment C**: Temporal consistency loss

### Phase 3: Alternative Approaches (if Phase 1-2 fail)
4. **Experiment D**: Separate motion encoder
5. **Experiment E**: Gate initialization change

## Validation Criteria

For each experiment, success criteria:

- **Gate values > 0.5**: Temporal features are being used
- **Temporal MSE < Frame-independent MSE**: Temporal model outperforms baseline
- **Motion embeddings distinguishable**: Cosine similarity < 0.9 for different motions
- **Temporal loss decreases**: Objective optimization is working

## CPU Constraints

All proposed experiments respect CPU-first constraints:
- No CUDA required
- Minimal parameter additions
- Efficient 1D temporal convolutions retained
- No expensive temporal attention

## Next Steps

1. Implement Experiment A (motion vocabulary)
2. Implement Experiment B (feature-level fusion)
3. Run small-scale training (50-100 steps) to validate
4. If successful, proceed to Experiment C
5. Generate full training run with all successful changes
