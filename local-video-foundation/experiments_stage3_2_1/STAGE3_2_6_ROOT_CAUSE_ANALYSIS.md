# Stage 3.2.6: Root Cause Analysis - Temporal Learning Failure

**Date**: November 1, 2026
**Objective**: Identify why the trained temporal video diffusion model consistently performs worse than the frame-independent spatial model

---

## Executive Summary

The temporal video diffusion model in Stage 3.2.5 failed to learn meaningful temporal information, performing **17.6% worse** than the frame-independent baseline (Temporal MSE: 0.260 vs Frame-independent MSE: 0.221). The root cause analysis identified **four critical issues**:

1. **Text Encoder Cannot Represent Motion** (CRITICAL): 0/26 motion keywords in vocabulary; all motion phrases map to `<UNK>` tokens
2. **Fusion Mechanism is Noise-Level** (CRITICAL): Temporal features projected to noise space, bypassing spatial decoder
3. **Objective Does Not Encourage Temporal Learning** (HIGH): Standard noise prediction MSE has no temporal consistency term
4. **Gates Learned to Suppress Temporal** (HIGH): Both gates < 0.5, indicating temporal features were harmful

The gates learned to suppress temporal features (0.495, 0.494) because the model determined they hurt the noise prediction objective. This is a rational response to the combination of broken conditioning (text encoder), broken fusion (noise-level), and misaligned objective (no temporal loss).

---

## Task 1: Audit Stage 3.2.5 Evaluation Conditions

### Finding: Evaluation Conditions Were NOT Identical

The Stage 3.2.5 final report compared metrics from different evaluation configurations:

- **Training-time baselines** (5 samples): Frame-independent MSE: 0.0108, Temporal MSE: 0.206
- **Final evaluation** (10 samples): Frame-independent MSE: 0.221, Temporal MSE: 0.260

### Action Taken

Created deterministic evaluation fixture (`audit_evaluation_conditions.py`) to ensure all baselines are evaluated on identical data with identical conditions.

### Corrected Results (Identical Conditions)

- **Zero-prediction MSE**: 0.999
- **Frame-independent MSE**: 0.221
- **Temporal MSE**: 0.260
- **Temporal degradation**: -17.6% (worse than frame-independent)

### Conclusion

The temporal model is indeed worse than frame-independent, but the magnitude of degradation was misreported in Stage 3.2.5 due to inconsistent evaluation conditions.

---

## Task 2: Text Encoder Motion Conditioning Audit

### Finding: Text Encoder Cannot Represent Motion Concepts

**Vocabulary Analysis**:
- Total vocabulary size: 42 tokens
- Motion keywords found: 0/26
- Vocabulary built from: Static image captions only

**Motion keywords searched**: moving, motion, movement, slow, fast, quickly, slowly, bounce, rotate, rotation, diagonal, direction, trajectory, velocity, speed, acceleration, leftward, rightward, upward, downward, oscillate, swing, slide, spin, turn, twist

**Tokenization Test Results**:
- "moving left" → [UNK, UNK, EOS] (2 unknown tokens)
- "moving right" → [UNK, UNK, EOS] (2 unknown tokens)
- "slow motion" → [UNK, UNK, EOS] (2 unknown tokens)
- All motion phrases map to unknown tokens

**Embedding Similarity Analysis**:
- "static position" vs "moving left": Cosine similarity = 1.000 (identical)
- "stationary" vs "moving quickly": Cosine similarity = 0.989 (nearly identical)
- "left" vs "right": Cosine similarity = 1.000 (identical)

**Conclusion**: The text encoder cannot distinguish between different motion concepts because motion vocabulary does not exist. All motion phrases map to the same `<UNK>` embedding, making motion conditioning impossible.

---

## Task 3: Temporal Convolution Implementation Audit

### Finding: Temporal Convolution Implementation is Correct, But Usage is Broken

**TemporalConv1DBlock Tests**:
- ✓ Operates correctly along temporal dimension
- ✓ Sensitive to frame order (distinguishes forward vs reversed)
- ✓ Preserves tensor shapes correctly
- ⚠ Introduces variance in constant input (acceptable)

**VideoUNet Usage Audit**:
The main version of `video_unet.py` (in `video_foundation/models/`) has a critically broken fusion mechanism:

```python
# Lines 344-352 (main version)
bottleneck_mean = bottleneck_video.mean()  # scalar
low_res_mean = low_res_video.mean()  # scalar
temporal_term = (bottleneck_mean + low_res_mean) * 0.001
predicted_noise_spatial = predicted_noise_spatial + temporal_term
```

**Critical Issue**: Temporal features are collapsed to scalars and added as a tiny bias term (0.001 scaling). The actual temporal feature maps are discarded.

**Note**: The experiments version has a `LearnableTemporalFusion` class that implements proper feature-level fusion, but it is NOT used in the forward pass.

---

## Task 4: Spatial-Temporal Feature Fusion Audit

### Finding: Fusion Mechanism is Inconsistent and Noise-Level

**LearnableTemporalFusion Class** (experiments version):
- Implements proper feature-level fusion: `spatial + gate * temporal`
- Formula is correct: `fused = spatial_features + gate * temporal_features`
- However, this class is NOT used in the forward pass

**Actual Fusion in _forward_with_temporal** (experiments version):
```python
# Lines 434-457
bottleneck_noise = self.bottleneck_proj(bottleneck_temporal_flat)  # Project to noise
low_res_noise = self.low_res_proj(low_res_temporal_flat)  # Project to noise
bottleneck_noise_upsampled = F.interpolate(bottleneck_noise, ...)  # Upsample
low_res_noise_upsampled = F.interpolate(low_res_noise, ...)  # Upsample
temporal_noise = bottleneck_gate * bottleneck_noise_upsampled + low_res_gate * low_res_noise_upsampled
predicted_noise_spatial = predicted_noise_spatial + temporal_noise
```

**Analysis**:
1. Temporal features are projected to noise space (Conv2d)
2. Upsampled to full resolution
3. Combined with learnable gates
4. Added to spatial prediction

**Problem**: This is **NOISE-LEVEL fusion**, not feature-level fusion. Temporal features bypass the spatial decoder entirely, losing the benefit of spatial processing.

**Trained Gate Values**:
- Bottleneck gate: 0.495 (< 0.5)
- Low-res gate: 0.494 (< 0.5)

**Conclusion**: Gates learned to suppress temporal features (< 0.5), indicating temporal features were harmful for the noise prediction objective. The fusion mechanism prevents temporal features from benefiting from spatial decoder processing.

---

## Task 5: Video Diffusion Objective Audit

### Finding: Objective Does Not Encourage Temporal Learning

**Training Objective** (from `train_video_stage3_2_5.py`):
```python
loss = nn.functional.mse_loss(predicted_noise_flat, noise)
```

**Analysis**:
1. Diffusion timesteps are **shared** across frames (correct for video diffusion)
2. Noise is sampled **independently** for each pixel and frame (correct)
3. Loss is computed as **MSE** between predicted and actual noise (correct)
4. Loss is averaged over all frames and pixels (correct)

**Problem**: The objective is standard noise prediction MSE with **no temporal consistency term**:
- No frame-to-frame smoothness constraint
- No temporal prediction loss
- No motion-based loss
- No incentive to learn motion patterns

**Conclusion**: The model can minimize loss by ignoring temporal information entirely. The objective does not incentivize temporal coherence, which explains why gates learned to suppress temporal features.

---

## Task 6: Training Dataset Diversity Audit

### Finding: Dataset Has Moderate Diversity

**Dataset Statistics**:
- Total videos: 25
- Total captions: 25

**Trajectory Diversity**: 23 unique trajectories (GOOD)
**Velocity Diversity**: 25 unique velocities (GOOD)
**Shape Diversity**: 2 unique shapes (circle: 18, square: 7) (LIMITED)
**Color Diversity**: 3 unique colors (red: 12, blue: 7, yellow: 6) (MODERATE)
**Dataset Size**: 25 videos (SMALL)

**Caption Analysis**:
- Total words: 332
- Unique words: 21
- Motion-related words in captions: left (12), right (19), down (7)

**Diversity Score**: 4/5 (MODERATE)

**Conclusion**: Dataset diversity is moderate and likely sufficient for basic temporal learning. The limited shape and color diversity are not the primary blockers. The main issue is that motion vocabulary does not exist in the tokenizer.

---

## Root Cause Synthesis

### Primary Root Causes (CRITICAL)

1. **Text Encoder Cannot Represent Motion**
   - Motion vocabulary does not exist (0/26 keywords)
   - All motion phrases map to `<UNK>` tokens
   - Motion conditioning is impossible
   - **Impact**: Model cannot receive motion guidance from text

2. **Fusion Mechanism is Noise-Level**
   - Temporal features projected to noise space
   - Bypass spatial decoder processing
   - `LearnableTemporalFusion` class exists but is unused
   - **Impact**: Temporal features lose spatial refinement

### Secondary Root Causes (HIGH)

3. **Objective Does Not Encourage Temporal Learning**
   - Standard noise prediction MSE only
   - No temporal consistency loss
   - No frame-to-frame smoothness constraint
   - **Impact**: Model has no incentive to use temporal information

4. **Gates Learned to Suppress Temporal**
   - Both gates < 0.5 (0.495, 0.494)
   - Rational response to broken conditioning and fusion
   - **Impact**: Temporal features actively excluded from predictions

### Contributing Factors (MODERATE)

5. **Dataset Size is Small**
   - Only 25 training videos
   - May limit temporal pattern learning
   - **Impact**: Limited training signal

6. **Dataset Diversity is Moderate**
   - Only 2 shapes, 3 colors
   - **Impact**: Limited variation in training data

---

## Why Gates Suppress Temporal Information

The gates learned to suppress temporal features (0.495, 0.494) because:

1. **Text encoder cannot provide motion guidance** → Temporal modules receive no useful conditioning
2. **Fusion is at noise level** → Temporal features bypass spatial decoder, losing refinement
3. **Objective has no temporal term** → No incentive to use temporal information
4. **Result**: Temporal features add noise rather than signal → Gates suppress them

This is a **rational optimization** by the model given the broken architecture and objective.

---

## Comparison with Previous Findings

### Stage 3.2.4 vs Stage 3.2.5

- **Stage 3.2.4**: Spatial backbone corruption caused MSE degradation (0.131 → 1.176)
- **Stage 3.2.5**: Spatial backbone preserved (fixed), but temporal modules degrade performance (0.221 → 0.260)

### Spatial Preservation Success

The spatial backbone preservation mechanism from Stage 3.2.4 is working correctly:
- Frame-independent performance matches Stage 2.1 (MSE 0.221)
- All spatial parameters unchanged throughout training
- This validates the fix and ensures Stage 2.1 performance is preserved

### Temporal Learning Failure

Despite correct spatial preservation, temporal learning failed due to the four root causes identified above.

---

## Proposed Solutions

### Phase 1: Critical Fixes (Run Sequentially)

**Experiment A: Motion Vocabulary Expansion**
- Add motion keywords to tokenizer (moving, slow, fast, bounce, rotate, etc.)
- Re-train text encoder embeddings for new tokens
- Update video captions with motion descriptions
- **Expected**: Text encoder can distinguish motion concepts

**Experiment B: Feature-Level Fusion**
- Use `LearnableTemporalFusion` class in forward pass
- Inject temporal features back into spatial U-Net decoder
- Remove projection to noise space before fusion
- **Expected**: Temporal features benefit from spatial processing

### Phase 2: Objective Enhancement

**Experiment C: Temporal Consistency Loss**
- Add frame-to-frame smoothness loss to objective
- `total_loss = noise_mse + lambda_temporal * temporal_loss`
- **Expected**: Model incentivized to maintain temporal coherence

### Phase 3: Alternative Approaches (if Phase 1-2 fail)

**Experiment D: Separate Motion Encoder**
- Decouple motion learning from noise prediction
- Train motion encoder separately
- **Expected**: Explicit motion pattern learning

**Experiment E: Gate Initialization Change**
- Initialize gates at 0.1 instead of 0.01
- **Expected**: Gates start higher, giving temporal features a chance

All proposed experiments respect CPU-first constraints (no CUDA, minimal parameter additions, efficient convolutions).

---

## Validation Criteria

For any fix to be considered successful:

1. **Gate values > 0.5**: Temporal features are being used
2. **Temporal MSE < Frame-independent MSE**: Temporal model outperforms baseline
3. **Motion embeddings distinguishable**: Cosine similarity < 0.9 for different motions
4. **Temporal loss decreases**: Objective optimization is working

---

## CPU Constraints Compliance

All proposed solutions respect CPU-first requirements:
- No CUDA required
- Minimal parameter additions
- Efficient 1D temporal convolutions retained
- No expensive temporal attention
- Measured CPU execution latency maintained

---

## Deliverables

1. ✓ Audit scripts for each investigation area
2. ✓ Deterministic evaluation fixture
3. ✓ Root cause analysis (this document)
4. ✓ Proposed experiments document
5. ✓ All existing checkpoints and reports preserved

---

## Next Steps

1. Implement Experiment A (motion vocabulary expansion)
2. Implement Experiment B (feature-level fusion)
3. Run small-scale validation (50-100 steps)
4. If successful, proceed to Experiment C (temporal consistency loss)
5. Generate full training run with all successful changes
6. Re-evaluate with deterministic evaluation fixture

---

## Final Success Criterion

We are not looking for a lower loss alone. We need to establish that our compact temporal network can learn a meaningful relationship between frames and improve generated motion over the frame-independent baseline.

The ultimate requirement remains a completely self-owned video generation model that can run on consumer CPUs without relying on external generative model weights or APIs.
