# Stage 3.2.5 Final Report: Extended Temporal Training

**Date:** November 1, 2026
**Objective:** Establish meaningful temporal learning through controlled extended training with spatial backbone preservation

---

## Executive Summary

**Result:** Temporal training did NOT achieve the objective. After 500 steps of training, the temporal model performs **worse** than the frame-independent baseline (MSE 0.374 vs 0.333, -12.32% degradation). The gates learned to suppress temporal information (0.49 instead of 0.5), indicating the model determined temporal features were harmful for noise prediction.

**Critical Finding:** The frame-independent mode successfully preserved Stage 2.1 performance (MSE 0.333, matches Stage 2.1 exactly), confirming spatial backbone preservation worked correctly. However, temporal modules learned to degrade performance rather than improve it.

---

## 1. Training Configuration

### 1.1 Setup
- **Dataset:** 10 synthetic videos (controlled_video/train)
- **Duration:** 500 training steps
- **Device:** CPU-only (8 cores, 15.65 GB RAM)
- **Learning rate:** 1e-4 (temporal and fusion)
- **Spatial backbone:** Frozen throughout training
- **Text encoder:** Frozen throughout training
- **Trainable parameters:** 124,034 (124,032 temporal + 2 fusion)

### 1.2 Spatial Preservation
- **Pre-training validation:** 504/504 spatial parameters frozen ✓
- **Runtime validation:** Checked every 50 steps, all unchanged ✓
- **Post-training validation:** All 504 spatial parameters unchanged ✓
- **Spatial scalar parameters:** 9,439,171 preserved

### 1.3 CPU Performance
- **Training time per step:** ~3.7-4.0 seconds
- **Total training time:** ~31 minutes for 500 steps
- **Memory overhead:** Minimal (training with frozen spatial backbone)

---

## 2. Training Progress

### 2.1 Metrics Over Training

| Step | Temporal MSE | Bottleneck Gate | Low-Res Gate | Frame-Indep MSE |
|------|--------------|-----------------|-------------|-----------------|
| 0 (initial) | 0.206 | 0.5025 | 0.5025 | 0.0108 |
| 50 | 0.104 | 0.5015 | 0.5014 | 0.0108 |
| 100 | 0.092 | 0.5006 | 0.5004 | 0.0108 |
| 150 | 0.083 | 0.4998 | 0.4995 | 0.0108 |
| 200 | 0.075 | 0.4990 | 0.4986 | 0.0108 |
| 250 | 0.070 | 0.4983 | 0.4977 | 0.0108 |
| 300 | 0.065 | 0.4976 | 0.4969 | 0.0108 |
| 350 | 0.061 | 0.4969 | 0.4961 | 0.0108 |
| 400 | 0.057 | 0.4962 | 0.4953 | 0.0108 |
| 500 (final) | 0.058 | 0.4948 | 0.4937 | 0.0108 |

**Observations:**
- Temporal MSE decreased from 0.206 to 0.058 during training (72% improvement)
- Gates decreased from 0.5025 to 0.4948 (learned to suppress temporal)
- Frame-independent MSE remained constant at 0.0108 (spatial preserved)

### 2.2 Training Loss
- Initial average loss: 0.205
- Final average loss: ~0.05
- Loss decreased steadily throughout training

---

## 3. Final Evaluation Results

### 3.1 Matched Baseline Comparison (10 samples)

| Baseline | MSE | Correlation | Improvement over Zero |
|----------|-----|-------------|------------------------|
| A. Zero-prediction | 1.000 | N/A | 0% |
| B. Stage 2.1 spatial | 0.333 | 0.852 | 66.68% |
| C. Frame-independent | 0.333 | 0.853 | 66.74% |
| D. Temporal (trained) | 0.374 | 0.838 | 62.64% |

### 3.2 Key Metrics
- **Temporal vs Frame-independent:** -12.32% degradation
- **Gate values:** 0.495 (bottleneck), 0.494 (low-res)
- **Frame-independent preservation:** ✓ Identical to Stage 2.1

---

## 4. Analysis

### 4.1 Why Temporal Training Failed

**Hypothesis:** The temporal modules learned to add noise rather than signal. Several factors may contribute:

1. **Inappropriate objective:** Noise prediction MSE does not directly encourage temporal coherence. The model may learn that adding temporal features increases prediction error because they don't align with the target noise.

2. **Insufficient data:** Only 10 videos in the training set. Temporal patterns may not be learnable from such limited data.

3. **Architecture mismatch:** The temporal modules process features at bottleneck and low-res levels, but these features may not contain sufficient temporal information for meaningful learning.

4. **Gate learning direction:** Gates learned to suppress temporal features (0.49 < 0.5), indicating the model found temporal contributions harmful for the noise prediction objective.

### 4.2 Spatial Preservation Success

The spatial backbone preservation mechanism worked perfectly:
- Frame-independent performance matches Stage 2.1 exactly (MSE 0.333 vs 0.333)
- All 504 spatial parameters unchanged throughout 500 steps
- This validates the fix from Stage 3.2.4 and ensures Stage 2.1 performance is preserved

### 4.3 Comparison with Previous Issue

In Stage 3.2.4, the corrupted spatial backbone caused MSE to degrade from 0.131 to 1.176. After fixing spatial preservation, the frame-independent mode now achieves MSE 0.333 (on different evaluation data). The key achievement is that spatial performance is now reliably preserved.

---

## 5. Deliverables

### 5.1 Checkpoints
- `checkpoints_stage3_2_5/checkpoint_step_000050.pt` through `checkpoint_step_000500.pt`
- `checkpoints_stage3_2_5/checkpoint_final.pt`
- Baseline evaluations at each checkpoint

### 5.2 Scripts
- `train_video_stage3_2_5.py` - Training script with spatial preservation
- `resume_training_stage3_2_5.py` - Resume training script
- `evaluate_baselines_stage3_2_5.py` - Baseline evaluation script
- `benchmark_quick.py` - CPU benchmark script

### 5.3 Reports
- `quick_benchmark_results.json` - CPU performance metrics
- `baseline_evaluation_stage3_2_5.json` - Final baseline comparison
- This report

---

## 6. Recommendations

### 6.1 For Stage 3.3

**DO NOT proceed with current temporal architecture.** The temporal modules are actively degrading performance.

**Required changes before continuing:**

1. **Re-evaluate temporal objective:** Noise prediction MSE may not be the right objective for temporal learning. Consider:
   - Temporal consistency loss
   - Frame-to-frame smoothness loss
   - Explicit motion prediction loss

2. **Increase training data:** 10 videos is insufficient for temporal learning. Need at least 100+ diverse videos.

3. **Redesign temporal architecture:** Current temporal modules at bottleneck/low-res may not capture meaningful temporal information. Consider:
   - 3D convolutions at multiple resolutions
   - Temporal attention mechanisms
   - Optical flow-inspired modules

4. **Different gate initialization:** Gates at 0.5 with sigmoid may bias toward suppression. Consider:
   - Start gates at 0.1 (initially rely on spatial)
   - Allow gates to grow if temporal is useful
   - Remove sigmoid and use learnable scaling directly

### 6.2 Alternative Approach

Consider training a **motion prediction model separately** from the diffusion model:
- Train a model to predict frame T+1 from frame T
- Use motion predictions to guide diffusion sampling
- This decouples temporal learning from noise prediction

### 6.3 For Now

**Stage 3.2.5 status: NOT ACCEPTED**

Acceptance criteria not met:
- ✓ Spatial checkpoint remains unchanged
- ✓ Temporal parameters received updates
- ✗ Temporal model performs WORSE than frame-independent baseline
- ✗ Temporal model does not learn meaningful temporal coherence
- ✗ Temporal model degrades over frame-independent

---

## 7. Conclusion

The spatial preservation fix from Stage 3.2.4 is working correctly and can be reused. However, the temporal learning approach in Stage 3.2.5 is fundamentally flawed - the model learns to suppress temporal features because they hurt the noise prediction objective.

**Key achievement:** Spatial backbone can be reliably frozen and preserved during training, solving the critical corruption issue from Stage 3.2.4.

**Critical blocker:** Temporal modules as currently designed and trained do not improve video generation. A different approach is required for Stage 3.3.

---

## 8. Appendix: Training Logs Summary

### Training Time
- Total steps: 500
- Average time per step: 3.8 seconds
- Total time: ~31 minutes
- Device: CPU (8 cores)

### Spatial Integrity
- Parameters monitored: 504 spatial parameter tensors
- Parameters changed: 0
- Maximum difference: 0.0
- Status: ✓ Fully preserved

### Gate Evolution
- Initial: 0.5025, 0.5025
- Final: 0.4948, 0.4937
- Trend: Decreasing (suppressing temporal)

### Performance Trend
- Initial temporal MSE: 0.206
- Best temporal MSE: 0.057 (step 400)
- Final temporal MSE: 0.058
- Frame-independent MSE: Constant 0.0108 (on synthetic data during training)
- Final evaluation MSE: 0.374 (on different test data)

**Note:** The discrepancy between training MSE (0.058) and evaluation MSE (0.374) suggests the model overfit to the 10 training videos and does not generalize.
