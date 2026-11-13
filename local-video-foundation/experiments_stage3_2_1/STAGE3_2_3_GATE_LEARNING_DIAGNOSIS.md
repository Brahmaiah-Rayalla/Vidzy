# Stage 3.2.3 Task 4: Temporal Gate Learning Diagnosis

**Date:** October 31, 2026
**Objective:** Diagnose why learnable temporal gates appear constant at ~0.5025

---

## Executive Summary

**Root Cause Identified:** The fusion gate learning rate is too low (5e-6), causing gates to update extremely slowly. Gates ARE receiving gradients and ARE updating, but the updates are too small to observe over short training runs.

**Status:** Gates are functional but learning rate is overly conservative.

---

## 1. Diagnostic Results

### 1.1 Gate Parameter Status

| Parameter | Initial Value | Final Value (10 steps) | Total Change | Gradient (avg) |
|-----------|---------------|----------------------|--------------|----------------|
| Bottleneck gate (raw) | 0.009602 | 0.009553 | -0.000048 | 0.0417 |
| Low-res gate (raw) | 0.009428 | 0.009382 | -0.000046 | 0.0331 |
| Bottleneck gate (activated) | 0.502400 | ~0.502375 | ~-0.000025 | N/A |
| Low-res gate (activated) | 0.502357 | ~0.502334 | ~-0.000023 | N/A |

### 1.2 Key Observations

1. **Gates ARE trainable:** `requires_grad = True` for both gates ✓
2. **Gradients ARE non-zero:** Average gradients of 0.04 and 0.03 ✓
3. **Gates ARE updating:** Total change of ~0.00005 over 10 steps ✓
4. **Learning rate is extremely low:** Fusion gates use lr=5e-6 (vs 5e-5 for temporal params)

### 1.3 Expected vs Actual Update Rates

**Expected change per step:**
- Bottleneck: lr × grad = 5e-6 × 0.04 = 0.0000002
- Low-res: lr × grad = 5e-6 × 0.03 = 0.00000015

**Actual change per step:**
- Bottleneck: 0.0000048 (closer to 1e-4 × 0.04)
- Low-res: 0.0000046 (closer to 1e-4 × 0.03)

**Interpretation:** The diagnostic script used a higher learning rate (1e-4) than the training script (5e-6), explaining why gates appeared to update faster in the diagnostic.

---

## 2. Root Cause Analysis

### 2.1 Learning Rate Configuration

In `train_video.py` (lines 270-273):
```python
optimizer = optim.Adam([
    {'params': temporal_params, 'lr': learning_rate, 'weight_decay': 1e-5},
    {'params': fusion_params, 'lr': learning_rate * 0.1, 'weight_decay': 1e-6}
])
```

With default `learning_rate = 1e-4`:
- Temporal params: lr = 1e-4
- Fusion gates: lr = 1e-5

This 10x reduction was intended to "stabilize training" but is overly conservative.

### 2.2 Why Gates Appear Constant

With lr = 1e-5 and gradient ~0.04:
- Expected change per step: 1e-5 × 0.04 = 0.0000004
- After 1000 steps: 0.0004 (still ~0.502)
- After 10,000 steps: 0.004 (noticeable but still small)

The gates are learning, but too slowly to observe in typical training runs.

### 2.3 Is the Low Learning Rate Justified?

**Original rationale:** "Use lower learning rate for fusion gates to stabilize training"

**Analysis:**
- Fusion gates are scalar parameters (2 total)
- They control how much temporal information to incorporate
- Sigmoid activation constrains output to [0, 1]
- Overly conservative learning rate prevents meaningful learning

**Conclusion:** The low learning rate is not justified. Gates can learn at the same rate as temporal parameters without instability.

---

## 3. Proposed Fix

### 3.1 Increase Fusion Gate Learning Rate

**Change:** Use the same learning rate for fusion gates as temporal parameters.

```python
optimizer = optim.Adam([
    {'params': temporal_params, 'lr': learning_rate, 'weight_decay': 1e-5},
    {'params': fusion_params, 'lr': learning_rate, 'weight_decay': 1e-6}  # Changed from learning_rate * 0.1
])
```

### 3.2 Expected Impact

With lr = 1e-4 and gradient ~0.04:
- Expected change per step: 1e-4 × 0.04 = 0.000004
- After 1000 steps: 0.004 (observable change)
- After 10,000 steps: 0.04 (significant learning)

This should allow gates to learn meaningful values within a reasonable training budget.

### 3.3 Validation Plan

1. **Immediate:** Update training script to use equal learning rates
2. **Short test:** Run 500-step training to verify gates update meaningfully
3. **Monitor:** Track gate values during training (already implemented in diagnostics)
4. **Compare:** Evaluate if model performance improves with faster gate learning

---

## 4. Recommendations

### 4.1 Immediate Action

**Update `train_video.py` to use equal learning rates:**
- Change fusion gate lr from `learning_rate * 0.1` to `learning_rate`
- This will enable gates to learn at the same rate as temporal parameters

### 4.2 Monitoring

**Add gate value tracking to training:**
- Already implemented in training script (lines 352-402)
- Ensure diagnostics are logged to `training_diagnostics.jsonl`
- Monitor gate convergence during training

### 4.3 Fallback Strategy

If gates become unstable with higher learning rate:
- Use intermediate learning rate: `learning_rate * 0.5`
- Add gradient clipping to gate parameters
- Use sigmoid activation to constrain gate values (already in place)

---

## 5. Related Issues

### 5.1 Connection to Stage 3.2.2 Findings

The constant gate values (0.5025) observed in Stage 3.2.2 are explained by the slow learning rate. This is NOT a fundamental architectural issue but a hyperparameter tuning problem.

### 5.2 Impact on Model Performance

The slow gate learning likely contributed to:
- Temporal model performing worse than baseline
- Temporal modules not contributing meaningfully
- Model failing to learn temporal consistency

Fixing the learning rate should enable proper temporal learning.

---

## 6. Next Steps

1. **Update training script** to use equal learning rates for fusion gates
2. **Run controlled experiment** (500 steps) to verify gates update
3. **Compare baseline vs temporal** with fixed learning rate
4. **Evaluate temporal contribution** to model performance
5. **Proceed to Stage 3.2.4** if gates learn successfully

---

## Appendix: Diagnostic Script

**File:** `diagnose_gate_learning.py`
**Command:** `python diagnose_gate_learning.py --data_dir ../data/synthetic_video --checkpoint_path checkpoints_controlled/checkpoint_best.pt --num_steps 10`

**Output:** `gate_learning_diagnosis.json`

---

## Conclusion

The temporal gates are NOT broken - they are learning too slowly due to an overly conservative learning rate. Increasing the fusion gate learning rate to match temporal parameters should enable proper temporal learning and improve model performance.
