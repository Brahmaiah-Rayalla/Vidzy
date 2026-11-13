# Stage 3.2.9B - Fair A/B/C Comparison Methodology

**Date:** 2026-11-01

## 1. Shared Resources

All experiments use identical:
- **Validation dataset:** `experiments_stage3_2_7/data_synthetic/val_metadata.json` (160 clips)
- **Test dataset:** `experiments_stage3_2_7/data_synthetic/test_metadata.json` (160 clips)
- **Random seeds:** 42 for all operations
- **Initial noisy inputs:** Generated from fixed seed
- **Diffusion timesteps:** Same sampling strategy
- **Prompts:** From dataset metadata
- **Sampling settings:** 100 DDIM steps, eta=0.0
- **Evaluation implementation:** Same code path

## 2. Experiment Configurations

### Experiment A - Spatial-Only Baseline
- **Model:** Original Stage 2.1 ConditionalUNet
- **Processing:** Frame-independent
- **Temporal modules:** Disabled (not present)
- **Trainable parameters:** None
- **Purpose:** Reference baseline

### Experiment B - Motion-Conditioning Only
- **Model:** Stage 3 VideoUNetFixed
- **Processing:** Frame-independent with split embeddings
- **Temporal modules:** Disabled via `enable_temporal_processing=False`
- **Trainable parameters:** 40 motion embeddings (10,240 parameters)
- **Purpose:** Isolate motion language learning

### Experiment C - Full Temporal Learning
- **Model:** Stage 3 VideoUNetFixed
- **Processing:** Full temporal processing
- **Temporal modules:** Enabled via `enable_temporal_processing=True`
- **Trainable parameters:** Motion + temporal + fusion (134,274 parameters)
- **Purpose:** Primary experiment

## 3. Fair Comparison Guarantees

### 3.1 Identical Inputs
```python
# All experiments use same inputs
torch.manual_seed(42)
noise = torch.randn(batch_size, channels, frames, height, width)
timesteps = torch.randint(0, 1000, (batch_size,))
```

### 3.2 Identical Prompts
Prompts are taken from dataset metadata in fixed order:
```python
prompts = [clip['caption'] for clip in val_metadata]
```

### 3.3 Identical Evaluation
Same evaluation function called for all experiments:
```python
metrics = evaluate_clip(model, clip_data, evaluator)
```

### 3.4 No Temporal Leakage
- Experiment A: No temporal modules exist
- Experiment B: Temporal modules exist but disabled
- Experiment C: Temporal modules enabled

Verification: Check `enable_temporal_processing` flag before evaluation.

## 4. Comparison Metrics

### 4.1 Primary Metrics
- **Motion direction accuracy:** Higher is better
- **Trajectory error:** Lower is better
- **Speed prediction error:** Lower is better

### 4.2 Secondary Metrics
- **Color consistency:** Higher is better
- **Temporal smoothness:** Moderate is best (not too high = stationary, not too low = flicker)
- **Unwanted movement (stationary):** Lower is better

### 4.3 Success Criteria
Experiment C is successful if:
- Motion direction accuracy > Experiment B by ≥5%
- Trajectory error < Experiment B by ≥10%
- Unwanted movement (stationary) not significantly worse than baseline

## 5. Statistical Comparison

For each metric, compute:
- Mean across all clips
- Standard deviation
- 95% confidence interval (if sample size permits)

Report:
- Absolute difference between experiments
- Relative improvement (%)
- Statistical significance (if applicable)

## 6. Failure Conditions

Temporal learning is considered unsuccessful if:
- Experiment C does not outperform Experiment B on primary metrics
- Experiment C degrades spatial quality significantly
- Experiment C introduces unwanted motion for stationary prompts
- Frozen parameters are modified during training

## 7. Reporting

Comparison table format:

| Metric | Experiment A | Experiment B | Experiment C | B vs A | C vs B | C vs A |
|--------|---------------|---------------|---------------|--------|--------|--------|
| Direction Acc | X% | Y% | Z% | +Δ% | +Δ% | +Δ% |
| Trajectory Error | X | Y | Z | -Δ | -Δ | -Δ |
| ... | ... | ... | ... | ... | ... | ... |

Color coding:
- Green: Improvement
- Red: Degradation
- Yellow: No significant change
