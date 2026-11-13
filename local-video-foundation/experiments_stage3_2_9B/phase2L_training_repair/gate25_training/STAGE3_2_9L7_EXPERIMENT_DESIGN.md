# Stage 3.2.9L.7 - Experiment Design: Controlled 25% Fusion Gate Initialization

**Date:** 2026-11-06
**Status:** Design Document

## Objective

Compare two otherwise identical training configurations differing only in fusion gate initialization:
- **Experiment C1:** 1% fusion activation (existing baseline)
- **Experiment C25:** 25% fusion activation (new hypothesis)

Determine whether 25% initialization allows temporal modules to learn more effectively while preserving spatial quality and stationary control.

## Experimental Design

### Hypothesis

Increasing fusion gate initialization from 1% to 25% will:
1. Allow fusion gates to converge more effectively
2. Enable temporal features to contribute meaningfully
3. Improve prompt-controlled motion without spatial degradation

### Controlled Variables

Both experiments share identical:
- Spatial U-Net parameters (frozen)
- Appearance embeddings (frozen)
- Motion embeddings (initial state)
- Text encoder parameters (frozen)
- Temporal convolution parameters (initial state)
- Normalization parameters
- Training data configuration
- Model configuration
- Optimizer configuration (AdamW, lr=0.0002, wd=0.01)
- Training budget (100 steps)

### Independent Variable

**Fusion gate initialization:**
- C1: Raw logit = -4.595 → activated = 0.01 (1%)
- C25: Raw logit = -1.099 → activated = 0.25 (25%)

### Trainable Parameters

- Motion embeddings: 10,240
- Temporal convolutions: 124,032
- Fusion gates: 2
- Total: 134,274

### Frozen Parameters

- Spatial U-Net backbone: 9,439,171
- Appearance embeddings: 10,752
- Text encoder transformer layers

## Training Configuration

### Hardware
- CPU-only execution
- 8 PyTorch threads

### Model
- Batch size: 1
- Resolution: 64×64
- Frames: 16 RGB
- Timesteps: 1000

### Optimizer
- AdamW
- Learning rate: 0.0002
- Weight decay: 0.01

### Training Budget
- Smoke test: 10 steps per configuration
- Controlled pilot: 100 steps per configuration (after smoke tests pass)

### Checkpoints
Save at steps: 25, 50, 75, 100

## Success Criteria

### Smoke Test Success
- Motion embeddings update
- Temporal parameters update
- Both fusion gates receive gradients
- Fusion gate values change
- Frozen parameters remain unchanged
- All losses and gradients finite
- No spatial degradation

### Pilot Success
- Both experiments complete 100 steps
- C25 shows meaningful gate convergence (>5% change)
- C25 loss ≤ C1 loss (or meaningful convergence trend)
- No spatial parameter corruption
- No appearance degradation

### Motion Improvement Success
- C25 shows better direction accuracy than C1
- C25 shows better motion consistency than C1
- No unacceptable stationary drift in C25
- No spatial degradation in C25

## Risk Mitigation

### If C25 shows instability
- Stop training immediately
- Investigate intermediate activation (10%)
- Consider separate gate learning rates

### If C25 shows spatial degradation
- Stop training immediately
- Preserve C1 as baseline
- Investigate temporal initialization

### If no meaningful difference
- Consider higher activation (50%)
- Consider separate gate configurations
- Investigate gate learning rates

## Deliverables

1. STAGE3_2_9L7_EXPERIMENT_DESIGN.md (this document)
2. STAGE3_2_9L7_INITIALIZATION_COMPARISON.json
3. STAGE3_2_9L7_SMOKE_TEST_RESULTS.json
4. STAGE3_2_9L7_TRAINING_RESULTS.json
5. STAGE3_2_9L7_FUSION_GATE_PROGRESSION.json
6. STAGE3_2_9L7_TEMPORAL_ACTIVATIONS.json
7. STAGE3_2_9L7_LOSS_COMPARISON.json
8. STAGE3_2_9L7_MOTION_EVALUATION.json
9. STAGE3_2_9L7_CHECKPOINT_INTEGRITY.json
10. STAGE3_2_9L7_CPU_PERFORMANCE.json
11. STAGE3_2_9L7_FINAL_REPORT.md

## Next Steps

1. Implement controlled training script with dual initialization
2. Run 10-step smoke tests
3. If smoke tests pass, run 100-step pilot
4. Analyze results and classify
