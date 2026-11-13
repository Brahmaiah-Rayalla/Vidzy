# Stage 3.2.9L.3 Experiment Design
## Controlled Motion Learning Training (100 Steps)

**Date**: 2026-11-05
**Status**: INITIALIZING
**Objective**: Validate motion learning after gradient-flow repair through controlled 100-step training

---

## 1. Objective

Stage 3.2.9L.2 successfully validated the gradient-flow repair. The following issues have been corrected:
- In-place assignment in `SplitEmbedding.forward()` broke motion embedding gradients
- The split encoder incorrectly used the expanded vocabulary size when extracting original embeddings
- Original appearance embeddings and motion embeddings are now separated correctly
- Motion embeddings, temporal convolutions, and fusion parameters receive gradients
- Frozen spatial parameters remain unchanged
- Checkpoint serialization and restoration work correctly

**Current Objective**: Conduct the first meaningful post-repair training experiment and determine whether the model can learn prompt-controlled motion.

**Experimental Approach**: Run two controlled experiments:
- **Experiment B**: Motion embeddings only
- **Experiment C**: Motion embeddings plus temporal learning

**Training Budget**: Exactly 100 optimizer steps for each experiment (no automatic extension)

---

## 2. Artifact Protection

### Directory Structure
```
experiments_stage3_2_9B/phase2L_training_repair/controlled_training/
├── ARTIFACT_HASHES.json                    # SHA256 hashes of critical artifacts
├── STAGE3_2_9L3_EXPERIMENT_DESIGN.md       # This document
├── experiment_B/                           # Experiment B outputs
│   ├── checkpoints/
│   │   ├── checkpoint_step_000025.pt
│   │   ├── checkpoint_step_000050.pt
│   │   ├── checkpoint_step_000075.pt
│   │   └── checkpoint_step_000100.pt
│   ├── logs/
│   └── metrics/
├── experiment_C/                           # Experiment C outputs
│   ├── checkpoints/
│   │   ├── checkpoint_step_000025.pt
│   │   ├── checkpoint_step_000050.pt
│   │   ├── checkpoint_step_000075.pt
│   │   └── checkpoint_step_000100.pt
│   ├── logs/
│   └── metrics/
└── deliverables/                          # Final reports
```

### Protected Artifacts (with SHA256 Hashes)

| Artifact | Path | SHA256 Hash |
|----------|------|-------------|
| Stage 2.1 spatial checkpoint | `experiments_stage3_2_7/checkpoints/checkpoint_stage2_spatial.pt` | a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c |
| Stage 2 motion expanded checkpoint | `experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt` | 9a75aadc6bae18b9083797fa4bf2424f58fdf73118eac620e259e092096fa28a |
| Tokenizer (motion expanded) | `experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json` | 9227a1001b4d3503fa59729c04a26a01307d4e0a050b579e858ee8a809d679e8 |
| Text encoder split | `experiments_stage3_2_7/video_foundation/models/text_encoder_split.py` | 0e5b90ba9df87d0c5e1a9ad1dde246eacb7e72c145db7588c88cb8c8e74e0789 |
| Video UNet fixed | `experiments_stage3_2_7/video_foundation/models/video_unet_fixed.py` | 999e206cf226c63eee798832aaad8e3e72450225db663ce20dd93f9b0105797e |

**Policy**: Fresh experiment checkpoints will be used for this training run. Previous Experiment B or C results will not be overwritten.

---

## 3. Experiment Initialization

### Common Initialization Requirements

Both experiments must begin with **identical common parameters**:
- Common spatial parameters initialized identically
- Motion embeddings initialized identically
- Temporal-only parameters using validated Stage 3 architecture initialization

### Experiment B Configuration

**Trainable Components**:
- Motion embeddings (original_vocab_size=42, motion_vocab_size=40)

**Frozen Components**:
- Temporal processing (disabled)
- Spatial backbone (frozen)
- Appearance embeddings (frozen)
- Transformer layers (frozen)

**Other text encoder parameters**: Follow validated trainability configuration

### Experiment C Configuration

**Trainable Components**:
- Motion embeddings (original_vocab_size=42, motion_vocab_size=40)
- Temporal convolutions
- Temporal fusion parameters

**Frozen Components**:
- Spatial backbone (frozen)
- Appearance embeddings (frozen)
- Transformer layers (frozen)

**Other text encoder parameters**: Same trainability configuration as B

### Initialization Constraints

- ✅ Initialize common spatial parameters identically
- ✅ Initialize motion embeddings identically
- ✅ Initialize temporal-only parameters using validated Stage 3 architecture
- ✅ Record all initial parameter tensors
- ✅ Record trainable parameter names and counts
- ✅ Record frozen parameter names and counts
- ❌ Do not initialize C from a previously corrupted or unverified checkpoint
- ❌ Do not introduce new architecture components
- ❌ Do not change fusion gate initialization to 0.5

---

## 4. Training Configuration

### Hardware Configuration
- **Execution**: CPU-only
- **PyTorch threads**: 8
- **Device**: CPU

### Model Configuration
- **Batch size**: 1
- **Resolution**: 64×64
- **Frames**: 16 RGB frames
- **Conditioning**: Genuine text conditioning
- **Text encoder**: Corrected SplitEmbedding implementation
- **Tokenizer**: Validated Stage 3 tokenizer (motion expanded)
- **Noise schedule**: Existing diffusion noise schedule
- **Objective**: Existing validated denoising objective

### Training Budget
- **Experiment B**: 100 optimizer steps (exactly)
- **Experiment C**: 100 optimizer steps (exactly)
- **Enforcement**: Explicit optimizer-step limits
- **No epochs**: Do not use epochs as a substitute for optimizer steps

### Reproducibility
- **Random seeds**: Record exact random seeds
- **Configuration**: Record complete training configurations

---

## 5. Training Data and Supervision

### Dataset Verification
Before training, verify that captions correspond to their actual trajectory metadata.

### Category Balance
Include balanced representation of supported categories:
- Leftward movement
- Rightward movement
- Upward movement
- Downward movement
- Stationary objects

Where supported by the existing dataset, also include:
- Slow movement
- Fast movement
- Bouncing movement

**Requirement**: Verify that the stationary category is sufficiently represented.

### Sample Counts
Report the actual sample counts by category.

### Data Splits
- **Training split**: Used for training
- **Validation split**: Used for validation only
- **Separation**: Do not allow validation examples to enter training
- **Stability**: Do not change the dataset during this experiment

---

## 6. Loss Consistency

### Requirement
Experiments B and C must use the same fundamental denoising objective.

### Consistent Components
- Noise prediction target
- Noise schedule
- Timestep sampling
- Loss calculation
- Reduction method
- Input normalization
- Video frame weighting
- Text conditioning
- Batch construction

### Matching
Use matched training examples, noisy inputs, noise tensors, and timestep selections wherever possible.

### Reporting
- Report per-frame losses
- Report aggregate losses
- If any loss component differs between B and C, document the reason
- Do not artificially force the two loss values to match

---

## 7. Training Checkpoints and Monitoring

### Checkpoint Schedule
Save checkpoints at:
- Step 25
- Step 50
- Step 75
- Step 100

**Requirements**:
- Save optimizer states alongside model checkpoints
- Include complete model state dictionaries
- Include training configuration metadata

### Per-Step Monitoring

**Experiment B** (record for every optimizer step):
- Training loss
- Motion embedding gradient norm
- Motion embedding parameter delta
- Updated embedding rows
- Frozen spatial integrity
- Step duration
- CPU configuration
- Peak RAM
- Learning rate
- Validation loss (at consistent intervals)

**Experiment C** (record for every optimizer step):
- Training loss
- Motion embedding gradient norm
- Temporal convolution gradient norm
- Fusion gradient norm
- Fusion gate values
- Temporal residual magnitude
- Parameter deltas for each trainable group
- Frozen spatial integrity
- Step duration
- CPU configuration
- Peak RAM
- Learning rate
- Validation loss (at consistent intervals)

### Parameter Counting
Distinguish between:
- Number of parameter tensors
- Number of scalar parameters
- Number of trainable tensors
- Number of trainable scalar elements

**Constraint**: Do not use embedding standard deviation alone as evidence of training.

---

## 8. Integrity Assertions

### Post-Update Verification
After every optimizer update, verify that:
- Spatial parameters remain frozen
- Appearance embeddings remain frozen
- Only approved parameter groups change
- Gradients are finite
- Loss is finite
- Optimizer updates affect the intended parameters

### Stop Conditions
Stop immediately if:
- Frozen parameters change
- Intended trainable parameters stop receiving gradients
- NaN or infinity appears
- An unexpected parameter enters the optimizer
- Checkpoint integrity is compromised

**Policy**: Do not silently skip failed updates.

---

## 9. Checkpoint Roundtrip Validation

### Post-Experiment Validation
After each experiment:
1. Save the final checkpoint
2. Load it into a fresh model instance
3. Compare all model parameter tensors
4. Compare text encoder and tokenizer configuration
5. Compare temporal parameter values
6. Restore optimizer state
7. Confirm the loaded checkpoint reproduces the saved model output
8. Verify the common B/C parameters against their initial states and against each other
9. Ensure C's temporal-specific tensors exist and have been updated
10. Produce complete checkpoint comparison reports

---

## 10. Prompt-Conditioned Evaluation

### Inference Configuration
- Use the corrected inference adapter
- Do not use random embeddings

### Test Prompts
Generate matched samples using:
- "Move left"
- "Move right"
- "Move upward"
- "Move downward"
- "Remain stationary"

### Reproducibility
- Use three fixed seeds per supported prompt category where CPU resources permit
- Use identical initial noise for comparable samples

### Evaluation Metrics
- Direction accuracy
- Object-centre displacement
- Direction consistency
- Stationary drift
- Object detection rate
- Shape and colour preservation

### Comparison Groups
Compare:
- **A**: Experiment B
- **B**: Experiment C with temporal enabled
- **C**: Experiment C with temporal disabled (D)

**Important**:
- B versus C compares two training configurations
- C versus D isolates the inference-time effect of temporal processing using the same checkpoint
- Do not claim meaningful motion learning merely because the predictions differ

---

## 11. CPU Performance

### Metrics to Record
- Total training time
- Median step time
- Mean step time
- P90 step time
- Maximum step time
- Outlier count
- Peak memory
- CPU thread count

### Separation
- Record text encoding timing separately
- Record inference timing separately

### Constraints
- Keep CPU mandatory
- Do not increase resolution or frame count

---

## 12. Deliverables

### Required Documents
1. `STAGE3_2_9L3_EXPERIMENT_DESIGN.md` - This document
2. `STAGE3_2_9L3_TRAINING_RESULTS.json` - Training results summary
3. `STAGE3_2_9L3_PARAMETER_UPDATES.json` - Parameter update tracking
4. `STAGE3_2_9L3_GRADIENT_MONITORING.json` - Gradient monitoring data
5. `STAGE3_2_9L3_LOSS_COMPARISON.json` - Loss comparison between B and C
6. `STAGE3_2_9L3_CHECKPOINT_INTEGRITY.json` - Checkpoint integrity verification
7. `STAGE3_2_9L3_MOTION_EVALUATION.json` - Motion evaluation metrics
8. `STAGE3_2_9L3_CPU_PERFORMANCE.json` - CPU performance metrics
9. `STAGE3_2_9L3_FINAL_REPORT.md` - Final experiment report

### Artifacts
- All checkpoints (steps 25, 50, 75, 100 for both experiments)
- Optimizer states
- Generated videos
- Generated frames
- Training logs

**Location**: `experiments_stage3_2_9B/phase2L_training_repair/controlled_training/`

---

## 13. Acceptance Criteria

### Training Milestone
The experiment passes as a training milestone when:
- Both B and C complete exactly 100 optimizer updates
- Motion embeddings update successfully in both
- Temporal parameters update successfully in C
- Fusion parameters update successfully in C
- Spatial parameters remain unchanged
- Appearance embeddings remain unchanged
- Checkpoint roundtrip preserves all updates
- Genuine text conditioning is used during training and inference
- Validation data remains isolated
- All generated samples and evaluation metrics are reproducible

**Important**: A successful training milestone does not automatically establish successful motion control.

---

## 14. Final Classification

### Possible Outcomes

**TRAINING VERIFIED**
- Both experiments learn through the intended parameter groups
- Preserve the spatial foundation

**MOTION LEARNING PROMISING**
- Motion-related validation metrics improve relative to the initial baseline

**TEMPORAL BENEFIT PROMISING**
- Experiment C demonstrates measurable improvements over B
- No unacceptable stationary or appearance degradation

**INCONCLUSIVE**
- Training is technically correct
- 100 steps do not yet demonstrate meaningful motion improvements

**TRAINING FAILURE**
- Expected parameter groups fail to update
- Checkpoint integrity is compromised

---

## 15. Next-Stage Decision

### No Automatic Extension
Do not automatically extend training to 1,000 steps.

### Review Criteria
Before deciding on extension, review:
- Actual parameter updates
- Motion conditioning sensitivity
- Validation trends
- Direction accuracy
- Stationary behaviour
- Temporal contribution
- CPU training cost

### Recommendation Logic
- **If results are promising**: Recommend next controlled extension (e.g., 250 steps) using the same validated initialization and training design
- **If results remain weak**: Investigate the learning objective or motion supervision before extending training

### Final Principle
The goal is not to complete a large training run. The goal is to establish that the repaired model learns to translate text instructions into predictable movement while retaining the existing spatial foundation.

---

## 16. Implementation Plan

### Phase 1: Setup
1. ✅ Create controlled_training directory
2. ✅ Record artifact hashes
3. ✅ Create experiment design document
4. ⏳ Set up training script with comprehensive monitoring

### Phase 2: Initialization
5. ⏳ Initialize Experiment B (motion only)
6. ⏳ Initialize Experiment C (motion + temporal)
7. ⏳ Record initial parameters and trainable/frozen counts

### Phase 3: Training
8. ⏳ Run Experiment B training (100 steps)
9. ⏳ Run Experiment C training (100 steps)
10. ⏳ Monitor gradients, losses, and parameter updates at each step
11. ⏳ Verify integrity assertions after each update

### Phase 4: Validation
12. ⏳ Checkpoint roundtrip validation for both experiments
13. ⏳ Prompt-conditioned motion evaluation
14. ⏳ Compare B vs C vs D (C with temporal disabled)

### Phase 5: Reporting
15. ⏳ Generate all deliverable reports
16. ⏳ Final classification and next-stage recommendation

---

**Document Created**: 2026-11-05
**Stage**: 3.2.9L.3
**Status**: READY FOR IMPLEMENTATION
