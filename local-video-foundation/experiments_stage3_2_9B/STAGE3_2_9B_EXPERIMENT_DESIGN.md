# Stage 3.2.9B - Controlled Temporal Learning Validation

**Date:** 2026-11-01
**Objective:** Validate whether the Stage 3 video architecture can learn meaningful, prompt-controlled temporal behavior while preserving the original Stage 2.1 spatial capabilities.

---

## 1. Protected Foundation Artifacts

### 1.1 Reference Checkpoint Hashes

- **Stage 2.1 Checkpoint:** `a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c`
  - Path: `checkpoints_conditional/checkpoint_final.pt`
  - Size: 139,453,045 bytes
  - Status: READ-ONLY

- **Stage 3.2.8 Spatial Checkpoint:** `a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c`
  - Path: `experiments_stage3_2_7/checkpoints/checkpoint_stage2_spatial.pt`
  - Size: 139,453,045 bytes
  - Status: READ-ONLY

- **Stage 3.2.8 Motion Checkpoint:** `9a75aadc6bae18b9083797fa4bf2424f58fdf73118eac620e259e092096fa28a`
  - Path: `experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt`
  - Size: 139,532,880 bytes
  - Status: READ-ONLY

- **Tokenizer:** `9227a1001b4d3503fa59729c04a26a01307d4e0a050b579e858ee8a809d679e8`
  - Path: `experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json`
  - Size: 4,128 bytes
  - Status: READ-ONLY

### 1.2 Hard Assertions

**Before Training:**
- No frozen spatial parameter appears in an optimizer parameter group
- No frozen appearance embedding appears in an optimizer parameter group
- Temporal-disabled output matches Stage 2.1 reference within Stage 3.2.9A tolerance
- Temporal-enabled initialization is close to temporal-disabled output
- Gradients reach intended temporal and motion parameters
- Frozen spatial and appearance parameters receive no gradients

**After Training:**
- Spatial parameters remain byte-for-byte unchanged
- Appearance embeddings remain unchanged
- Original checkpoint hashes remain identical
- Every experiment records exact trainable parameter names and count

**Fail immediately if any assertion is violated.**

---

## 2. Experiment Variants

### 2.1 Experiment A — Spatial-Only Baseline

**Configuration:**
- Architecture: Original Stage 2.1 ConditionalUNet
- Processing: Frame-independent (no temporal modules)
- Checkpoint: Verified Stage 2.1 checkpoint
- Training: None (no parameter updates)
- Purpose: Reference baseline for comparison

**Key Characteristics:**
- Processes each video frame independently
- No temporal convolutions
- No temporal fusion
- No motion embeddings
- All parameters frozen

**Expected Behavior:**
- Generates coherent spatial appearance
- No temporal coherence across frames
- Motion only from random noise

---

### 2.2 Experiment B — Motion-Conditioning Only

**Configuration:**
- Architecture: Stage 3 VideoUNetFixed
- Temporal convolutions: DISABLED
- Temporal fusion: DISABLED
- Trainable parameters: 40 motion-token embeddings only
- Frozen parameters: Spatial backbone, appearance embeddings
- Purpose: Isolate motion language learning without temporal processing

**Key Characteristics:**
- Split embedding architecture (appearance tokens 0-41, motion tokens 42-81)
- Only motion embeddings trainable (40 tokens × 256 dims = 10,240 parameters)
- Temporal modules defined but not executed
- Frame-independent processing with motion conditioning

**Expected Behavior:**
- May learn motion-related language representations
- No temporal coherence (no temporal processing)
- Potential improvement in motion understanding vs baseline

---

### 2.3 Experiment C — Motion Conditioning + Temporal Learning

**Configuration:**
- Architecture: Stage 3 VideoUNetFixed
- Temporal convolutions: ENABLED
- Temporal fusion: ENABLED
- Trainable parameters:
  - Motion-token embeddings (10,240 parameters)
  - Temporal convolutions (124,032 parameters)
  - Fusion gates (2 parameters)
  - Total: 134,274 parameters
- Frozen parameters: Spatial backbone, appearance embeddings, transformer
- Purpose: Primary experiment - full temporal learning

**Key Characteristics:**
- Complete temporal architecture
- Identity-friendly temporal initialization
- Near-zero initial fusion gates (0.1 → sigmoid ≈ 0.525)
- Feature-level temporal fusion at bottleneck and low-res

**Expected Behavior:**
- Learn motion-related language representations
- Learn temporal coherence across frames
- Learn prompt-controlled motion
- Preserve spatial appearance quality

---

## 3. Dataset Configuration

### 3.1 Dataset Specifications

**Source:** `experiments_stage3_2_7/data_synthetic/`

**Configuration:**
- Resolution: 64×64
- Frames per clip: 16
- Output: RGB
- Motion categories: 6 types
  - top_to_bottom
  - bottom_to_top
  - left_to_right
  - right_to_left
  - diagonal
  - stationary

**Splits:**
- Train: 800 clips
- Validation: 160 clips
- Test: 160 clips

### 3.2 Metadata Fields

Each clip includes:
- `id`: Unique identifier
- `motion_type`: Motion category
- `shape`: Object shape (circle, triangle, square)
- `color`: Object color
- `background`: Background color
- `size`: Object size (small, medium, large)
- `velocity`: Motion speed (slow, medium, fast)
- `caption`: Text description
- `video_path`: Path to video tensor
- `trajectory_path`: Path to trajectory tensor

### 3.3 Verification Requirements

**Before Training:**
- No duplicate clips across splits
- No trajectory-seed leakage between splits
- Motion categories represented in each split
- Stationary examples included as negative controls
- Sufficient diversity in directions, speeds, shapes, colors, backgrounds

**Usage:**
- Training data: Optimization
- Validation data: Model selection
- Test data: Final evaluation only

---

## 4. Training Protocol

### 4.1 Pilot Training (100 Steps)

**Purpose:** Initial exploration of learning trends

**Configuration:**
- Steps: 100
- Batch size: 2 (CPU-optimized)
- Learning rate: 0.0002
- Optimizer: AdamW
- Weight decay: 0.01
- Random seed: 42 (for reproducibility)

**Recorded Metrics:**
- Training denoising loss
- Validation denoising loss
- Spatial-only baseline loss
- Motion embedding gradient norms
- Temporal convolution gradient norms
- Fusion gate values
- Frozen-parameter integrity
- CPU utilization
- Memory consumption

**Decision Criteria:**
- Extend to 250 steps if validation results justify continuing
- Stop and diagnose if temporal learning degrades validation performance
- Do not claim convergence from short pilot

### 4.2 Extended Training (250 Steps)

**Conditions:**
- Only when pilot shows promising trends
- Validation results must justify continuation
- Same configuration as pilot

**Termination Conditions:**
- If temporal learning consistently degrades validation performance
- If overfitting observed
- If frozen-parameter integrity violated

---

## 5. Evaluation Metrics

### 5.1 Denoising Quality

**Metrics:**
- Predicted-noise MSE
- Reconstructed clean-video MSE
- Prediction-target correlation
- Difference from frame-independent baseline

**Purpose:** Measure overall reconstruction quality

### 5.2 Motion Accuracy

**Using ground-truth trajectory metadata:**

**Metrics:**
- Object-center trajectory error (average L2 distance)
- Endpoint position error (final frame position error)
- Motion direction accuracy (classification accuracy)
- Speed prediction error (speed magnitude error)
- Frame-to-frame displacement accuracy (average displacement error)

**Purpose:** Measure how well model learns motion

### 5.3 Stationary Behavior

**For stationary prompts:**

**Metrics:**
- Object-center drift (average displacement from initial position)
- Unwanted movement (magnitude of motion when none expected)
- Unnecessary frame-to-frame changes (average change magnitude)

**Purpose:** Ensure model doesn't generate unwanted motion

### 5.4 Appearance Preservation

**Metrics:**
- Color consistency (color distance across frames)
- Shape preservation (shape similarity)
- Object visibility (object detectability)
- Background stability (background change magnitude)
- Spatial quality vs Stage 2.1 (SSIM, PSNR)

**Purpose:** Ensure spatial quality not degraded

### 5.5 Temporal Behavior

**Metrics:**
- Frame-to-frame consistency (temporal smoothness)
- Unwanted flicker (high-frequency temporal changes)
- Temporal smoothness (low-frequency temporal changes)
- Intended movement preservation (motion not smoothed away)

**Purpose:** Measure temporal coherence quality

**Warning:** Do not optimize for smoothness alone. Stationary video can look smooth while failing moving-object prompts.

---

## 6. Prompt-Controlled Generation Tests

### 6.1 Test Prompts

**Direction Tests:**
- Moving left
- Moving right
- Moving upward
- Moving downward

**Speed Tests:**
- Slow movement
- Fast movement

**Complex Motion Tests:**
- Bouncing movement
- Rotating movement

**Control Tests:**
- Stationary object

### 6.2 Comparison Tests

**Direction Reversal:**
- Left vs right
- Up vs down

**Movement vs Stationary:**
- Moving vs stationary

### 6.3 Evaluation

**Check:**
- Generated trajectory agrees with prompt
- Generated trajectory matches expected ground truth
- Consistent behavior across multiple seeds

**Artifacts:**
- Save generated videos
- Save frame sequences
- Save prompts
- Save random seeds
- Save evaluation results

---

## 7. CPU-First Constraints

### 7.1 Configuration

**Hardware:**
- CPU: Intel64 Family 6 Model 140 (4 physical, 8 logical cores)
- RAM: 15.65 GB
- GPU: Not used (optional only)

**Configuration:**
- Benchmark: 8-thread configuration from previous stages
- Measure optimal thread count for this workload

### 7.2 Recorded Metrics

**Model:**
- Parameter count
- Trainable parameter count

**Performance:**
- Peak RAM usage
- Training time per step
- Inference time per frame
- Inference time per 16-frame clip
- Frames per second

**Sampling:**
- DDIM sampling steps
- Resolution
- Frame count

**Requirement:** GPU must remain optional. No mandatory GPU dependencies.

---

## 8. Acceptance Criteria

Stage 3.2.9B is successful only when:

1. **Spatial and appearance parameters remain unchanged**
   - Byte-for-byte verification
   - Hash verification

2. **Original Stage 2.1 checkpoint remains protected**
   - Not modified
   - Not overwritten

3. **Temporal training produces measurable learning**
   - Not merely executing temporal layers
   - Evidence from validation metrics

4. **Experiment C outperforms Experiment B**
   - Meaningful improvement on held-out motion-related metrics
   - Statistically significant or clearly evident

5. **Motion direction and speed are prompt-controlled**
   - Measurable influence from prompt
   - Correct direction/speed for test prompts

6. **Stationary prompts do not generate unnecessary movement**
   - Low drift for stationary objects
   - No unwanted motion

7. **Spatial appearance quality not degraded**
   - Acceptable color consistency
   - Acceptable shape preservation
   - Comparable to Stage 2.1

8. **Improvements are reproducible**
   - Across multiple validation seeds
   - Consistent behavior

9. **CPU-compatible execution**
   - All mandatory training and inference on CPU
   - No GPU dependency

**If Experiment C does not outperform Experiment B, report result honestly. Do not label stage successful merely because training completes or gates change.**

---

## 9. Deliverables

### 9.1 Scripts

1. `stage3_2_9b_experiment.py` - Training script with 3 experiment variants
2. `stage3_2_9b_evaluation.py` - Evaluation script with motion metrics
3. `test_stage3_2_9b_integrity.py` - Integrity test script

### 9.2 Reports

1. `STAGE3_2_9B_EXPERIMENT_DESIGN.md` - This document
2. `STAGE3_2_9B_TRAINING_RESULTS.md` - Training results
3. `STAGE3_2_9B_MOTION_EVALUATION.json` - Motion evaluation metrics
4. `STAGE3_2_9B_SPATIAL_INTEGRITY.json` - Spatial integrity verification
5. `STAGE3_2_9B_PERFORMANCE.json` - CPU performance metrics
6. `STAGE3_2_9B_FINAL_REPORT.md` - Final comparison report

### 9.3 Artifacts

- Model checkpoints for each experiment
- Generated sample videos
- Evaluation results
- Training logs

---

## 10. Final Report Requirements

The final report must:

1. **Compare Experiments A, B, and C**
   - Tabular comparison of all metrics
   - Statistical analysis where applicable

2. **Explain temporal module contribution**
   - Whether temporal modules provide measurable value
   - Evidence from experiments

3. **Identify remaining weaknesses**
   - Areas where performance is lacking
   - Known limitations

4. **Make recommendation for next stage**
   - Proceed to Stage 3.3?
- Scale architecture?
- Address specific weaknesses first?
- Different approach needed?

**Do not proceed to Stage 3.3 or scale architecture until Stage 3.2.9B produces credible evidence of prompt-controlled temporal learning.**

---

## 11. Objective Reminder

The objective is not simply to generate a sequence of changing frames.

**The objective is to learn coherent, controllable motion while retaining the spatial capabilities already established in Stage 2.1.**

Key requirements:
- Motion must be prompt-controlled
- Motion must be coherent (not random)
- Spatial quality must be preserved
- Temporal behavior must be meaningful (not just smooth)
