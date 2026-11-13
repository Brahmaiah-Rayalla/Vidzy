# Stage 3.2.9L.10 - Recovery Preflight Report

**Date:** 2026-11-07
**Status:** Phase 1 Complete

## Current Artifact Inventory

### L.7 Artifacts (gate25_training/)

**C1 (1% init):**
- checkpoint_step_000025.pt ✅
- checkpoint_step_000050.pt ✅
- checkpoint_step_000075.pt ✅
- checkpoint_step_000100.pt ✅

**C25 (25% init):**
- checkpoint_step_000025.pt ✅
- checkpoint_step_000050.pt ✅
- checkpoint_step_000075.pt ✅
- checkpoint_step_000100.pt ✅

**Results:**
- STAGE3_2_9L7_TRAINING_RESULTS.json ✅
- STAGE3_2_9L7_SMOKE_TEST_RESULTS.json ✅
- STAGE3_2_9L7_ANALYSIS.json ✅

### L.8 Artifacts (extended_250_steps/)

**C1 (1% init):**
- checkpoint_step_000150.pt ✅ (48,162,343 bytes)
- checkpoint_step_000200.pt ✅ (48,162,343 bytes)
- checkpoint_step_000250.pt ❌ NOT FOUND

**C25 (25% init):**
- checkpoint_step_000150.pt ❌ NOT FOUND
- checkpoint_step_000200.pt ❌ NOT FOUND
- checkpoint_step_000250.pt ❌ NOT FOUND

**Configuration:**
- run_extended_training.py ✅
- STAGE3_2_9L8_EXPERIMENT_DESIGN.md ✅
- STATUS_REPORT.md ✅

**Results:**
- STAGE3_2_9L8_TRAINING_RESULTS.json ❌ NOT FOUND

### L.9 Artifacts (extended_250_steps/)

**Analysis:**
- L8_EXECUTION_STATUS.md ✅
- L9_CHECKPOINT_INTEGRITY.json ✅
- L9_VALIDATION_LOSS_COMPARISON.json ✅
- L9_GATE_PROGRESSION.json ✅
- L9_TEMPORAL_ACTIVATION_ANALYSIS.json ✅
- L9_MOTION_EVALUATION.json ✅
- L9_C1_C25_COMPARISON.json ✅
- STAGE3_2_9L9_FINAL_REPORT.md ✅

## Training Status Assessment

### C1 Status
- Latest checkpoint: Step 200
- Training appeared to complete to step 200
- No evidence of progress beyond step 200
- Likely stopped after step 200 due to script termination or system interruption

### C25 Status
- Latest checkpoint: Step 100 (from L.7)
- No L.8 checkpoints found
- L.8 training did not produce any C25 checkpoints
- Likely C25 training never started or failed before first checkpoint save

### Missing Data
- STAGE3_2_9L8_TRAINING_RESULTS.json - not found
- No training logs or CSV files found
- No evidence of running processes (checked, none active)

## L.8 Script Analysis

**Script:** run_extended_training.py
**Purpose:** Continue C1 and C25 from step 100 to 250
**Configuration:**
- Resume from L.7 step-100 checkpoints
- Restore optimizer states
- Save checkpoints at steps 150, 200, 250
- Use seed 42, lr=0.0002, wd=0.01

**Why C25 May Have Failed:**
1. Script may have crashed after C1 step 200
2. C25 training may have encountered an error before first checkpoint
3. System interruption during C25 training
4. Resource exhaustion

## Conclusion

L.8 training is incomplete:
- C1 reached step 200 (partial success)
- C25 did not progress beyond step 100 (failed)
- No training results JSON available
- Need to recover C25 from step 100 using L.7 checkpoint
