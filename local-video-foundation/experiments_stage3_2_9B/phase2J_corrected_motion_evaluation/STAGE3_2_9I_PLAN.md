# Stage 3.2.9I - Motion Evaluation Plan

**Date:** 2026-11-03
**Status**: In Progress

## Completed Steps

### 1. Directory Structure
- Created `phase2J_corrected_motion_evaluation/` directory
- Copied corrected inference components from phase2I

### 2. Checkpoint Integrity
- Verified SHA256 hashes for all checkpoints
- All hashes match Stage 3.2.9H.2 baseline
- Documented in `STAGE3_2_9I_CHECKPOINT_INTEGRITY.json`

### 3. Dataset Analysis
- Analyzed training dataset from `data/controlled_video/train`
- Identified supported motion categories:
  - Moving left (red_circle_left_to_right) - 8 examples
  - Moving right (red_circle_right_to_left) - 8 examples
  - Moving diagonal (blue_square_diagonal) - 8 examples
  - Stationary (yellow_circle_stationary) - 8 examples
- Documented in `STAGE3_2_9I_PROMPT_MAPPING.json`

### 4. Inference Path Validation
- All 9 validation checks passed:
  - Tokenizer loaded: PASS
  - Text encoder loaded: PASS
  - encode_prompt method exists: PASS
  - Text encoder in eval mode: PASS
  - Text encoding deterministic: PASS
  - Seed-independent encoding: PASS
  - Embedding metadata recorded: PASS
  - Temporal control available: PASS
  - No random embeddings: PASS
- Documented in `STAGE3_2_9I_CORRECTED_CONDITIONING_VALIDATION.json`

## Next Steps

### 5. Create Evaluation Script
Create `run_motion_evaluation.py` that:
- Loads B, C, and D configurations
- Uses matched prompts and seeds (42, 123, 456)
- Generates videos with corrected text conditioning
- Reuses identical initial noise across configurations
- Records comprehensive metadata

### 6. Motion Evaluation
Use the validated object-tracking evaluator to measure:
- Detection success rate
- Direction accuracy
- Motion magnitude
- Stationary behavior
- Temporal consistency

### 7. Generate Deliverables
Create the required JSON reports:
- STAGE3_2_9I_MOTION_METRICS.json
- STAGE3_2_9I_DIRECTION_ACCURACY.json
- STAGE3_2_9I_STATIONARY_CONTROL.json
- STAGE3_2_9I_TEMPORAL_ABLATION.json
- STAGE3_2_9I_CPU_PERFORMANCE.json
- STAGE3_2_9I_GENERATION_COMPARISON.md
- STAGE3_2_9I_FINAL_REPORT.md

### 8. Final Classification
Classify results as:
- PROMISING
- PARTIAL
- TEMPORAL CONTRIBUTION NOT DEMONSTRATED
- STATIONARY CONTROL FAILURE
- INCONCLUSIVE

## Current Status
✅ Directory structure created
✅ Checkpoint integrity verified
✅ Dataset analyzed and prompt mapping created
✅ Inference path validated
⏳ Evaluation script creation (in progress)
⏳ Video generation
⏳ Motion evaluation
⏳ Report generation
