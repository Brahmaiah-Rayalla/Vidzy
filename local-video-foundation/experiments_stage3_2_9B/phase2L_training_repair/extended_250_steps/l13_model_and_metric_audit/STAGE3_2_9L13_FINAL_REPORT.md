# Stage 3.2.9L.13 — Final Report

**Date:** 2026-11-09 | **Status:** COMPLETE | **No training performed**

## Phase outcomes

- **P1 Preflight:** All checkpoints hashed; reference SHA `a7f5142e…`
  matches `checkpoint_stage2_spatial.pt` and `checkpoints_conditional/
  checkpoint_final.pt` (NOT `checkpoint_stage2_motion_expanded.pt`, though A/A2
  outputs proved spatial-equivalent). Tokenizer compatible; original vocab 42,
  motion vocab 40, split encoder frozen-transformer as designed.
- **P2 Three-way comparison:** See `L13_THREE_WAY_SPATIAL_COMPARISON.md`.
  No config produced recognizable objects; B diverges from A due to split
  text-encoder embedding shift (spatial weights identical).
- **P3/P3b Fusion forensics:** Bottleneck path empirically dead (rand×10 →
  0.0025 output diff; gate grad ~4 orders below low_res). low_res path fully
  alive. Fusion equation injects the FULL fused tensor (`h = h + fused`),
  not `gate*residual` only; interpolation form not implemented.
- **P4 Dataset audit:** 176 synthetic videos, 64×64×16f, consistent
  caption↔trajectory mapping, GT trajectories saved. Adequate for the
  intended toy task.
- **P5 Metric calibration:** On GT synthetic videos the diff-centroid and
  block-matching metrics achieve cosine ≈1.0 vs true displacement; detection
  tiers work where objects exist. On model outputs they correctly report
  UNRELIABLE — the failure is in the generations, not the metrics.
- **P6 Readiness:** See `L13_TRAINING_READINESS_RECOMMENDATION.md`.

## Final classification

# MULTIPLE ISSUES IDENTIFIED

1. **SPATIAL MODEL QUALITY ISSUE (primary):** Stage2.1 itself does not render
   clean prompted objects — abstract texture output on both simple and
   in-distribution captions, on both reference-equivalent checkpoints.
2. **VIDEO INTEGRATION ISSUE (secondary):** `TextEncoderSplit` alters text
   conditioning (mean |Δemb| 0.69) so the video model's frame-independent
   path does not reproduce the Stage2.1 image output despite identical
   spatial weights.
3. **TEMPORAL FUSION DEFECT (tertiary):** bottleneck fusion is a dead path;
   low_res gate is the only effective temporal control and injects the full
   fused tensor rather than a gated residual.

Motion supervision cannot be assessed until spatial frames contain objects.
L.11/L.12 motion-direction numbers should be treated as texture-drift
artifacts, not object-motion evidence.

## Concrete next experiment

Stage 3.2.9L.14 as specified in `L13_TRAINING_READINESS_RECOMMENDATION.md`:
conditioning/embedding audit → ~200-500-step spatial-only retrain probe →
split-encoder parity fix. Estimated CPU runtime ~1-1.5 h. Not started.

## Deliverables

`STAGE3_2_9L13_FINAL_REPORT.md`, `L13_THREE_WAY_SPATIAL_COMPARISON.md`,
`L13_SPATIAL_QUALITY_METRICS.json`, `L13_CHECKPOINT_AND_CONDITIONING_AUDIT.json`,
`L13_BOTTLENECK_FUSION_AUDIT.{json,md}`, `L13_LOWRES_FUSION_AUDIT.json`,
`L13_DATASET_GROUND_TRUTH_AUDIT.json`, `L13_DETECTOR_CALIBRATION.json`,
`L13_GLOBAL_MOTION_CALIBRATION.json`, `L13_TRAINING_READINESS_RECOMMENDATION.md`,
`L13_THREE_WAY_RAW.json`, `L13_ARTIFACT_HASHES.json`, `three_way/` outputs,
`gt_examples/` contact sheets, and scripts `l13_phase1_3.py`,
`l13_phase2_threeway.py`, `l13_phase4_5.py`.
