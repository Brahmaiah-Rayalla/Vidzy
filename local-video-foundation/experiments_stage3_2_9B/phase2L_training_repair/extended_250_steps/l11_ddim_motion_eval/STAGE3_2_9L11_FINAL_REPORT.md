# Stage 3.2.9L.11 - Final Report

**Date:** 2026-11-09
**Status:** COMPLETE
**Classification:** MOTION ALIGNMENT IMPROVEMENT (25% gate init), WITH CAVEATS

## Objective

Determine whether 25% fusion-gate initialization (C25) improves
prompt-controlled video motion versus 1% initialization (C1), using matched
DDIM generation from the step-250 checkpoints recovered in L.10.

## Method

- **Checkpoints:** `recovery_l10/C1/checkpoint_step_000250.pt` (gates ≈ 1.04%)
  and `recovery_l10/C25/checkpoint_step_000250.pt` (gates ≈ 26.1%).
- **Pipeline:** Real tokenizer -> `video_unet.text_encoder` ->
  `VideoUNetFixed(x_t, t, text_emb, text_mask)`; DDIM sampler identical to the
  Stage 3.2.9E implementation (linear betas 0.0001->0.02, 1000 timesteps,
  25 sampling steps = 24 deterministic updates).
- **Matching:** identical seeded initial noise per prompt across both models;
  6 in-distribution training captions covering all motion types
  (left_to_right, right_to_left, top_to_bottom, bottom_to_top, diagonal,
  stationary).
- **Metrics:** temporal activity (mean |frame diff|), changed-region weighted
  centroid trajectory, displacement, path length, straightness, and cosine
  similarity between measured displacement and expected motion direction.

## Results

### Direction alignment (5 directional prompts)

| Motion type | C1 cosine | C1 class | C25 cosine | C25 class |
|---|---|---|---|---|
| left_to_right | -0.31 | MISMATCH | -0.69 | MISMATCH |
| right_to_left | -0.93 | MISMATCH | +0.99 | MATCH |
| top_to_bottom | -0.87 | MISMATCH | +0.68 | MATCH |
| bottom_to_top | +0.76 | MATCH | +0.99 | MATCH |
| diagonal | -0.78 | MISMATCH | +0.76 | MATCH |

- **C1:** 1/5 direction matches, mean cosine **-0.43**
- **C25:** 4/5 direction matches, mean cosine **+0.54**

### Temporal activity (mean |consecutive frame diff|)

- **C1:** 0.121
- **C25:** 0.265 (**2.2x** C1)

C25 produces substantially more temporal change, and that change is aligned
with the prompted direction in 4 of 5 directional cases. C1's motion is both
weaker and essentially uncorrelated with the prompt direction.

### Stationary control

- C1 stationary displacement: 5.5 px, temporal activity 0.131
- C25 stationary displacement: 20.5 px, temporal activity 0.313

C25 over-activates temporal change even for the stationary prompt - the
higher fusion gates inject more temporal signal indiscriminately. This is a
real cost: improved directional alignment but reduced ability to hold a
scene still.

## Caveats

1. **Absolute generation quality is poor for both models.** After only 250
   training steps the outputs are abstract textured fields, not recognizable
   shapes/objects (verified numerically: ~230-310 unique colors/frame, no
   dominant object color matching prompts). All conclusions are *relative*
   C1-vs-C25 comparisons under identical noise/conditioning.
2. Trajectory straightness is low for both (0.02-0.13), meaning centroids
   wander; direction metrics are noisy per-prompt. The aggregate 4/5 vs 1/5
   match pattern is the robust signal.
3. C25's stationary-control regression suggests gate activation trades off
   controllability vs stability; a moderate init (e.g., 10%) or per-gate
   init may be worth testing.

## Classification Rationale

Prior stages (L.3-L.10) established a 17-30% denoising-loss advantage for
C25 but classified results as "DENOISING IMPROVEMENT ONLY" because motion
evaluation was deferred. L.11 supplies the missing evidence: with matched
noise and prompts, C25's generated temporal evolution aligns with the
prompted direction far more often than C1 (4/5 vs 1/5, mean cosine +0.54 vs
-0.43) with 2.2x temporal activity. This is a **motion-control improvement**,
not merely denoising - the first positive motion result in the 9L series.

## Recommendation

- **Adopt 25% fusion-gate initialization as the new baseline** for future
  temporal training stages.
- Investigate the stationary-control regression: consider 10% init ablation,
  asymmetric bottleneck/low-res init, or gate regularization toward zero for
  stationary captions.
- Extend training well beyond 250 steps before judging absolute quality;
  re-run this L.11 evaluation harness at later checkpoints (script is
  checkpoint-agnostic - point CKPT_C1/CKPT_C25 at new files).

## Deliverables

- `L11_DDIM_INFRASTRUCTURE_AUDIT.md` - Phase 2 audit
- `L11_INFERENCE_SMOKE_TEST.json` - Phase 3 smoke test (PASSED)
- `L11_MOTION_EVALUATION.json` - Phase 4/5 per-video metrics + diagnostics
- `L11_C1_C25_COMPARISON.json` - Phase 6 matched comparison + summary
- `videos/{C1,C25}/<motion_type>/frame_*.png` - 192 generated frames for audit
- `ARTIFACT_HASHES.json` - SHA-256 of all L.11 artifacts
- `run_l11_evaluation.py` - reproducible evaluation script
