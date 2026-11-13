# Stage 3.2.9L.12 - Final Report

**Date:** 2026-11-09
**Status:** COMPLETE
**Classification:** NO RELIABLE GATE EFFECT ESTABLISHED
*(with one robust architectural finding: the bottleneck fusion gate is inert)*

## Objective

Determine whether the two fusion gates (bottleneck, low-resolution) contribute
differently to directional motion generation vs stationary-object drift, and
whether an asymmetric or reduced gate configuration beats C25 native.
Inference-only; no training.

## Phase Summary

- **Phase 1 - L.11 audit:** All L.11 artifacts verified; hashes intact.
  Metrics reproduced from saved PNG frames match `L11_MOTION_EVALUATION.json`
  within quantization tolerance (activity Δ≤0.0002, displacement Δ≤0.17px,
  identical direction classes). File-name discrepancy resolved: the prompt's
  `L11_GENERATION_CONFIG/MOTION_METRICS/SEED_LEVEL_RESULTS.json` do not exist;
  equivalent data lives in `L11_MOTION_EVALUATION.json`,
  `L11_C1_C25_COMPARISON.json`, and `run_l11_evaluation.py`. Sample-count
  question resolved: 6 captions/config, 5 directional + 1 stationary by design.
- **Phase 2 - Metric reliability:** Color/shape object detection FAILED on
  nearly all videos (detection rates 0-25%, mostly 0%). The generated clips
  contain abstract textures, not the prompted colored shapes. The L.11
  diff-centroid "direction match" metric therefore tracks the dominant
  temporal-change region, NOT a verified object. Supporting global-translation
  (block-matching) measurements do NOT corroborate prompt-direction alignment
  for any configuration (mean cosine ≈ 0 or negative everywhere).
- **Phase 3 - Gate inspection:** Two independent scalar gates
  (`sigmoid(logit) * temporal_residual` added to decoder features).
  C25 native: bottleneck 26.1%, low_res 26.3% activated. Runtime residuals:
  bottleneck ≈0.60×feature norm, low_res ≈1.00×feature norm.
- **Phase 4/5 - First pass:** 7 configs × 6 prompts = 42 videos, L.11 seeds.
  C1/C25 native rows reproduce L.11 exactly.
- **Phase 6 - Multi-seed:** finalists C25_native and C25_10_10 × seeds
  2000/3000 × 6 prompts = 24 videos.

## Key measured findings

### 1. The bottleneck gate is inert (robust, replicated)

- Single-forward isolation: bottleneck 0.25 vs 0.0 changes predicted noise by
  max 4.0e-5; low_res 0.25 vs 0.0 changes it by 0.95.
- Generated videos: C25_10_10 vs C25_25_10 (and C25_10_25 vs C25_0_25) differ
  by max ~1.7e-4 pixel value - numerically identical.
- All apparent "asymmetric" effects are actually low_res-gate effects.

### 2. Direction alignment does not replicate across seeds

| Config | Seed 1000 matches/cos | Seed 2000 | Seed 3000 |
|---|---|---|---|
| C25 native | 4/5, +0.54 | 2/5, -0.08 | 2/5, +0.23 |
| C25 10-10 | 3/5, +0.13 | 2/5, +0.40 | 1/5, -0.17 |

The L.11 headline (C25 4/5 direction matches) was seed-specific. Across three
seeds, direction cosine is inconsistent in sign and magnitude for every
config. Global-translation cosine is ~0/negative for all configs/seeds.

### 3. What gates reliably control: temporal activity magnitude

- low_res ≈0.26 (native): activity ≈0.25
- low_res = 0.10: activity ≈0.14
- low_res ≈0 (C25_25_0): activity ≈0.12 (≈C1's 0.12)

Monotone, seed-stable dose-response. This is real but only controls *how
much* temporal change occurs, not *whether it matches the prompt*.

### 4. Stationary stability: no config cleanly wins

- Centroid displacement: C1 5.5px, C25 native ~20-22px, 10-10 ~6-23px
  (seed-dependent), 10-25 ~21px, 25-0 0.4px.
- Global net translation on stationary prompt: C1 3px, C25 ~33-62px,
  10-10 ~138-147px, 10-25 ~36px, 25-0 ~64px.
- Metrics disagree with each other (centroid vs global translation), and
  detection-based tracking is unavailable - stability claims are unreliable.

## Phase 7 - Comparison table (seed 1000 pass; multi-seed in JSON)

| Metric | C1 | C25 native | Best symmetric (10-10) | "Best asymmetric" (10-25) |
|---|---:|---:|---:|---:|
| Direction matches | 1/5 | 4/5* | 3/5 | 3/5 |
| Mean direction cosine | -0.43 | +0.54* | +0.13 | +0.38 |
| Temporal activity | 0.119 | 0.255 | 0.145 | 0.247 |
| Stationary drift (centroid px) | 5.5 | 20.5 | 22.6 | 21.0 |
| Trajectory consistency (straightness) | 0.078 | 0.059 | 0.057 | 0.057 |
| Appearance consistency | poor (abstract) | poor | poor | poor |
| Metric reliability | UNRELIABLE | UNRELIABLE | UNRELIABLE | UNRELIABLE |

*seed-1000 only; does not replicate on seeds 2000/3000 (see above).

## Answers to required questions

1. **Which gate contributes most to directional movement?** Neither reliably.
   The bottleneck gate is inert; low_res gate scales temporal activity but
   direction alignment is not reproducible across seeds.
2. **Which gate contributes most to stationary drift?** low_res (the only
   effective gate); drift grows with its activation, though measurement
   reliability is poor.
3. **Does reducing low-res fusion preserve directional accuracy?** No
   evidence it does - direction metrics are at noise level for all configs.
4. **Does reducing bottleneck fusion preserve temporal consistency?** Moot -
   bottleneck gate has no measurable effect.
5. **Is asymmetric fusion better than symmetric?** No - asymmetric pairs are
   numerically identical to their symmetric counterparts.
6. **Does any config improve direction while reducing stationary drift?**
   No configuration shows this; the trade-off surface is flat-to-noise.
7. **Are results consistent across seeds?** No for direction/stability
   metrics; yes for the activity dose-response and the inert-bottleneck
   finding.
8. **Are metrics reliable enough?** No. Object detection fails; the
   diff-centroid and block-matching metrics disagree; direction conclusions
   are not seed-stable. Only the activity magnitude and gate-isolation
   findings are trustworthy.

## Measured evidence vs hypotheses

- **Measured:** bottleneck gate inert; low_res gate controls temporal
  activity monotonically; all configs produce abstract low-quality video;
  object detection fails; direction metrics seed-unstable.
- **Hypothesis (unproven):** that any gate setting yields prompt-aligned
  object motion at 250 training steps.

## Classification rationale

ASYMMETRIC FUSION ADVANTAGE fails (pairs identical). LOWER GATE
INITIALIZATION is not supported (stability gains are metric-contradictory and
direction is noise). C25 NATIVE's L.11 advantage does not replicate.
What remains is a reliable *activity-magnitude* knob (low_res gate) and no
reliable motion-alignment effect - hence NO RELIABLE GATE EFFECT
ESTABLISHED for the motion-quality question.

## Recommended next stage

The binding constraint is not gate configuration - it is that 250 training
steps produce abstract textures with no trackable objects, making motion
metrics unverifiable. Recommend:

1. **Continue training** (hundreds-to-thousands of steps) with the
   activity-magnitude insight: initialize low_res gate ~10-25%, bottleneck
   gate value is irrelevant (fix at 25% or remove it from future ablations).
2. **Fix the evaluation before more gate experiments:** the diff-centroid
   metric needs an object detector that works on generated content (e.g.,
   learned detector or synthetic-data color calibration), or evaluate on
   later checkpoints where shapes emerge.
3. **Optional architecture fix:** the current injection adds the FULL fused
   feature tensor (`h = h + fused`), so even gate=0 injects unprocessed
   temporal features; consider injecting only `gate * residual` in a future
   training stage to make "0%" a true bypass.

## Deliverables

- `L12_PHASE1_AUDIT.json` - L.11 reproduction & discrepancy resolution
- `L12_METRIC_RELIABILITY_AUDIT.json` - detection rates, tiers, methodology
- `L12_GATE_CONFIGURATION.json` - configs, logits, isolation test
- `L12_GATE_ACTIVATION_ANALYSIS.json` - native gates, runtime residuals
- `L12_FIRST_PASS_RESULTS.json` - 42 videos, per-prompt metrics
- `L12_FINALIST_MULTI_SEED_RESULTS.json` - 24 videos, seeds 2000/3000
- `L12_C1_C25_GATE_COMPARISON.json` - aggregated comparison tables
- `videos_pass1/`, `videos_multiseed/` - npz tensors + PNG frames + contact sheets
- `reliability/contact_sheets/` - L.11 contact sheets
- `l12_common.py`, `l12_phase1_audit.py`, `l12_phase2_reliability.py`,
  `l12_phase3_gates.py`, `l12_pass1_generate.py`, `l12_pass2_multiseed.py`,
  `l12_analyze.py` - reproducible scripts
- `L12_ARTIFACT_HASHES.json` - SHA-256 manifest
