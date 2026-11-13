# Stage 3.2.9L.20 - Final Report

**Date:** 2026-11-09 | **Classification: BOTH ARCHITECTURES FAIL TO BIND CAPTIONS**

## Results
- Arm A (learned-query + FiLM): post-training attention entropy 0.9997 - queries NEVER became selective. Predicted-noise caption swap cos 0.99998 unchanged. Best mse_mean 0.34.
- Arm B (pooled FiLM): best mse_mean 0.24 (s500) but regressed; no caption sensitivity. Simpler but equally ineffective.
- Both worse than L17 baseline (0.13) on memorization - FiLM path may even dilute learning.
- FiLM heads zero-init preserved behavior at start (identity), but 750 steps produced no measurable conditioning.

## Answers
1. Learned-query selective? NO (entropy ~1.0 throughout).
2. Queries attend differently to shape/color? NO.
3. Learned-query > pooled-FiLM? No measurable difference.
4. Either binds captions? NO.
5. Improved semantic generation? NO.
6. Selectivity -> visual improvement? N/A (no selectivity).
7. Memorize 10 images? Partially, worse than baseline.
8. Generalize? Untested - memorization itself failed.
9. Best capability/CPU trade-off? Neither; architecture is not the bottleneck.

## Deeper finding
Even fresh learned queries with direct text access stayed uniform under eps-denoising. The conditioning failure is therefore NOT fixable by attention architecture alone - the training objective/data scale does not reward caption use. Next stage must test whether ANY conditioning path can drive generation (direct text-channel injection existence proof) before more architecture work.

## Separate reports
- Learned-query selectivity: FAIL | Representation quality: N/A | FiLM effectiveness: none measured | Spatial response: none | Pred-noise sensitivity: ~0 | Semantic quality: FAIL | Overfit: partial/worse | Validation: n/a | CPU: ~25min | Video-ready: NO

## Recommendation
L.21: direct-conditioning existence proof (text channels injected into UNet input) + FiLM capacity probe.
