# Stage 3.2.9L.16 - Final Report

**Date:** 2026-11-09 | **Classification: OPTIMIZATION OR MODE COLLAPSE**

## Results
- 2000 steps bs4 lr2e-4 UNet-only, frozen TE (bit-identical verified every ckpt), 8000 presentations, 98.6min CPU.
- val_loss improved steadily: 0.0216->0.0104 BUT visuals degraded - loss/visual decoupling.
- Diversity: pairwise diff 0.152->0.089 (mid-run near-collapse 0.037 at s1000).
- Color/prompt fidelity COLLAPSED by s2000: all simple-prompt color fractions <0.011; s1000 showed washed-yellow-field artifact.
- No recognizable objects at ANY checkpoint (0/6 eval points).
- Cumulative spatial exposure ~12000 presentations - still no objects.

## Interpretation
Conditioning verified (parity 0.999), optimization stable (no NaN, TE frozen), data verified, yet 12k presentations yield abstract/desaturated output with declining diversity. Either the architecture/objective cannot bind text->image at this scale, or conditioning signal is too weak. val_loss improvement without visual gain suggests the model learns dataset-average denoising, not prompt alignment.

## Separate reports
- Stage2.1 history: 1000 steps bs4 (~4000 presentations)
- L.16 warm-start: YES (fresh AdamW)
- Total presentations: ~12000 cumulative
- TE freeze: VERIFIED bit-identical
- Visual recognition: FAILED all checkpoints
- CPU time: 98.6 min
- Ready for video: NO

## Recommendation
L.17 diagnostic stage (see L16_NEXT_STAGE_RECOMMENDATION.md): visual PNG inspection, caption-swap conditioning test, 10-image overfit probe, possibly joint training at original bs4 recipe.
