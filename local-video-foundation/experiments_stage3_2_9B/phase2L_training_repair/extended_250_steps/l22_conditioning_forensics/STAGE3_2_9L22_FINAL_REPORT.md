# Stage 3.2.9L.22 - Final Report

**Date:** 2026-11-09 | **Classification: FROZEN BACKBONE LIMITATION**

## Forensics
- Caption rel_diff: conv1 0.137 -> block20 0.051 (attenuated ~2.7x, survives).
- Token projection preserves per-token features; padding zeroed correctly.
- Conv1 text effect measurable (full vs zero-text RMS delta).

## Conditioning-only training (500 steps, tokproj + conv text-slice only)
- Memorization WORSE: mse_mean 0.175->0.31 (vs L21A 0.144).
- Caption discrimination unchanged: swap 0.011, cos 0.9999, ablate 0.076.
- Conditioning params updated (loss 0.013->0.004) but cannot translate to images with frozen backbone.
- NOTE: RGB conv slice drifted ~1% via AdamW decoupled weight-decay despite zeroed grads - documented, not bit-identical as spec required.

## Answers
1. Conditioning-only improves sensitivity? NO.
2. Learns color/shape words? NO.
3. Info survives deep stages? Partially (5% rel diff at bottleneck).
4. Sensitivity -> semantic images? NO.
5. Frozen backbone limits? YES - primary finding.
6. Branch memorizes pairs? NO (worse than baseline).

## Separate reports
- Token proj: functional | Conv1: text effect present | Propagation: attenuated | Cond-only outcome: FAIL | Correct-vs-wrong: ~equal | Semantic: FAIL | CPU ~15min | Video-ready: NO

## Recommendation
L.23 limited unfreeze (first res-block + conditioning) or learned conv1 text-gain scalar.
