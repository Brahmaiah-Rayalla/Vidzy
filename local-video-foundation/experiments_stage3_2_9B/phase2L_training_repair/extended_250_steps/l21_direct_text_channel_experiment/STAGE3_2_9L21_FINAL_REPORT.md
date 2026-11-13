# Stage 3.2.9L.21 - Final Report

**Date:** 2026-11-09 | **Classification: CONDITIONING SENSITIVITY ONLY**

## Results
- Direct text channels causally affect predictions: zero-ablation delta 0.062-0.078 mean|d| (vs ~0.005 attention) - 12-15x stronger.
- Caption-to-caption discrimination still weak: swap delta 0.009-0.015, cosine 0.9999+.
- Arm A (standard eps): best mse_mean 0.144 (s750) ~ L17 0.13.
- Arm B (binding loss): worse (0.20-0.53) - ranking objective failed again (2nd time).
- Text channel content IS delivered (ablation proves it); UNet cannot yet convert caption differences into image differences.

## Answers
1. Predictions depend on text? YES via direct channels.
2. Standard eps learns dependency? Partially - memorization yes, caption binding weak.
3. Binding objective improves? NO.
4. Ablation removes improvement? Text contributes; caption-discrimination is the gap.
5. Memorize 10 pairs? Partially (min mse 0.018).
6. Color/shape changes -> visual changes? Not demonstrated.
7. Stable across seeds? Untested.
8. Beats L17/L20? Arm A ~equal L17, better L20.

## Bottleneck
Pathway FIXED. Remaining gap: caption-discriminative REPRESENTATION - UNet receives text but cannot map token differences to image differences.

## Separate reports
- Direct channel functional: YES | Standard: partial | Binding: FAIL x2 | Memorization: partial | Visual: partial | Ablation: confirms | CPU ~50min | Video-ready: NO

## Recommendation
L.22 conv1 text-branch forensics + conditioning-params-only probe.
