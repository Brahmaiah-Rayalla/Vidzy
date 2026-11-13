# Stage 3.2.9L.17 - Final Report

**Date:** 2026-11-09 | **Classification: TEXT CONDITIONING DEFECT**

## Diagnostic results
1. **Objective/sampler audit: CONSISTENT.** eps-prediction both sides; DDIM update inverts forward process exactly (max err <1e-5). No mismatch.
2. **Caption sensitivity: ~ZERO.** Swapping captions on fixed noise/timestep changes predicted noise by mean|d|=0.008-0.017, cosine 0.9998-0.9999 across timesteps. The U-Net essentially ignores text.
3. **Cross-attention gradients: present but weak** (mean 4.3e-4 vs overall 2.2e-3).
4. **10-image overfit: PARTIAL.** MSE-to-GT 0.41->0.13 (s750), some pairs near-memorized (min 0.02), regressed to 0.27 at s1000; diversity maintained (pairwise ~0.41-0.44). Denoiser can learn images; binding to captions does not emerge.
5. **L.16 reinspection:** loss/visual decoupling confirmed; step1000 partial collapse, step2000 desaturated.

## Answers
1. U-Net sensitive to text? NO (cosine 0.9998).
2. Caption change -> predicted noise change? Negligible.
3. Caption change -> visual difference? No (consequence of #2).
4. Cross-attn grads meaningful? Weak.
5. Memorize 10 pairs? Partially (image content yes, caption binding no).
6. Visual quality improve? No.
7. Sampler faithful? YES.
8. Remaining bottleneck: TEXT CONDITIONING PATH - cross-attention contributes ~nothing; explains all prior abstract outputs and val_loss decoupling.

## Separate reports
- Objective audit: PASS | Caption sensitivity: FAIL(~0) | Cross-attn training: WEAK | Overfit: PARTIAL | Visual generation: FAIL | Video-ready: NO

## Recommendation
L.18 conditioning-amplification diagnostic (see L17_NEXT_STAGE_RECOMMENDATION.md) - inspect text_proj magnitude, verify attention-mask semantics, add conditioning-strength probe.
