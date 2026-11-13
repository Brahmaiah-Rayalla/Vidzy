# Stage 3.2.9L.24 - Final Report

**Date:** 2026-11-10 | **Classification: UNCONDITIONAL DENOISING DOMINATES**

## The decisive experiment
10,000 steps, batch 4, full spatial UNet trainable, direct text channels (512ch via token_proj+expanded conv1), frozen TE, cross-attn bypassed, eps-MSE objective, 700-image dataset (~57 passes, 40k presentations), 8.78h CPU.

## Results
- val_loss: 0.0123 -> 0.0032 (monotonic improvement)
- Diversity: 0.266 -> 0.071 (collapse, same as L16)
- Caption sensitivity at s10000: swap cos 0.9999, mean_abs 0.001-0.006
- Zero-text ablation: 0.003 - text channel contribution SHRANK during training
- The model actively unlearned conditioning: full UNet training found unconditional denoising optimal

## Conclusion
This is the definitive answer to the conditioning question. EVERY conditioning variant fails identically because the eps-denoising objective on this synthetic dataset does not reward caption use - the gradient signal for prompt alignment is too weak relative to unconditional denoising at any tested scale (500 to 10k steps). It is NOT an architecture problem, NOT a conditioning-path problem, NOT a mask/encoder problem.

## Separate reports
- Full training: COMPLETE 10k steps | Presentations: 40k | TE integrity: frozen | Caption sens: ~0 (unlearned) | Semantic: FAIL | Held-out: FAIL | CPU: 8.78h | Video-ready: NO

## Recommendation
L.25: decision point - abandon prompt-conditioning on this architecture (document ceiling) OR try fundamentally different objective. No more eps-loss probes will resolve this.
