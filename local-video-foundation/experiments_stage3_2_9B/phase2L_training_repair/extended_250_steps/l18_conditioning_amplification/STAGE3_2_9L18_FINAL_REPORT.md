# Stage 3.2.9L.18 - Final Report

**Date:** 2026-11-09 | **Classification: TEXT PROJECTION OR RESIDUAL SCALING DEFECT**

## Phase results
- P2 Mask audit: PASS - mask==0 -> -inf pre-softmax; padding receives exactly 0 attention (unit test). No mask defect.
- P3 Activation: text signal ARRIVES - attn_out RMS 0.13-0.29 vs h_in 0.6-0.7 across ~10 cross-attn blocks; text_emb RMS ~0.99.
- P4/P7 Gradient: cross-attn grads weak but present.
- P5 Gain sweep: sensitivity scales linearly but stays negligible - G4.0 gives mean|d|=0.017, cosine 0.99975. Even 4x amplification cannot restore caption sensitivity.
- P7 Arms: A (standard) and B (binding loss, lambda 0.1, margin 0.02) both ~750 steps bs1 on L17 10-pair subset. Best mse_mean: A 0.221 (s500), B 0.181 (s250) - no meaningful advantage; B regressed by s750. Binding term saturated at margin (lc~ls) throughout = correct and swapped captions yield identical denoising error.

## Answers
1. Mask exposes valid tokens? YES.
2. Projection magnitude sufficient? Signal arrives but content-invariant.
3. Attention depends on caption content? NO - near-uniform attention washes out token identity.
4. Cross-attn contributes enough? Magnitude OK; CONTENT is the problem.
5. Binding objective improves correct-caption generation? NO.
6. Merely increases sensitivity? It did not even do that.
7. Standard control differ? No.
8. Bottleneck: cross-attention softmax collapses to near-uniform distribution -> output is caption-invariant average. Conditioning is architecturally present but informationally dead.

## Separate reports
- Mask audit: PASS | Activation: signal present | Gain: insufficient | Arm A: no binding | Arm B: no improvement | Visual generation: FAIL | Video-ready: NO

## Recommendation
L.19 attention-weight forensics (entropy/top-token per head) then architectural conditioning fix if uniform attention confirmed.
