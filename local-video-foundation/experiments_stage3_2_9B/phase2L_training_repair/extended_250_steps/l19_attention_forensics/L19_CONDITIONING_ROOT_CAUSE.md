# L.19 - Conditioning Root Cause

## Verdict: UNIFORM ATTENTION CONFIRMED

Measured across all 21 cross-attention blocks and 8 captions:

| Metric | Observed | Uniform reference (N=16) |
|---|---:|---:|
| Normalized entropy | 0.989-0.992 | 1.000 |
| Mean top-1 prob | ~0.098 | 0.0625 |
| Effective attended tokens | ~13.8 / 16 | 16 |

Attention is within ~1% of uniform on every head and block. Top-attended
tokens per head show no consistent semantic pattern (color/shape words get
no preferential attention; top-3 probs ~0.07 ~ uniform share).

## Mechanism

scores = Q.K/sqrt(d). For softmax to be selective, Q.K must vary enough
across the ~16 valid tokens. With Q from spatial features and K from
text_proj(token_emb), the logit variance is evidently too small after only
~12k training presentations - the model never learned discriminative
token addressing. Uniform weights => context = mean of value vectors =>
caption-invariant conditioning. KV representations themselves are healthy
(k_rms 0.39, v_rms 0.45, context differs 0.023 across captions).

Interventions confirm: single-token attention gives even LESS caption
sensitivity (0.002-0.004 vs native 0.005); temperature scaling cannot fix
a fundamentally flat logit landscape.

## Why prior stages failed

The denoising loss is minimized by unconditional prediction; there is no
gradient pressure forcing attention selectivity. Caption-binding loss
(L18 Arm B) stayed at margin floor because correct vs swapped captions
produced identical predictions - the pathway cannot express the difference.
