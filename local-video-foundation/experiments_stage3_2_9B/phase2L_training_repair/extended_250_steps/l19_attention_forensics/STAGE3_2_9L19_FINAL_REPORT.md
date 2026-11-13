# Stage 3.2.9L.19 - Final Report

**Date:** 2026-11-09 | **Classification: UNIFORM ATTENTION CONFIRMED**

## Results
- All 21 cross-attn blocks, 8 captions: normalized entropy 0.989-0.992 (uniform=1.0), effective tokens ~13.8/16, top-1 prob ~0.098.
- No head is selective; no caption produces a distinct attention map (weight cosine ~1, JS div ~0).
- Token identity audit: top-attended tokens are arbitrary; color/shape words get no preference.
- KV forensics: k/v representations healthy (rms 0.39/0.45); context differs only 0.023 mean-abs across captions.
- Interventions: single-token attention REDUCES sensitivity; uniform/temp sweeps cannot restore it. Native swap cosine 0.99998.
- Root cause: Q.K logit variance too small -> softmax ~uniform -> context = caption-invariant mean of values. Denoising loss provides no gradient pressure for selectivity.

## Separate reports
- Entropy: UNIFORM | Token identity: no semantic selectivity | KV: healthy | Context: caption-invariant | Interventions: confirm | Architecture: recommend learned-query cross-attn (Option B) with FiLM fallback | Spatial generation: NOT READY | Video: NOT READY

## Recommendation
L.20: implement learned-query conditioning probe on the 10-image overfit subset; gate success on attention entropy <0.8 and caption JS-divergence before image metrics.
