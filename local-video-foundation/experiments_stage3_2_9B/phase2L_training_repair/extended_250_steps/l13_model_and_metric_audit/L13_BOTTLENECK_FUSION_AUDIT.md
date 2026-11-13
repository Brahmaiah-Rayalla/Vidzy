# L.13 Phase 3 — Bottleneck Fusion Forensic Audit

## Implementation (verified)

`video_unet_fixed.py`: two independent scalar gates
`fusion_modules.{bottleneck,low_res}.gate` (raw logit → sigmoid).
`_forward_with_temporal` computes per-level temporal residuals via
`TemporalConv1DBlock`, forms `fused = temporal + gate*residual`, and passes
`injected_features` into a second `spatial_unet` call. In
`ConditionalUNet.forward`, `h = h + injected_features[name]` — the **full
fused tensor** is added (not just `gate*residual`), so even gate≈0 injects
unprocessed temporal features. The interpolation form
`(1-g)*spatial + g*temporal` is NOT implemented.

Injection points: bottleneck after `middle_block` (128ch, 8×8);
low_res after decoder `up_blocks`, before `conv_out` (64ch, 64×64).

## Measurements (C25 step-250, fixed input/timestep/embeddings)

| Test | Result |
|---|---|
| Random×10 tensor injected at bottleneck | output maxdiff **0.0025** — path effectively dead |
| Random×10 at low_res | output maxdiff **29.9** — fully alive |
| Gate sweep bottleneck 0/0.1/0.25 (low_res fixed) | maxdiff ~2e-5 – 4e-5 |
| Gate sweep low_res 0/0.1/0.25 | maxdiff 0.38 – 5.18 |
| Gate gradients (single backward) | bottleneck **-3.7e-6** vs low_res **-0.56** (~4 orders smaller) |

## Verdict

The bottleneck injection is a **structurally dead path**: even a 10× random
tensor perturbation at the injection point moves the final prediction by only
0.0025, while the identical perturbation at low_res moves it by ~30. The
bottleneck temporal branch computes features that the decoder effectively
ignores (injection occurs where subsequent GroupNorm/attention + deep decoder
attenuate it, and/or the additive `h + fused` semantics bury the gated term
under the dominant un-gated `temporal` term). Its gradient is ~150,000×
smaller than low_res's, consistent with L-series observations that the
bottleneck gate never learns.

**Smallest correction (for a future branch, NOT applied):** inject only
`gate * residual` (not the full fused tensor) and/or relocate bottleneck
injection later in the decoder; alternatively drop the bottleneck branch.
See `L13_BOTTLENECK_FUSION_AUDIT.json`, `L13_LOWRES_FUSION_AUDIT.json`.
