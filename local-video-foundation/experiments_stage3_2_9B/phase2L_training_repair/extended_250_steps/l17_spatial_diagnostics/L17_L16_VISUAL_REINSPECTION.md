# L.17 — L.16 Visual Reinspection

L.16 generated 12 prompts × 6 checkpoints (step0/250/500/1000/1500/2000),
saved under `l16_extended_spatial_training/images/`. Reinspection via
quantitative image statistics (pixel-level visual review not possible in
this environment — noted as limitation):

## Findings across checkpoints

| Checkpoint | Pairwise diversity | Corner std | Color fidelity (simple prompts) |
|---|---:|---:|---|
| step0 | 0.152 | 0.30–0.44 | low color_frac, textured bg |
| step250 | 0.123 | 0.30–0.41 | similar |
| step500 | 0.158 | 0.38–0.44 | similar |
| step1000 | **0.037** | 0.37–0.47 | yellow_triangle 95.8% yellow = washed-field artifact |
| step1500 | 0.164 | 0.15–0.37 | partial recovery |
| step2000 | 0.089 | 0.11–0.29 | color_frac <0.011 all prompts — desaturated |

## Interpretation

- No checkpoint produced a compact, prompt-colored geometric object on a
  plain background. Backgrounds stayed high-frequency textured (corner_std
  0.3+) until late training, when outputs became smoother but desaturated.
- Step1000 shows a partial mode-collapse event (near-identical outputs);
  step1500 recovered diversity; step2000 declined again while val_loss hit
  its minimum (0.0104) — confirming loss/visual decoupling.
- Metric-vs-visual discrepancy: val_loss improved monotonically while
  generated images degraded — diffusion loss on this data does not track
  object recognizability.

Contact sheet note: per-checkpoint PNGs are preserved in
`l16_extended_spatial_training/images/` for manual review.
