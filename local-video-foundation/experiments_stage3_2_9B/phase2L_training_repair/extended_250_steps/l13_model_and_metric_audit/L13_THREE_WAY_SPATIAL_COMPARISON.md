# L.13 Phase 2 — Three-Way Spatial Quality Comparison

Configs: A = Stage2.1 conditional image model (`checkpoint_stage2_motion_expanded.pt`);
A2 = Stage2.1 reference spatial (`checkpoint_stage2_spatial.pt`, SHA matches
reference `a7f5142e…`); B = C25 step-250, temporal disabled (frame-independent,
A's noise tiled ×16); C = C25 step-250, temporal enabled. 5 simple prompts,
same tokenizer/seed schedule. Raw data: `L13_THREE_WAY_RAW.json`, images in
`three_way/`.

## Results

- **A and A2 outputs are pixel-identical** → the motion-expanded checkpoint's
  spatial weights equal the reference spatial checkpoint. Spatial lineage is clean.
- **Neither A/A2 nor B/C produce recognizable prompted objects.** All outputs
  are abstract textures: corner-patch std 0.34–0.44 (flat synthetic backgrounds
  would be ~0), prompt-color fractions low and inconsistent (blue prompt → 1%
  blue pixels; green prompt → 0.7% green). A probe with actual training captions
  shows the same textured output → not a prompt-generalization artifact.
- **B diverges from A despite identical spatial weights** (frame-0 mean |diff|
  0.20–0.55). Root cause measured: `TextEncoderSplit` produces different
  embeddings than the standard `TextEncoder` for identical tokens
  (mean |Δ| = 0.69, max 3.66; mask identical). Spatial weight diff = 0.
  → The L-series integration changes text conditioning itself.
- **C adds temporal activity** (0.09–0.18 mean frame diff) on top of the same
  abstract appearance; temporal_activity = 0.0 for B confirms the bypass works.

## Decision-logic mapping

| Check | Result |
|---|---|
| A and B both abstract | YES — spatial model itself lacks object quality |
| A good, B bad | N/A (A is also abstract) |
| B good, C degraded | N/A |
| Integration degrades conditioning | YES — split text encoder measurably alters embeddings |

**Conclusion:** primary failure is upstream of video modeling — the Stage2.1
spatial generator does not render clean prompted objects even in-distribution.
Secondary integration defect: split-encoder conditioning shift. Temporal fusion
cannot be fairly evaluated until spatial quality exists.
