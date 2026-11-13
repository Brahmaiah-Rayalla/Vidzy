# L.13 Phase 6 — Training Readiness Recommendation

## Diagnosis

1. **Spatial model quality is the primary blocker.** Stage2.1 — the same
   weights used by all video configs — produces abstract textures, not
   recognizable prompted objects, even on in-distribution training captions
   and reference-checkpoint-equivalent weights. No downstream temporal work
   can be evaluated until frames contain objects.
2. **Video integration has a conditioning defect.** `TextEncoderSplit`
   changes text embeddings vs the standard encoder (mean |Δ| 0.69) while
   spatial weights are bit-identical → frame-independent video output diverges
   from the Stage2.1 image output. Worth auditing `load_from_standard_encoder`
   / split `token_emb` before more video training.
3. **Bottleneck fusion is a dead path** (see audit). Do not spend training
   compute on it; treat as experimental/broken until the injection semantics
   are corrected.
4. **Dataset is adequate** for what it teaches: 176 clean synthetic clips,
   consistent captions/trajectories, GT masks implicit in trajectories.
   It teaches exactly one object on a flat background — sufficient for a
   proof-of-motion model but explains why outputs look like textured blobs
   if spatial training is undercooked.
5. **Metrics are calibrated and trustworthy on clean data** (cosine ≈1.0 vs
   GT on synthetic); they correctly report "no object" on current outputs.
   L.11/L.12 direction results should be re-interpreted as texture-drift
   measurements, not object motion.

## Recommended next experiment (do NOT start yet)

**Stage 3.2.9L.14 — Spatial conditioning audit + spatial retrain probe**

- Hypothesis: the Stage2.1 conditional model underfit its text→image mapping
  (or the conditioning path — vocab/tokenizer/mask — is subtly broken), so
  frames never contain objects; all video metrics are downstream noise.
- Controls: fixed 5 validation prompts (training captions + simple shapes),
  fixed seeds, the calibrated L.13 metrics + human contact-sheet review.
- Steps: (a) verify text-encoder forward correctness (embedding norms,
  attention-mask handling, EOS/PAD semantics) against a trivial overfit probe;
  (b) short spatial-only retrain probe (~200-500 steps, CPU ≈ 6-7 s/step →
  ~25-60 min) on the synthetic image dataset to test whether the architecture
  CAN render a circle-on-background at all;
  (c) fix `TextEncoderSplit` embedding parity (make split encoder reproduce
  standard-encoder outputs for original tokens exactly) before any video work.
- Success criteria: generated image contains a single dominant shape whose
  color/position matches the caption (color-mask IoU > 0.3, corner std < 0.1);
  split-encoder embedding parity |Δ| < 1e-3.
- CPU estimate: ~1-1.5 h total.
