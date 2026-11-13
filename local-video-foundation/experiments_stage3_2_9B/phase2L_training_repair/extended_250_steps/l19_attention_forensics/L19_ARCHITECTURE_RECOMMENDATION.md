# L.19 - Architecture Recommendation

Recommended: Option B - Learned-query cross-attention, with Option A
(pooled caption conditioning via FiLM) as fallback.

Rationale: the failure is not mask semantics, not signal magnitude, and
not training budget alone - it is that spatial-position queries produce
near-identical attention over all tokens. Learned global queries
(conditioning tokens that attend to text, then inject a pooled context)
decouple what-to-look-for from spatial position and are far more likely
to develop selective attention under weak supervision. FiLM/pooled
conditioning (Option A) is simpler and CPU-cheaper but compresses the
caption to one vector - acceptable for shape/color/position attributes in
this synthetic domain.

Implementation sketch (next stage, do NOT implement now):
- Add N=8-16 learned query vectors; cross-attend them to text tokens
  (mask-respecting); pool outputs; project to per-block scale/shift or
  additive residual fused into spatial features.
- ~small param cost (<0.5M), CPU-cheap.
- Keep frozen text encoder and tokenizer unchanged.
- Validate first with the L19 entropy harness: require norm_entropy < 0.8
  and caption-JS-divergence > 0 before any image-quality claims.
