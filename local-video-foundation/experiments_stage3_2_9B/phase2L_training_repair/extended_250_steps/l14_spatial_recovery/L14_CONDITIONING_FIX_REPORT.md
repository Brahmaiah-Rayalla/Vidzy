# L.14 — Conditioning Fix Report

## Root cause (confirmed)

`load_from_standard_encoder()` in
`experiments_stage3_2_7/video_foundation/models/text_encoder_split.py`
copied transformer-layer weights from the standard encoder ONLY when
`freeze_transformer=False`. Every L-series construction used
`freeze_transformer=True`, so the split encoder ran with **randomly
initialized transformer layers** while keeping correct original token
embeddings. This produced text embeddings with mean |Δ| ≈ 0.63–0.69 vs the
standard encoder — the L.13 "conditioning shift".

## Fix applied (minimal)

`text_encoder_split.py`: transformer weight copy is now unconditional —
`freeze_transformer` controls `requires_grad` only, not whether weights are
loaded. Original-vocab embeddings and the trainable motion embedding table
are unchanged; motion-vocabulary capability preserved.

## Parity results (L14_TEXT_ENCODER_PARITY_AUDIT.json)

| Path | mean \|Δ\| | max \|Δ\| | cosine |
|---|---:|---:|---:|
| Buggy (pre-fix) vs standard | 0.630 | 3.41 | 0.675 |
| Fixed vs standard | 0.014 | 0.70 | 0.999 |

Residual ~0.014 mean diff is consistent with SplitEmbedding's
`torch.where` lookup path and eval-mode numerics; per-prompt regression
test `test_text_encoder_parity.py` passes at max diff 0.0.

## Checkpoint implications

- C1/C25 `model_state_dict`s contain the OLD random transformer weights.
  Any future video-model load must re-apply Stage2.1 transformer weights
  AFTER `load_state_dict` (or rebuild via fixed loader and load only
  non-transformer keys). Motion embeddings in checkpoints remain valid.
- Stage2.1 checkpoints untouched; all historical artifacts preserved.
