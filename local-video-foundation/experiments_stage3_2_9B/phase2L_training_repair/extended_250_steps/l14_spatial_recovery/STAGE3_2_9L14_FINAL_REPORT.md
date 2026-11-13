# Stage 3.2.9L.14 - Final Report

**Date:** 2026-11-09 | **Classification: INCONCLUSIVE (conditioning resolved; spatial quality unresolved)**

## Phase results
- P1: L.13 recommendation extracted; baselines hashed; all checkpoints preserved.
- P2 PARITY: ROOT CAUSE FOUND - load_from_standard_encoder skipped transformer weight copy when freeze_transformer=True -> random transformer layers (mean|dE|=0.63). Fixed (unconditional copy). Post-fix parity: cosine 0.999, regression test passes.
- P3: Corrected-inference baseline regenerated (6 prompts x 2 seeds): STILL ABSTRACT (corner_std 0.29-0.46, no recognizable shapes). Conditioning fix alone insufficient.
- P4 PROBE: 200-step spatial retrain (unet+text_encoder, AdamW lr2e-4, bs1, 240 train/60 val imgs from synthetic dataset, ~2.6 min CPU). COLLAPSED: all outputs ~constant green regardless of prompt; val_loss unstable. Training dynamics, not evidence of architecture capability.
- P5-6: comparison tables in L14_SPATIAL_QUALITY_EVALUATION.json; video conditioning path now parity-correct for future use.

## Separate reports
- Text-encoder parity: PASS (post-fix)
- Inference correctness: PASS (DDIM/normalization unchanged)
- Spatial quality: FAIL (baseline and probe)
- CPU probe completed: YES (200/200 steps, 2.6 min)
- Video model ready for temporal training: NO - spatial frames lack objects

## Recommendation
L.15 spatial probe v2: freeze text encoder, UNet-only, lr 1e-4, ~500 steps; verify original Stage2.1 training recipe/steps first.
