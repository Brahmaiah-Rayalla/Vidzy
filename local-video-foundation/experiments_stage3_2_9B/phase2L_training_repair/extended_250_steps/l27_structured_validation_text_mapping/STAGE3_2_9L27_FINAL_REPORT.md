# Stage 3.2.9L.27 - Final Report

**Date:** 2026-10-12 | **Phase A: STRUCTURED CONDITIONAL GENERATION VALIDATED | Phase B: TEXT-TO-CONDITION MAPPING VALIDATED**

## Phase A (steps 8000->10000)
val_loss 0.0015->0.0013; color accs 0.792/0.667/0.667/0.917/0.708 across checkpoints (noisy n=24 evals, all >>0.333 chance). Structured circle+color generation confirmed learnable and stable.

## Phase B (text->color adapter)
- Dataset: 300 prompts, 10 template families, 70/15/15 split by template
- Adapter: masked-mean pooled frozen-TE embedding -> MLP 256->256->3, CE loss, 80 epochs
- Train/val/test accuracy: 1.0 / 1.0 / 1.0 - PERFECT generalization to held-out templates
- E2E: adapter->onehot->frozen generator produces correctly colored circles

## Key conclusion
The frozen Stage2.1 text encoder DOES retain color information - the L17-L24 failures were the diffusion conditioning pathway, not text understanding. A tiny supervised adapter bridges language to the proven structured interface. Full pipeline prompt->image works for constrained circle+color task.

## Separate reports
- Phase A: VALIDATED | Phase B: VALIDATED (test_acc 1.0) | Generator frozen: YES | E2E: works | CPU: ~1.3h total

## Recommendation
L.28: expand structured task to shape+color joint conditioning (6 classes), retrain generator, extend adapter - before attempting open-vocabulary.
