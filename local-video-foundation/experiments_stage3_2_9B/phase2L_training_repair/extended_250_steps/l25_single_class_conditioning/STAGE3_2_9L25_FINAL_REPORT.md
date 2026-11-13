# Stage 3.2.9L.25 - Final Report

**Date:** 2026-11-10 | **Classification: CONDITIONING SENSITIVITY WITHOUT VISUAL SUCCESS**

## Experiment
600 train/150 val synthetic circles (red/blue/yellow, varied pos/radius, masks), 3-channel one-hot conditioning broadcast to 64x64, Stage2.1 UNet + 6ch conv1, 2000 steps bs1 AdamW lr1e-4, 27.6min CPU.

## Results
- Color accuracy: 0.389 (chance 0.333) - above chance, far below 80% threshold
- val_loss: 0.0075->0.0037 improving
- Pre-training sensitivity: red_vs_blue 0.025 (real signal in cond branch)
- First measurable conditioning effect in the entire L-series

## Interpretation
Structured labels produce stronger conditioning signal than natural-language captions EVER did (direct comparison: text swap cos 0.9999 vs here measurable class effects). The architecture is NOT incapable of conditional generation - the natural-language pathway was the failure, not the U-Net. But 2000 steps bs1 (~3 epochs) is insufficient for reliable binding.

## Answers
- Condition channel functional: YES (measurable)
- Conditional generation quality: WEAK (39% vs 80% needed)
- Color accuracy: 39% | Shape: untested separately | Class balance: perfect | Diversity: present | CPU: 27.6min
- Natural-language conditioning revisit? YES - with this evidence the failure is the text pathway/objective, not the UNet
- Video ready: NO

## Recommendation
L.26: extend THIS task to 5000-8000 steps bs4 (~1-2h) - if accuracy crosses 80%, structured conditioning is validated and the path forward is fixing the text encoder->condition mapping, not the UNet.
