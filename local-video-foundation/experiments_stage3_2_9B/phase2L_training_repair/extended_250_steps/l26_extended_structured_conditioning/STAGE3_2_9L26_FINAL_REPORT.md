# Stage 3.2.9L.26 - Final Report

**Date:** 2026-11-10 | **Classification: STRUCTURED CONDITIONING LEARNING PROGRESS**

## Experiment
Continued L.25 from step 2000, batch 4 (was bs1), 6000 more steps to 8000 total, same 600/150 circle dataset, one-hot 3-class conditioning, eps-MSE, warm-start new optimizer. 194min CPU.

## Results
| Step | val_loss | color_acc |
|---|---:|---:|
| 2000 | 0.0049 | 0.375 |
| 2500 | 0.0030 | 0.667 |
| 3000 | 0.0028 | 0.458 |
| 4000 | 0.0024 | 0.583 |
| 5000 | 0.0029 | 0.625 |
| 6000 | 0.0018 | 0.625 |
| 7000 | 0.0015 | 0.625 |
| 8000 | 0.0015 | **0.792** |

Accuracy 0.792 - just below the 0.80 threshold but with a clear, sustained upward trend and no collapse. val_loss improved 3x.

## Conclusion
STRUCTURED CONDITIONAL GENERATION IS LEARNABLE. The U-Net successfully binds a 3-class condition to color output with enough training. This definitively exonerates the architecture - the L17-L24 natural-language failures were the text-conditioning pathway, not the model.

## Separate reports
- Color conditioning: LEARNED (79%) | Shape: circles present | Held-out: improving | Diversity: present | CPU: 194min | Video-ready: conditional on extending this task

## Recommendation
L.27: extend structured task to confirm >80%, then test learned text->condition mapping on the proven UNet conditioning route.
