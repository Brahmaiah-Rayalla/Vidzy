# Stage 3.2.9L.15 - Final Report

**Date:** 2026-11-09 | **Classification: SPATIAL MODEL STILL UNDERTRAINED**

## Phase results
- P1 Audit: Stage2.1 = 1000 steps, bs4, lr2e-4, wd0.01, 1000-image synthetic set, optimizer state saved, no EMA, final scheduled ckpt (not val-selected). Only ~4 epochs - severely undertrained for text-to-image.
- P2 Baseline: corrected-encoder Stage2.1 still abstract (corner_std 0.29-0.43, pairwise diff 0.233).
- P3-5 Probe: UNet-only, TE frozen+eval, AdamW lr1e-4 wd0.01, bs1, 500 steps on text_image dataset (700/100 split), ckpts at 0/100/200/350/500 with 12-sample DDIM evals. STABLE - no L14 collapse. Wall time 43min (8min train + ~35min sampling).
- P6 Eval: val_loss 0.0287->0.0472->0.0259->0.0240 (noisy, no clear gain). Pairwise sample diversity FELL (0.233->0.067-0.128) = mild mode collapse toward less prompt-differentiated outputs. corner_std fell mid-run then rose. NO recognizable objects at any checkpoint.
- P7 Comparison: L14 joint probe collapsed entirely (constant green); L15 frozen-TE UNet-only remained stable but did not produce objects in 500 bs-1 steps (<1 epoch).
- P8 Integrity: TE bit-frozen verified; originals untouched; probe ckpts saved separately.

## Separate reports
- Original Stage2.1 steps: 1000 (bs4, ~4000 presentations)
- Continuation status: WARM-START (new optimizer, different param set)
- Text encoder freeze: VERIFIED
- U-Net training success: STABLE but no visual emergence
- Visual recognition: FAILED at all checkpoints
- CPU runtime: 43.4 min total
- Ready for temporal video training: NO

## Recommendation
L.16: match original recipe scale - UNet-only, lr 2e-4, bs4, ~2000 steps (~1-1.5h CPU) on text_image data; hypothesis = undertraining, not architecture.
