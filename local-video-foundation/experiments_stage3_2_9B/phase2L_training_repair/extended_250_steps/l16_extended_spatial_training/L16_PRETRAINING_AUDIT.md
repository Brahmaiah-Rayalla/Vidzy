# L.16 Phase 1 — Pretraining Audit

- Stage2.1 original: 1000 steps, bs4, lr2e-4, wd0.01, seed42, 1000-image synthetic set (~4000 presentations, ~4 epochs), final scheduled ckpt, optimizer state saved but for joint UNet+TE param set -> L.16 is WARM-START with fresh AdamW.
- L.15 probe: 500 steps bs1 lr1e-4 UNet-only, stable, no objects, mild diversity decline. Init = same Stage2.1 ckpt.
- L.16 init: ORIGINAL Stage2.1 ckpt (not L.15 step500) per spec.
- text_image dataset: 800 metadata entries; L.16 uses first 700 train / last 100 val (disjoint by index; images are unique files). Captions are single-image scene descriptions - correct for spatial model.
- Difference vs original dataset: original generated 1000 images (data/text_image has 800 metadata files present); same generator family/config (image_size 64, max_objects 2).
- Parity: L14 fix verified; L16 loads standard TextEncoder directly (no split encoder) - parity regression test passed in L14.
