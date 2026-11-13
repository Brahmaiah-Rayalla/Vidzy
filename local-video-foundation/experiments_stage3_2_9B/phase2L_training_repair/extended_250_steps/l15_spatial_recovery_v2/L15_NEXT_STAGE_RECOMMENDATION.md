# L.15 Next Stage Recommendation

Stage2.1 completed only 1000 steps at batch 4 (~4 epochs over 1000 images). L.15 added just 500 bs-1 presentations (<1 epoch) - probe was far too short to expect emergence. UNet-only lr1e-4 training was STABLE (no L14 collapse) but showed mild output-diversity collapse and no recognizable objects.

Recommended L.16: LONG spatial training run matching original recipe scale - UNet-only (TE frozen, now parity-correct), lr 2e-4 (original) or 1e-4, batch 4 if memory allows (~1.5-2s/step -> 2000 steps ~1h), cosine or constant LR, checkpoints every 250, same text_image dataset. Hypothesis: Stage2.1 was simply undertrained; ~4000+ additional presentations should reveal whether objects emerge. Do NOT touch video model yet.
