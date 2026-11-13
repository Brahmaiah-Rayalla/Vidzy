# L.16 Next Stage Recommendation

8000 additional presentations (total ~12000) produced NO recognizable objects and DEGRADED diversity/color fidelity while val_loss improved - the loss is decoupled from visual quality. Do NOT run another blind extension.

Recommended L.17: diagnostic, not training - (a) inspect actual generated PNGs at each checkpoint to confirm metric interpretation; (b) test whether text conditioning signal reaches UNet (cross-attn weight analysis / caption-swap test); (c) try original joint UNet+TE training at original recipe (bs4 lr2e-4 was the original config - L14 collapse may have been bs1-specific); (d) verify DDIM sampler produces sane images on TRAINING data reconstruction (overfit 10 imgs test). If overfit test fails -> objective/architecture issue confirmed.
