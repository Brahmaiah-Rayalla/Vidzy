# L.17 Next Stage Recommendation

Verified: (1) objective/sampler mathematically consistent (eps-pred, DDIM inverts forward exactly); (2) U-Net text sensitivity ~ZERO (cosine 0.9998 on caption swap); (3) cross-attn grads ~5x smaller than average; (4) 10-image overfit PARTIALLY succeeded (MSE-to-GT 0.41->0.13 at s750, some pairs near-memorized) but regressed by s1000; (5) diversity preserved.

Diagnosis: the denoiser CAN learn images, but text conditioning contributes almost nothing to predictions - the cross-attention pathway is effectively dead weight at current scale. This explains ALL prior failures: lower val_loss = better unconditional denoising, not prompt alignment.

Recommended L.18: conditioning-signal amplification probe - (a) measure text_proj output magnitude vs h magnitude inside cross_attn; (b) try scaling text_proj init or adding a learned scale; (c) overfit 10 pairs WITH caption-swap loss term to force conditioning use; (d) check whether attention_mask is being applied correctly in cross_attn (mask semantics bug possible - mask=1 real tokens; verify it is not inverted).'
