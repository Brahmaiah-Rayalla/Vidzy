# L.18 Next Stage Recommendation

Verified chain: mask semantics CORRECT (padding gets exactly 0 attention); text signal reaches UNet (attn_out RMS 0.13-0.29 vs h_in 0.6-0.7); BUT caption CONTENT does not affect output (swap cosine 0.9998) and 4x residual gain only reaches mean|d|=0.017. Binding loss stayed at margin floor (lc~ls) - correct vs swapped captions produce identical denoising error. Both arms failed to improve memorization vs L17.

Root cause hypothesis: cross-attention softmax over ~10-16 valid tokens is near-uniform, so output = mean of value vectors = caption-invariant bag-of-embeddings. Conditioning is averaged away, not masked.

Recommended L.19: attention-weight forensics - log per-head attention entropy and top-token identity for correct vs swapped captions; test whether attention concentrates on content words; if uniform, the fix is architectural (e.g., caption-level pooled conditioning or cross-attn with learned queries), not more training.
