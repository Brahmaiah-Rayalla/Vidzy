# L.24 Next Stage Recommendation

RESULT: 10k steps bs4 (40k presentations, ~57 passes) with FULL UNet trainable + direct text channels. val_loss 0.0123->0.0032 BUT caption sensitivity cos 0.9999 and zero-ablate shrank to 0.003 - the model UNLEARNED the text channel during full training. Diversity collapsed 0.266->0.071.

This is the definitive negative: even at 5x L16 scale with verified conditioning pathway, eps-loss produces unconditional denoising. The objective itself does not reward caption use on this synthetic data.

Options for L.25 (pick ONE): (a) abandon prompt-conditioning on this architecture - document ceiling and redirect to unconditional/video work; (b) try a fundamentally different objective (e.g., reconstruction+caption-contrastive at feature level, NOT the failed noise-MSE ranking); (c) reduce problem scope - single-shape-class conditional generation first.
