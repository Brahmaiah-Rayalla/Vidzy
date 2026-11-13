# L.25 Architecture

Stage2.1 UNet + 6ch input (3RGB+3 onehot color). SplitConv6: conv_rgb+conv_cond(3ch,std0.02). Cross-attn bypassed. eps-MSE, AdamW lr1e-4 wd0.01, bs1, 2000 steps.
