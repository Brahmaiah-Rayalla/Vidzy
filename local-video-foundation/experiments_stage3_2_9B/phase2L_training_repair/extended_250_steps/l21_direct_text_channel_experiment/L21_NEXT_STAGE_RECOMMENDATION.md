# L.21 Next Stage Recommendation

Direct channels WORK mechanically: zero-ablation delta 0.06-0.08 (12x attention path) proves text channels causally drive predictions. BUT caption discrimination remains weak (swap delta 0.009-0.015) and Arm B binding loss did not help. Arm A reached mse_mean 0.144 ~ L17 baseline.

Bottleneck localized to REPRESENTATION not pathway: UNet receives text but cannot yet map caption differences to image differences.

Recommended L.22: (a) measure conv1 text-branch activation difference between color-swapped captions (is discriminative signal present pre-UNet?); (b) conditioning-params-only training probe; (c) reconsider binding loss formulation (feature-level not noise-MSE).
