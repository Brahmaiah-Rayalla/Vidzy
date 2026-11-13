# L.20 Next Stage Recommendation

Both architectures FAILED to bind captions: learned-query entropy stayed ~0.9997 after 750 steps (queries did NOT become selective); FiLM modulation produced no measurable caption sensitivity (swap cos 0.99998 pre AND post training); MSE-to-GT worse than L17 baseline (A 0.34, B 0.24-0.65 vs L17 0.13).

Deeper conclusion: the bottleneck is NOT the attention architecture - even fresh learned queries with direct text access failed to differentiate captions under the eps-denoising objective at this data scale. The objective itself may not reward caption use: with only 10 pairs and heavy noise, unconditional denoising dominates. OR FiLM injection point (post-block output) is too weak.

Recommended L.21: two-pronged diagnostic - (a) DIRECT conditioning test: train FiLM to predict a known function of caption (e.g., classify caption attribute from modulated features) to verify the modulation path can carry information at all; (b) much stronger conditioning signal test: concatenate text embedding channels directly into UNet input (bypass attention entirely) as an existence proof that ANY conditioning path can drive generation on this data. If (b) fails too, the problem is data/objective scale, not architecture.
