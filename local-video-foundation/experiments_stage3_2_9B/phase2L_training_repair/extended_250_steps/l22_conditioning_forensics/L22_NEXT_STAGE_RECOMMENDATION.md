# L.22 Next Stage Recommendation

Forensics: caption signal enters at conv1 (rel_diff 0.137) and attenuates ~2.7x by block20 (0.051) - it SURVIVES but is weak. Conditioning-only training (500 steps, frozen backbone) made memorization WORSE (mse 0.175->0.31) and did not improve caption discrimination (swap 0.011, cos 0.9999). AdamW weight-decay also nudged the RGB conv slice (~1%, documented - hook zeroed grads but decoupled decay still applied).

Conclusion: conditioning params alone cannot bind captions; the frozen backbone both limits translation AND the conditioning adaptation interferes with denoising. Caption info is present but too weak at ~5% relative feature difference.

Recommended L.23: LIMITED UNFREEZE - train conv1 text-slice + tokproj + FIRST residual block only (or first down-stage), keeping deeper backbone frozen. This tests whether early-stage plasticity can amplify the conditioning signal without full end-to-end training. Alternative: amplify text-channel gain at conv1 (learned scalar) as cheap sensitivity boost.
