# L.29 Next Stage Recommendation

Extended training 10k->20k did NOT improve joint generation: baseline joint 0.428 (n=180) -> s20k 0.283 (n=180), with non-monotonic trajectory 0.361/0.328/0.406/0.283. val_loss improved 0.0011->0.0009 but semantics plateaued/regressed - loss is not the binding metric. Shape accuracy dropped below chance mid-run (0.34-0.44) while color held ~0.6-0.8, indicating shape binding is fragile and possibly interfering.

The epsilon-MSE objective at this dataset scale appears saturated for joint shape-color binding. More of the same training is unlikely to help.

Recommended L.30: change ONE variable - either (a) increase shape salience in the data (larger shape differences, e.g. circle vs triangle with distinct silhouettes) to give the condition a stronger gradient, or (b) test whether the bottleneck is the DETECTOR (shape classification via fill-ratio may be unreliable on blurry DDIM outputs) by evaluating generated images visually first. Do not extend training again without a new hypothesis.
