# L.30 Next Stage Recommendation

Detector exonerated: orig fill-ratio detector holds 95% on degraded benchmark and 100% on clean val; independent corner-fill detector agrees generated accuracy is ~0.27-0.41 (not detector artifact). L29 genuinely regressed vs L28 - more epsilon-MSE training is NOT justified.

The generator produces correct COLOR (~0.6-0.8) but shapes are not reliably circle-vs-square. Shape binding is the bottleneck and it is a GENERATOR limitation, not measurement error.

Recommended L.31 (single variable): increase SHAPE SALIENCE - retrain the 6-class generator on a dataset where circle vs square differ more strongly (e.g., add triangle as third shape OR use shapes with more distinct silhouettes like circle vs X/plus), keeping color classes identical. If shape accuracy jumps, salience was the issue; if not, the architecture's shape-conditioning capacity is the ceiling. Alternative: accept 3-class color-only generator as the validated milestone and pivot to temporal work with color conditioning.
