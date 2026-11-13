# L.30 Existing Detector Audit

The "detector" used in L.28/L.29 was an inline heuristic (not a saved module):
- Color: mean RGB distance of pixels >0.15 from bg(245) to 3 class colors -> argmin
- Shape: fill-ratio of axis-aligned bounding box of mask; >0.8 => square else circle

Assumptions: flat uniform object color, sharp boundaries, low bg variance, axis-aligned bbox meaningful, fill-ratio separates circle(~0.785) from square(1.0) - margin only ~0.015 at threshold 0.8! Very brittle: antialiasing, blur, or irregular blobs flip the decision. Circle fill is ~0.785 < 0.8 threshold - razor thin.

Independent detector (L30): radial-distance variance from centroid (circle=low var, square=high var due to corners) + same color rule.
