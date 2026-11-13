# Stage 3.2.9L.29 - Final Report

**Date:** 2026-11-13 | **Generator: JOINT CONDITIONING LEARNING PROGRESS (plateaued) | Pipeline: MODULES VALIDATED - GENERATOR QUALITY INSUFFICIENT**

## What was done
Continued L28 6-class generator 10k->20k steps (bs4, opt restored, ~4.5h CPU). Baseline re-eval n=30/class. Evals at 12.5k/15k/17.5k/20k.

## Results (n=30/class=180 per eval)
| Step | val_loss | joint | color | shape |
|---|---:|---:|---:|---:|
| 10000 | 0.0011 | 0.428 | 0.822 | 0.500 |
| 12500 | 0.0009 | 0.361 | 0.772 | 0.444 |
| 15000 | 0.0010 | 0.328 | 0.789 | 0.422 |
| 17500 | 0.0009 | 0.406 | 0.778 | 0.511 |
| 20000 | 0.0009 | 0.283 | 0.583 | 0.339 |

## Findings
- Extended training did NOT improve joint generation - plateau ~0.3-0.4, final 0.283 vs baseline 0.428
- val_loss kept improving (0.0011->0.0009) while semantics stagnated - confirms loss != generation quality
- Shape binding is fragile: dropped below chance mid-run, partially recovered, fell again at 20k
- Color more robust (~0.6-0.8) but also declined at 20k
- Condition still causal (joint >> 0.167 chance) - not unconditional collapse
- L28 adapter regression: still ~1.0 on test prompts (frozen, unchanged)

## Classification rationale
Not VALIDATED (below 80% gate). Not pure TRADE-OFF (both attributes declined at end). Best fit: JOINT CONDITIONING LEARNING PROGRESS - the capability exists (L28 proved learnability) but this objective+scale has plateaued. High eval variance (n=180, single-seed DDIM) means differences are noisy, but no checkpoint shows consistent improvement over baseline.

## Separate reports
- 6-class gen: plateau ~0.3-0.4 joint | color ~0.7 | shape ~0.4-0.5 | adapter: 1.0 frozen | e2e: limited by generator | CPU: 4.5h

## Recommendation
L.30: change one variable - increase shape salience in data OR validate the shape detector on generated images before more training. Do NOT extend training again without new hypothesis.
