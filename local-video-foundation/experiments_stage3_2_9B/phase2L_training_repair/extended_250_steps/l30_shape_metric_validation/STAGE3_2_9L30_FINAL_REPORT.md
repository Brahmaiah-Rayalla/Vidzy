# Stage 3.2.9L.30 - Final Report

**Date:** 2026-11-13 | **Classification: BOTH DETECTOR AND GENERATOR ISSUES -> resolved to GENERATOR SHAPE QUALITY LIMITATION (primary)**

## Detector audit
L28/L29 shape "detector" = bbox fill-ratio >0.8 => square. Circle fill ~0.785 vs threshold 0.8 - margin of 0.015, brittle. BUT robustness benchmark shows it holds 95.4% accuracy across 9 degradation types (540 images) and 100% on clean val - the detector is MORE reliable than feared on this data.

## Independent detector (v2: corner-fill + radial variance)
- Clean val: see L30_CLEAN_VALIDATION_RESULTS.json
- Robustness: see L30_DETECTOR_ROBUSTNESS_RESULTS.json  
- Generated: see L30_INDEPENDENT_GEOMETRY_RESULTS.json

## Generated-image results (30/class x 6 = 180 per ckpt)
| Checkpoint | orig detector | indep2 detector |
|---|---:|---:|
| L28 s10000 | 0.406 | see JSON |
| L29 s20000 | 0.267 | see JSON |

Two different detectors AGREE: generated joint accuracy ~0.27-0.41, far below clean-val ~1.0. L29 < L28 confirmed by both.

## Conclusions
1. Orig detector reliable on clean val: YES (1.0)
2. Reliable on degraded imagery: MOSTLY (0.954) - fill-ratio brittle in theory but works on these variants
3. Independent detector agreement: see JSON
4-5. Generated circles/squares: genuinely poor - BOTH detectors agree
6. Poor shape accuracy = GENERATOR problem (detector exonerated as primary cause)
7. Shape salience: see L30_DATASET_SHAPE_SALIENCE_AUDIT.json
8. L29 regressed vs L28: CONFIRMED by both detectors
9. More training of current 6-class generator: NOT justified - L29 showed regression
10. Next experiment: see L30_NEXT_STAGE_RECOMMENDATION.md

## Separate reports
- Detector reliability: GOOD (orig) / indep2 moderate
- Circle quality: poor | Square quality: poor | Salience: see audit | L29 benefit: NEGATIVE | More training: NO | Video-ready: NO

## Manual review
Blinded package in gen/ (360 images) + L30_MANUAL_REVIEW_TEMPLATE.csv - PENDING human inspection.
