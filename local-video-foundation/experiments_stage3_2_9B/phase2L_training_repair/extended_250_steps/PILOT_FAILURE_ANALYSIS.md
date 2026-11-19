# Shape/color pilot failure analysis

**Status:** raw RGB shape/color gate remains closed. A separate structured-geometry renderer route has passed its visual gate, and an event-driven collision pilot has passed its physics/runtime gate; learned motion remains unimplemented.

## What failed

The first “L31 failed” conclusion was itself based on an unreliable generated-image detector. Reclassifying its saved outputs changed its point estimates from the original report: L27→L31 moved joint accuracy 28.89%→33.33%, shape 47.22%→51.67%, and color 62.78%→67.78%. The paired 95% intervals cross zero for all three, with severe class-to-class swings, so this is **inconclusive and unpromoted**, rather than a confirmed regression.

The compact L32 curriculum corrected several known defects, but generation still collapses toward triangles. Corrected DDIM, timestep embeddings, AdaGN, and a direct silhouette head did not produce dependable circles. L33’s silhouette head segments held-out forward-noised training examples accurately, but its thresholded projection fails on reverse-process samples. The underlying denoiser’s generated states are out of the head’s training distribution.

## Confirmed contributing factors

| Factor | Evidence | Change made / current status |
|---|---|---|
| L31 size confound | Mean circle area 470.54 px² vs triangle area 235.96 px² | L32 data is area matched: 293.57 vs 291.21 px² |
| L31 salience objective not implemented | L31 saved masks but optimized global epsilon MSE only | L32 added foreground-weighted epsilon, RGB reconstruction and silhouette terms |
| Weak/washed-out class pathway | Constant one-hot maps; L32b timestep embeddings did not improve shape; AdaGN improved color only | L32b and L32c evaluated and stopped; circle collapse remains |
| Sampling inconsistencies | No final t=0 x0 reconstruction; intermediate x0 estimates were not clipped to trained range [-1,1] | Fixed in the pilot runner and configurable sampler evaluator; same-checkpoint shape score barely changed |
| Aggregate denoising metric overstates generation quality | L32 held-out noise MSE 0.01187, while generated shape accuracy was about 50% | Reports now separate raw sample accuracy, masks, color and visual grids |
| Mask-head reverse-process domain shift | L33 validation mask IoU 0.934 / boundary F1 0.998 at t=100, while generated thresholded masks are fragmented and shape accuracy drops to 39.44% | Projection rejected; head is not used as a generation fix |
| Detector calibration gap | Clean and additive Gaussian-noise accuracy was 100%, but structured diffusion artifacts differ | Keep scores paired with sample grids; treat current radial classifier as provisional |
| Pilot provenance/evaluation bookkeeping | L31 timestep RNG was not explicit; L32 paired helper omitted its return value | Seed, manifests, sampler settings and paired reports are now saved; L32 comparisons recomputed from per-sample rows |

## Matched L32-family results

Each run uses 30 fixed-noise samples for each of six color/shape classes. The selected results use the same corrected 12-step DDIM sampler: clip each predicted x0 to [-1,1], update to the next timestep, then reconstruct x0 at t=0.

| Checkpoint/output | Joint accuracy | Shape accuracy | Color accuracy | Decision |
|---|---:|---:|---:|---|
| L32 raw diffusion | 41.11% | 50.56% | 81.67% | Fails: circles collapse |
| L32b raw diffusion, timestep embedding | 41.11% | 48.89% | 87.78% | No shape gain |
| L32c raw diffusion, AdaGN | 42.22% | 49.44% | 89.44% | Color gain; shape gate fails |
| L33 raw diffusion, silhouette-head fine-tune | 45.56% | 48.89% | 96.11% | Color gain; no shape gain |
| L33 thresholded mask projection | 39.44% | 39.44% | 81.11% | Rejected; fragmented masks |

The L33 raw-vs-L32c paired shape change is −0.56 percentage points (95% interval −3.33 to +1.67). L33 color improves by 6.67 points (3.33 to 10.56), but circles remain at 0–6.7% joint accuracy. The mask projection regresses shape by 10 points (−15 to −5). These numbers do not pass the shape gate.

## Interpretation

The simple 2D U-Net is small enough for CPU use; there is no evidence here that gradient vanishing or 3D/temporal layers caused the failure. The central problem is that epsilon/noise prediction quality does not translate to a stable reverse trajectory for this small, class-conditional shape dataset. Model samples drift from the forward-noised data distribution, which also explains why the mask head works on held-out images but not on generated states. Extra sampling steps and extra fine-tuning on the same objective are not justified by these results.

**Recommendation:** keep raw diffusion and renderer-backed quality separate. L34 implements the compact structured geometry route and passes crisp shape/color rendering by construction, although its learned geometry distribution is approximate (worst held-out KS distance 0.32). L35 uses that route for an event-driven collision baseline and passes 12-seed collision/bounds/runtime checks. It does not learn trajectories: the event solver is explicit physics. The next motion experiment should define held-out trajectory data and add a small learned variation model only if it beats this physics baseline on endpoint, trajectory, and event metrics. Do not resume claims about raw shape fidelity until the raw path passes its own matched edge gate.

## Subsequent structured-route results

The [L34 gate decision](l34_structured_geometry_diffusion/L34_GATE_DECISION.json) records a renderer-backed visual pass and the remaining geometry-distribution limitation. The [L35 event-motion gate](../../../../outputs/cpu_event_motion_l35/L35_GATE_DECISION.json) records 12 held-out seeded simulations: exactly one circle collision per six-second clip, no sampled overlap, no wall-bound violation, and a maximum collision-time error below 3e-15 seconds. This is a verified physics/rendering result, not a learned motion result. The generated [two-circle collision GIF](../../../../outputs/cpu_event_motion_l35/two_red_circles_6s.gif) and its [metrics/provenance](../../../../outputs/cpu_event_motion_l35/two_red_circles_6s.json) are available for inspection.

L36 adds an optional train-only quantile calibration layer to L34 sampling. It reduced mean per-feature KS from 0.1873 to 0.1440 over three checkpoints, with each seed's worst feature distance improving; some individual class-feature scores still regress. It does not retrain model weights, and the worst distance remains 0.33, so keep raw L34 as the default pending a larger validation set. The [L36 report](l34_structured_geometry_diffusion/l36_quantile_calibration/L36_RESULTS.json), [gate](l34_structured_geometry_diffusion/l36_quantile_calibration/L36_GATE_DECISION.json), and [comparison grid](l34_structured_geometry_diffusion/l36_quantile_calibration/L36_raw_vs_calibrated_grid.png) show the comparison. The [optional calibrated six-second collision video](../../../../outputs/cpu_event_motion_l35/two_red_circles_6s_calibrated.gif) also passes the existing 12-case physics checks.

L37 compared joint held-out geometry using a multivariate energy distance and correlation-matrix error. Since the curriculum generator samples center, size, and rotation independently, a train-fitted factorized empirical sampler beat raw L34 and L36 on these joint metrics across all three seeds. It is available only as an optional renderer-backed path; it is not neural generation and does not pass the raw RGB gate. The [L37 results](l34_structured_geometry_diffusion/l37_joint_geometry/L37_RESULTS.json), [gate decision](l34_structured_geometry_diffusion/l37_joint_geometry/L37_GATE_DECISION.json), [comparison grid](l34_structured_geometry_diffusion/l37_joint_geometry/L37_joint_geometry_comparison.png), and [six-second factorized-prior video](../../../../outputs/cpu_event_motion_l35/two_red_circles_6s_factorized.gif) record the outcome.

L38 connects the existing v37 trained slow/quick velocity table to the exact event simulator. It produced visibly different first-impact times (2.33 vs 2.08 seconds) while retaining one collision, no overlap, no crop, and sub-second CPU generation across both 12-seed gates. The learned table selects tempo; it does not predict framewise trajectories, and no weights were retrained. The [L38 gate](../../../../outputs/cpu_event_motion_l38/L38_GATE_DECISION.json), [slow video](../../../../outputs/cpu_event_motion_l38/two_red_circles_6s_slow.gif), and [quick video](../../../../outputs/cpu_event_motion_l38/two_red_circles_6s_quick.gif) preserve the result.

L39 trains a small CPU MLP on synthetic analytical speed profiles and uses its bounded positive speed curves inside the L35 event solver. Three model seeds reduced held-out profile RMSE by 41.1% versus constant speed; 600 collision scenarios had no overlaps/crops and exactly one circle impact. A first quick-profile evaluation found repeat impacts for some trajectories; the impact window was moved later and the 100-case/style and 600-case multi-model gates then passed. This is synthetic-teacher profile regression, not learning motion from video. The [L39 gate](../../../../outputs/cpu_event_motion_l39/L39_GATE_DECISION.json), [multi-seed report](l39_learned_speed_profiles/L39_MULTI_SEED_GATE.json), [slow demo](../../../../outputs/cpu_event_motion_l39/two_red_circles_6s_slow_profile.gif), and [quick demo](../../../../outputs/cpu_event_motion_l39/two_red_circles_6s_quick_profile.gif) preserve the result.

## Artifacts

- [Current model status](../../../CURRENT_MODEL_STATUS.md)
- [L31 robust detector and area audit](l31_high_salience_shape_conditioning/eval_checkpoint_2500_seed42_n30/failure_audit_robust_metrics.json)
- [L32 corrected-sampler report](l32_cpu_factorized_salience_pilot/eval_clipped_x0/L32_RESULTS.json)
- [L32b corrected-sampler report](l32b_conditioned_timestep_pilot/eval_clipped_x0/L32_RESULTS.json)
- [L32c corrected-sampler report](l32c_adagn_conditioning_pilot/eval_clipped_x0/L32_RESULTS.json)
- [L33 gate decision](l33_auxiliary_silhouette_head_pilot/L33_GATE_DECISION.json)
- [L33 raw and projected sample grid](l33_auxiliary_silhouette_head_pilot/eval_threshold_050/generation_grid.png)
- [L33 held-out mask-head threshold audit](l33_auxiliary_silhouette_head_pilot/eval_mask_head_validation/MASK_HEAD_VALIDATION_AUDIT.json)
- [L34 multi-seed summary](l34_structured_geometry_diffusion/L34_MULTI_SEED_SUMMARY.json)
- [L34 gate decision](l34_structured_geometry_diffusion/L34_GATE_DECISION.json)
- [L35 event-motion gate and video metrics](../../../../outputs/cpu_event_motion_l35/L35_GATE_DECISION.json)
- [L36 geometry calibration report and gate](l34_structured_geometry_diffusion/l36_quantile_calibration/L36_RESULTS.json)
- [L37 joint geometry report and gate](l34_structured_geometry_diffusion/l37_joint_geometry/L37_RESULTS.json)
- [L38 learned tempo event-motion gate](../../../../outputs/cpu_event_motion_l38/L38_GATE_DECISION.json)
