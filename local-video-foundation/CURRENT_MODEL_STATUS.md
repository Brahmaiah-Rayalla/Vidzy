# Current Model Status

**Updated:** 2026-11-18  
**Focus:** CPU-only shape and short-video generation

This file and the README are reconciled to the current v93/v90 CPU route. Raw diffusion results, renderer-assisted outputs, and video runtime are reported separately.

## Latest small-triangle route: v93 (2026-11-18)

v93 is a 1,000-step CPU fine-tune from v92 at learning rate `5e-6`. The remaining area error exposed a training/evaluation threshold mismatch: training marked any nonblack antialiased pixel above about 6/255 as foreground, while blue-shape evaluation counts blue pixels above 55/255. v93 sets the training threshold to normalized `-0.5686` and the occupancy sigmoid scale to 12, matching the evaluator threshold while keeping pure black near zero occupancy. Data, seed list, resolution, sampler, and the other loss weights are held fixed.

On the same 12 blue-triangle seeds against v92, v93 improved centered IoU on 10/12 seeds, boundary F1 on 11/12, and closeness of area to target on 11/12. Centered IoU and boundary F1 paired-bootstrap 95% intervals exclude zero; raw unaligned IoU changed only `-0.0043` on average, with interval `[-0.0423, 0.0390]`, so the small decline is inconclusive. Raw centroid error rose `0.36 px` on average, also with an interval spanning zero. v93 is routed only for small triangles; keep v90 for medium and large.

| Small triangle metric | v90 | v91 | v92 | v93 specialist |
|---|---:|---:|---:|---:|
| Centered IoU | 0.625 | 0.718 | 0.757 | 0.874 |
| Raw all-component unaligned IoU | 0.577 | 0.615 | 0.625 | 0.621 |
| Boundary F1 @ 1 px | 0.267 | 0.561 | 0.666 | 0.970 |
| Foreground area ratio | 1.605 | 1.401 | 1.332 | 0.998 |

Small area is now calibrated on average; per-seed variance remains. Held-out small v-MSE improved from `0.01801` to `0.01536` on 43 balanced validation examples. The [v93 routed raw sample](outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/routed_small_raw_samples/grid.png), [paired 12-seed comparison](outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/paired_eval_v92_to_v93_12seed/paired_seed_1234.png), and [metrics](outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/paired_eval_v92_to_v93_12seed/evaluation_metrics.json) record the result.

The routed v93 small blue triangle generated a raw-anchor 72-frame, six-second video at 384 px in `1.03 s` on CPU with four threads; there was no canonical geometry renderer or palette replacement. See the [video](outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/raw_triangle_small_6s_384.mp4), [frame preview](outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/raw_triangle_small_6s_384_preview.png), and [generation metrics](outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/raw_triangle_small_6s_384.metrics.json).

The compact download bundle is `release/vidzy_cpu_raw_triangle_bundle.zip`; its checksum is in the adjacent `.sha256` file. It contains v93/v92/v91/v90 spatial weights, v86 initialization weights, v37 motion weights, configs, routed evaluation metrics, and sample outputs. Optimizer state is stripped from packaged checkpoints. The source workspace checkpoints remain unchanged.

## Latest raw-triangle result: v89 diagnosis and v90 fix (2026-11-18)

The v89 pass did not fix triangle clarity. The run used a `2e-7` learning rate; all 115 logged gradient norms exceeded the clip threshold of 1 (mean 25.88), so the already undertrained v86 checkpoint barely adapted: the common weights moved only `0.00265` relative L2 (`0.000152` mean absolute delta). Its large-only focus also damaged the small route. The default training preview cycled circle/square/triangle labels, so three of its four tiles were not trained by that specialist. The native-resolution v89 targets also varied position by up to 8 px and angle by up to 5 degrees without conditioning those attributes, while the validation loop took the first 128 size-sorted images and therefore evaluated only small shapes.

v90 addresses those causes: it uses centered 128 px antialiased triangles with no hidden pose variation, balanced color/size combinations for the triangle class, a `5e-5` learning rate, and triangle-only previews. The evaluator now chooses a deterministic balanced validation subset. Training completed 5,000 CPU steps in 1,797 seconds with four threads; the [training run manifest](outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/training_run_manifest.json) records the config, checkpoint and training-data hashes. The model remains a compact 2D U-Net; no 3D convolutions or broad temporal attention were added.

Paired raw generation used 12 fixed seeds, the same sampler, and blue triangles at each size. Raw IoU retains every color-connected component and does not align the generated position; centered IoU and boundary F1 use the legacy largest-component, center-aligned comparison.

| Size | Centered IoU v86 → v90 | Raw unaligned IoU v86 → v90 | Boundary F1 v86 → v90 | Centroid error v86 → v90 |
|---|---:|---:|---:|---:|
| Small | 0.405 → 0.625 | 0.201 → 0.577 | 0.245 → 0.267 | 10.41 → 3.08 px |
| Medium | 0.537 → 0.842 | 0.461 → 0.719 | 0.168 → 0.602 | 7.62 → 3.60 px |
| Large | 0.706 → 0.894 | 0.547 → 0.827 | 0.263 → 0.653 | 11.10 → 2.49 px |

Balanced held-out v-MSE improved for small/medium/large from `0.0277/0.0554/0.0815` to `0.0165/0.0224/0.0210` over `43/43/42` images. Small-triangle edge quality remains the weakest slice: its mean boundary F1 is only `0.267` with high seed variance, so this is a material improvement, not a claim that every raw diffusion issue is solved.

The [v90 targeted color preview](outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/samples_final.png), [paired 12-seed evaluation](outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/raw_eval_12seed/paired_seed_1234.png), and [metrics](outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/raw_eval_12seed/evaluation_metrics.json) show the outcome. A raw RGB v90 anchor plus the v37 learned motion table generated a 72-frame, six-second, 12 fps, 384 px CPU video in `2.82 s` using four threads (`0.47 s` diffusion sampling); the path used neither canonical geometry rendering nor palette replacement. See the [video](outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/raw_triangle_large_6s_384.mp4), [frame preview](outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/raw_triangle_large_6s_384_preview.png), and [generation metrics](outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/raw_triangle_large_6s_384.metrics.json).

## Latest raw-triangle follow-up: 2026-11-18

At that point in the history, the selected 64 px routes were v65 (small), v69 (medium), and v70 (large). The 128 px v90 result above is newer and is the current measured triangle route; these 64 px results remain historical runtime references. Candidate metrics and provenance for the intervening experiments are stored under `outputs/cpu_triangle_large_raw_rgb_v73` through `outputs/cpu_triangle_large_raw_rgb_v78`.

| Experiment | Result | Decision |
|---|---|---|
| v73 background clean-pixel loss | Large geometry was effectively tied with v72; no visible RGB cleanup | Not promoted |
| v74 full-resolution residual block | 12-seed large centered IoU 0.8994 and boundary F1 0.9056, below v72's 0.9023 / 0.9118 | Not promoted |
| v75 stronger foreground/background RGB losses | Blue intensity rose slightly, but background noise rose and centered IoU/boundary F1 fell to 0.8937 / 0.8963 | Not promoted |
| v76 canonical 42 px training silhouettes | Shape and background metrics regressed; the isolated data normalization was not sufficient | Not promoted |
| v77 128 px fine-tune | Better than its upscaled 128 px baseline, but large centered IoU was only 0.6162 and edges remained weak | Not promoted |
| v78 timestep-gradient balancing | Large centered IoU / boundary F1 were 0.9027 / 0.9144, essentially tied with v71; raw RGB error did not improve | Not promoted |
| v79 64 px specialist from scratch | After 4,000 CPU steps, large centered IoU / boundary F1 were 0.8459 / 0.8047; blue raw RGB MAE was 28.22 versus v70's 14.78 on the same seed | Not promoted |

The best v78 checkpoint screen was step 200; its four-seed boundary F1 was only 0.0007 higher than v71 and its raw RGB MAE changed from 14.75 to 14.74, too small to count as an improvement. The trainer now compensates the clean-image auxiliary-loss gradient for the `sqrt(1-alpha_bar)` factor in v-prediction, capped to avoid extreme weights. Training YAML loss weights also now take effect when CLI overrides are omitted. These corrections are useful, but the measured samples show that the raw RGB quality issue is not solved yet.

The raw video path now enlarges the generated RGB anchor with Lanczos and translates it at output resolution, avoiding a 64 px bilinear warp followed by nearest-neighbor enlargement. This is display/motion interpolation only: the shape and colors still come from raw diffusion, with no palette replacement or mask renderer. The v70 reference is 72 frames / 6 seconds at 12 fps, sampled in 24 DDIM steps; denoising took 0.21 s and total CPU render time was 2.04 s with four threads. The [raw video](runs/raw_diffusion_fix01/raw_rgb_anchor_large_v70_s2345_24steps_hiresmotion.mp4) and [mid-frame preview](runs/raw_diffusion_fix01/raw_rgb_anchor_large_v70_hiresmotion_midframe.png) show the remaining colored speckle. A 100-step sample took 2.70 s total and looked materially the same, so 24 steps remains the better speed/quality tradeoff.

The sampler also supports `--diffusion_ensemble_samples N`, averaging independent denoised RGB anchors before rigid motion. On one matched v70 seed, four samples reduced upper-left background-patch mean absolute RGB from 2.82 to 2.33 and mean channel standard deviation from 5.08 to 3.84, but the triangle edges became softer. Eight samples looked softer still. This is an experimental diffusion-sample consensus, not a single stochastic draw, and it is **not accepted as a clarity fix**. See the [four-sample video](runs/raw_diffusion_fix01/raw_rgb_anchor_large_v70_s2345_24steps_ensemble4.mp4) and its [mid-frame preview](runs/raw_diffusion_fix01/raw_rgb_anchor_large_v70_ensemble4_midframe.png).

The continued 128 px v81 fine-tune (2,000 more steps) improved its own weak 128 px baseline over 12 seeds: large centered IoU / boundary F1 rose from 0.5637 / 0.1783 to 0.6190 / 0.2541. That remains well below the selected 64 px v70 route (0.9028 / 0.9045), and the generated preview is still not a clean triangle. v81 is not promoted. Its [paired metrics](outputs/cpu_triangle_large_raw_rgb_v81_128_finetune/raw_eval_12seed_vs_v72_128/evaluation_metrics.json) and [sample preview](outputs/cpu_triangle_large_raw_rgb_v81_128_finetune/samples_final.png) document the failure.

## Historical 64 px raw-diffusion baseline: v54/v65/v69/v70

This section preserves the earlier **raw diffusion image evaluation only**: 12 matched noise seeds per size, 24 DDIM steps, 64x64 samples, and four CPU threads. Those 64 px routes are historical; the newer 128 px route at the top of this file uses v91 small and v90 medium/large. Held-out denoising used all 128 validation images (40 small, 43 medium, 45 large). The 4x previews are for visibility; metrics use the original raster.

Unaligned metrics include placement error. Centered metrics translate the detected mask to the target centroid and isolate silhouette quality; they do not measure correct placement.

| Raw route | Size | Unaligned IoU | Centered IoU | Centroid error (px) | Unaligned boundary F1 | Area ratio |
|---|---|---:|---:|---:|---:|---:|
| v54 baseline | Small | 0.0410 | 0.4538 | 20.10 | 0.1171 | 0.659 |
| **v65** | Small | **0.6928** | **0.7281** | **0.99** | **0.8875** | **1.064** |
| v54 baseline | Medium | 0.4061 | 0.5628 | 6.95 | 0.2869 | 1.088 |
| **v69** | Medium | **0.7876** | **0.7804** | **1.27** | **0.8340** | **1.130** |
| v54 baseline | Large | 0.5030 | 0.5914 | 7.11 | 0.2719 | 1.060 |
| **v70** | Large | **0.8758** | **0.9028** | **1.16** | **0.9045** | **0.977** |

The corrected evaluator uses each size's mean held-out triangle bounding-box center, matching the preprocessing and video metadata convention. Earlier reports placed the target at the triangle's pixel centroid; this introduced an approximately 6–7 px vertical target mismatch and made their unaligned placement and edge scores invalid. Area ratio and centroid-aligned silhouette metrics were not affected by that translation error. With the corrected target, all selected routes show strong held-out alignment and boundaries; the [12-seed comparison grid](outputs/cpu_triangle_large_centroid_v70/raw_eval_12seed_bbox_centered_v2/paired_seed_1234_raw_comparison_grid.png), [per-sample metrics](outputs/cpu_triangle_large_centroid_v70/raw_eval_12seed_bbox_centered_v2/paired_generation_metrics.json), [summary](outputs/cpu_triangle_large_centroid_v70/raw_eval_12seed_bbox_centered_v2/evaluation_metrics.json), and [manifest](outputs/cpu_triangle_large_centroid_v70/raw_eval_12seed_bbox_centered_v2/run_manifest.json) record this rerun.

The selected-route held-out v-MSE is 0.05247 / 0.06003 / 0.08742 for small / medium / large, versus 0.05285 / 0.06590 / 0.09644 for v54. These denoising values are diagnostic only and do not guarantee better generated edges.

Reproduce the latest raw baseline:

```powershell
.\venv\Scripts\python.exe -m video_foundation.training.evaluate_raw_triangle_baseline `
  --data_dir data\cpu_size_conditioned_v1\val `
  --baseline_checkpoint checkpoints\cpu_size_embedding_calibrated_v54\checkpoint_final.pt `
  --small_checkpoint checkpoints\cpu_triangle_small_area_loss_v65\checkpoint_final.pt `
  --medium_large_checkpoint checkpoints\cpu_triangle_small_specialist_v59\checkpoint_step_001600.pt `
  --medium_checkpoint checkpoints\cpu_triangle_balanced_contour_v69\checkpoint_final.pt `
  --large_checkpoint checkpoints\cpu_triangle_large_centroid_v70\checkpoint_final.pt `
  --output_dir outputs\cpu_triangle_large_centroid_v70\raw_eval_12seed_bbox_centered_v2 `
  --samples 128 --sampling_steps 24 --seed 1234 `
  --generation_seeds 1234 2345 3456 4567 5678 6789 7890 8901 9012 11223 22334 33445 `
  --threads 4
```

## Priority 2: correct target geometry and specialist gates

The source video metadata and prepared stills center objects by **bounding-box center**. Upright triangles have a pixel centroid about 6–7 px below that point. The previous evaluator centered its synthetic triangle target by pixel centroid, making placement and unaligned edge metrics invalid. I changed the evaluator to derive the target bounding-box center from held-out triangle rasters by size, then center the canonical comparison mask on that target. The measured target centers are small (31.21, 31.00), medium (30.73, 31.00), and large (30.94, 31.00) pixels. Previous unaligned scores in the older v68/v69 evaluation folders are superseded; centered silhouette and area metrics were translation-invariant, but centroid error, unaligned IoU, and unaligned boundary F1 were not.

The current specialist route is v65 small, v69 medium, and v70 large. v69 adds occupancy/silhouette and contour-band losses to the ordinary v-prediction objective; it was fine-tuned from v65 for 800 CPU steps and remains unsuitable as a global checkpoint because its small-triangle area ratio is 1.543 versus 1.064 for v65. It does improve the medium route under the corrected gate. v70 was separately fine-tuned from v59 for 800 CPU steps with a low-noise centroid Huber term (weight 0.01, learning rate 5e-7; [config](configs/cpu_triangle_large_centroid_v70.yaml), [training log](outputs/cpu_triangle_large_centroid_v70/training_metrics.csv)).

| Corrected matched comparison | Metric | Previous route | Candidate route | Paired change (95% t interval) |
|---|---|---:|---:|---:|
| Medium: v59 → v69 | Unaligned IoU | 0.7669 | 0.7876 | +0.0207 [+0.0159, +0.0256] |
| Medium: v59 → v69 | Unaligned boundary F1 | 0.7895 | 0.8340 | +0.0445 [+0.0258, +0.0631] |
| Medium: v59 → v69 | Centroid error (px) | 1.19 | 1.27 | +0.08 [+0.00, +0.17] |
| Large: v59 → v70 | Unaligned IoU | 0.8508 | 0.8758 | +0.0251 [+0.0150, +0.0351] |
| Large: v59 → v70 | Unaligned boundary F1 | 0.8388 | 0.9045 | +0.0657 [+0.0380, +0.0934] |
| Large: v59 → v70 | Centroid error (px) | 1.63 | 1.16 | −0.47 [−0.68, −0.27] |

Both comparisons use 12 matched noise seeds and 24 DDIM steps. The medium [v69-v59 report](outputs/cpu_triangle_large_centroid_v70/raw_eval_12seed_v69_vs_v59_bbox_centered/paired_generation_metrics.json) and large [v70-v59 report](outputs/cpu_triangle_large_centroid_v70/raw_eval_12seed_v70_vs_v59_bbox_centered/paired_generation_metrics.json) preserve per-seed samples and provenance. Corrected selected routes against v54 are in the baseline links above. No new training is warranted until the next gate is defined against this corrected evaluator.

## Priority 3: raw-video temporal stability pilot

Per-frame raw diffusion independently denoises frame content and retains visible shape variation. I compared it with a single denoised RGB diffusion anchor translated along the same learned trajectory. The matched runs use v70, seed 2345, 72 frames, six seconds at 12 fps, 24 DDIM steps, and four CPU threads. The palette-projected version keeps the predicted mask and motion, then cleans the output colors and edges.

| Mode | Area coefficient of variation | Centroid-aligned adjacent-mask IoU | CPU generation time |
|---|---:|---:|---:|
| Per-frame raw diffusion | 0.0344 | 0.9271 | 10.03 s |
| One raw RGB diffusion anchor + translation | **0.0049** | 0.9696 | **0.87 s** |
| Palette-projected learned mask + translation | 0.0244 | **0.9716** | 1.36 s |

The raw RGB anchor reduces area variation by about 86% but retains colored background speckle. The palette-projected mode extracts the model's own largest color component, closes one-pixel gaps, fills enclosed holes, normalizes mask extent, and antialiases display upscaling. It removes background speckle without drawing a canonical triangle. In a decoded-frame check, its centered boundary F1 was 0.9787 versus 0.9754 for the raw anchor; centered IoU was 0.9095 versus 0.9130, so the contour cleanup is a small tradeoff rather than a full shape correction. Its area varies more while the translated binary mask is resampled. See the [raw-versus-cleaned frame grid](runs/raw_diffusion_fix01/raw_vs_palette_projected_large_v70.png), [cleaned six-second clip](runs/raw_diffusion_fix01/palette_projected_learned_mask_large_v70_s2345.mp4), and [video metrics](runs/raw_diffusion_fix01/palette_projected_learned_mask_large_v70_s2345.metrics.json). This is a diffusion-derived, palette-projected path; the denoised RGB anchor remains the raw-pixel result. The [matched frame grid](runs/raw_diffusion_fix01/matched_temporal_modes_v70_s2345.png), [per-frame clip](runs/raw_diffusion_fix01/matched_per_frame_large_v70_s2345_24.mp4), and [raw anchor clip](runs/raw_diffusion_fix01/matched_anchor_large_v70_s2345_24.mp4) show the temporal comparison. The sampler records foreground area variation and centroid-aligned adjacent silhouette IoU in each video metrics JSON. The [size-routed still-image preview](runs/raw_diffusion_fix01/size_routed_raw_image_smoke.png) confirms all three image routes.

Reproduce the anchor route with the size-specific raw checkpoint:

```powershell
.\venv\Scripts\python.exe -m video_foundation.inference.sample_cpu_video `
  --checkpoint runs\verification_cpu_prompt_20261118\train\checkpoint_final.pt `
  --large_spatial_checkpoint checkpoints\cpu_triangle_large_centroid_v70\checkpoint_final.pt `
  --output runs\raw_diffusion_fix01\rigid_anchor_large_v70_s2345.mp4 `
  --prompt "A large blue triangle moving left to right quickly" `
  --shape 2 --color 2 --size large --frames 72 --duration_seconds 6 --fps 12 `
  --steps 24 --threads 4 --seed 2345 --motion_mode rigid `
  --rigid_appearance denoised --motion left_to_right --speed quickly
```

For the cleaner mask-projected version, use the same arguments with these changes:

```powershell
--output runs\raw_diffusion_fix01\palette_projected_learned_mask_large_v70_s2345.mp4 `
--rigid_appearance palette --rigid_geometry learned
```

## Priority 4: shape/color gate diagnosis — still closed

The full factor-by-factor audit and experiment history is in [PILOT_FAILURE_ANALYSIS.md](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/PILOT_FAILURE_ANALYSIS.md).

This is a separate six-class experiment (red/blue/yellow × circle/triangle); never pool these scores with the raw-triangle v54/v59/v65/v69 metrics above. The previous L31 “failure” table used a detector that was not reliable on the diffusion outputs. Reclassification with the blur/palette/connected-component detector changed the point estimates, so the original decision is invalidated. On 180 matched samples the robust point estimates moved upward, but every paired 95% interval includes zero and the per-class results remain unstable. **L31 stays unpromoted and the shape/color gate stays closed.**

| L27 migrated baseline → L31 step 2,500 | Baseline | L31 | Paired change (95% interval) |
|---|---:|---:|---:|
| Joint class accuracy | 28.89% | 33.33% | +4.44 pp (−6.11, +14.44) |
| Shape accuracy | 47.22% | 51.67% | +4.44 pp (−6.67, +15.56) |
| Color accuracy | 62.78% | 67.78% | +5.00 pp (−3.33, +13.33) |

The audit found several concrete L31 defects. Circle analytic area averaged 470.54 px² while triangle area averaged 235.96 px², a twofold size confound. Masks were generated but the L31 objective trained only global epsilon MSE. Class control was weak and text conditioning was unused; the 7.7M-parameter model also took roughly an hour for the final 1,500-step interval on CPU. The run’s timestep RNG was not explicitly seeded; that was fixed in the script after the checkpoint was trained. The original detector was calibrated on clean images, then treated performance on noisy generated outputs as certain; its Gaussian-noise calibration does not fully represent structured diffusion artifacts. See the [robust audit](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l31_high_salience_shape_conditioning/eval_checkpoint_2500_seed42_n30/failure_audit_robust_metrics.json), [corrected gate decision](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l31_high_salience_shape_conditioning/eval_checkpoint_2500_seed42_n30/gate_decision.json), [provenance manifest](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l31_high_salience_shape_conditioning/eval_checkpoint_2500_seed42_n30/run_manifest.json), and [generated-sample grid](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l31_high_salience_shape_conditioning/eval_checkpoint_2500_seed42_n30/matched_generation_grid_compact.png).

### Compact CPU follow-up

L32 fixes the dataset confound: 1,200 train/300 held-out examples with analytic mean area 293.57 px² for circles and 291.21 px² for triangles. It uses a 785,283-parameter 2D U-Net without attention, explicit foreground weighting and a silhouette loss. Training took **1,112 seconds (18m32s)** on CPU with four threads. The model’s held-out noise MSE fell to 0.01187, yet its raw samples remain visibly noisy and shape-ambiguous. The aggregate MSE is not a reliable proxy for generated edge quality.

| Same 30 fixed-noise samples per class | Joint | Shape | Color | Gate outcome |
|---|---:|---:|---:|---|
| L32, corrected DDIM sampler | 41.11% | 50.56% | 81.67% | Fail; circle classes collapse |
| L32b, 400-step timestep-embedding fine-tune, same sampler | 41.11% | 48.89% | 87.78% | Fail; joint change 0.00 pp, shape −1.67 pp |
| L32c, 400-step AdaGN fine-tune, same sampler | 42.22% | 49.44% | 89.44% | Fail; color improves, shape does not |

L32b compared with L32 on the same seeds: joint 0.00 pp (95% interval −2.78 to +2.78), shape −1.67 pp (−3.89 to 0.00), and color +6.11 pp (+2.78 to +10.00). L32c compared with L32: joint +1.11 pp (−1.67 to +3.89), shape −1.11 pp (−2.78 to 0.00), and color +7.78 pp (+3.89 to +11.67). Both improve color conditioning but neither fixes circle generation: red/blue circle joint accuracy is 3.3%/0% for L32c, and yellow circle is 0%. The [L32c corrected-sampler report](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l32c_adagn_conditioning_pilot/eval_clipped_x0/L32_RESULTS.json), [fixed-seed grid](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l32c_adagn_conditioning_pilot/eval_clipped_x0/generation_grid.png), and [run manifest](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l32c_adagn_conditioning_pilot/eval_clipped_x0/run_manifest.json) preserve this comparison. The [L32b corrected-sampler report](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l32b_conditioned_timestep_pilot/eval_clipped_x0/L32_RESULTS.json) contains its paired comparison with L32.

The pilot inference code had two additional correctness bugs: it returned the slightly noised t=0 state without a final x0 reconstruction, and it never clipped intermediate x0 estimates even though the training image range is [-1, 1]. Both are now fixed in the pilot runner and step-count evaluator. Correct clipping makes color blobs less speckled but does not restore clean shape edges. The earlier 12-vs-48-step ablation used the old unclipped sampler and should not be treated as a final sampler comparison.

### L33 silhouette-head result

The L33 follow-up added a one-channel silhouette head to the compact U-Net. It retained epsilon diffusion, and trained the head with foreground-balanced BCE, Dice, and a 3×3 boundary-band loss for 400 steps (3m18s, four CPU threads). Its auxiliary loss fell from 1.256 to 0.118. The head scored well on real held-out images with mild forward noise: at t=100 and threshold 0.5 it reached 0.934 IoU / 0.998 boundary F1 over 300 images. That clean-input segmentation score did **not** transfer to generated reverse-process states.

The paired fixed-seed raw branch reached 45.56% joint / 48.89% shape / 96.11% color versus L32c’s 42.22% / 49.44% / 89.44%. The shape change was −0.56 pp (95% interval −3.33 to +1.67); this is a color gain without shape improvement. The learned mask projection at threshold 0.5 produced fragmented, hollow masks and fell to 39.44% joint / 39.44% shape / 81.11% color. Its paired shape regression was −10 pp (−15 to −5). Do not use that projection. The [L33 gate decision](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l33_auxiliary_silhouette_head_pilot/L33_GATE_DECISION.json), [raw and projected samples](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l33_auxiliary_silhouette_head_pilot/eval_threshold_050/generation_grid.png), [raw-only grid](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l33_auxiliary_silhouette_head_pilot/eval_threshold_050/generation_grid_raw.png), and [held-out mask-head audit](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l33_auxiliary_silhouette_head_pilot/eval_mask_head_validation/MASK_HEAD_VALIDATION_AUDIT.json) capture the result.

This pins down an additional failure factor: **reverse-process distribution shift**. The mask head learned to segment real examples with forward-process noise, but the denoiser’s generated t=0 states do not resemble those inputs closely enough for that segmentation to survive. Lower validation MSE and lower head loss therefore do not demonstrate successful generation. L33 is not promoted.

**Historical decision before v90:** the raw RGB diffusion shape/color gate remained closed for the then-tested general shape/color models. The v90 triangle-specific result above improves that triangle route; it does not resolve the separate general shape/color failures described in this historical section. The L34 structured-geometry route remains renderer-backed and its results must not be presented as raw diffusion quality.

### L34 structured-geometry route

L34 replaces pixel generation with a 71,085-parameter CPU MLP diffusion model over center, extent, and triangle rotation, then uses a supersampled CPU primitive renderer for visible pixels. Three independently trained seeds produced 1,800 samples. Requested shape and color are exact by renderer construction, all decoded geometry stayed in the training support, and no object was cropped. Average time to sample and render 600 outputs was **1.68 seconds** on four CPU threads. This clears the renderer-backed visual shape gate only.

The learned geometry distribution is still approximate: worst per-feature held-out KS distance was **0.32** across the three seeds. This is not a raw RGB diffusion improvement and does not establish that the denoiser itself learned sharp edges. See the [multi-seed summary](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l34_structured_geometry_diffusion/L34_MULTI_SEED_SUMMARY.json), [L34 gate decision](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l34_structured_geometry_diffusion/L34_GATE_DECISION.json), and [sample grid](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l34_structured_geometry_diffusion/generation_grid.png).

### L35 event-aware collision pilot

L35 samples circle radii and colors from the L34 EMA checkpoint, rasterizes them with the CPU renderer, and advances them with an exact event-driven equal-mass 1D collision and wall-reflection rule. Events carry timestamps, types, and object IDs. Across 12 held-out seeds, every clip had exactly one circle collision, no sampled overlap, and no wall-bound violation; maximum collision-time error was **2.22e-15 seconds**. The six-second output is 72 frames at 12 fps, 162×64, and completed in **0.64 seconds total** on CPU with four threads, including checkpoint load, geometry sampling, rendering, GIF encoding, and the 12-case physics evaluation.

This passes the structured physics/rendering gate, not a learned-motion gate: only static geometry is sampled from learned weights; collision timing and trajectories are explicitly computed by physics. The equal-mass assumption and horizontal motion are deliberate scope limits. The [six-second collision GIF](outputs/cpu_event_motion_l35/two_red_circles_6s.gif), [contact-frame preview](outputs/cpu_event_motion_l35/two_red_circles_6s_contact.png), [metrics/provenance](outputs/cpu_event_motion_l35/two_red_circles_6s.json), and [L35 gate decision](outputs/cpu_event_motion_l35/L35_GATE_DECISION.json) preserve the run.

### L36 train-only geometry quantile calibration

L36 fits a class-conditioned empirical quantile map from **training geometry only** and independent L34 samples, then evaluates it against the held-out split. Across L34 model seeds 42, 123, and 777 (100 generated samples/class/seed), mean per-feature KS fell from **0.1873 to 0.1440** (23.1% lower). The worst feature KS also fell on each seed: **0.35→0.33**, **0.27→0.23**, and **0.40→0.28**. All calibrated centers and extents remained in training support. The three-seed calibration and evaluation took about **1.08 seconds** on this four-thread CPU run.

This did not retrain or alter L34 weights. It adjusts sample marginals, and some individual class-feature distances regress; the worst held-out distance is still 0.33. Therefore it is available as an **optional** sampler, not the default. The [L36 report](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l34_structured_geometry_diffusion/l36_quantile_calibration/L36_RESULTS.json), [gate decision](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l34_structured_geometry_diffusion/l36_quantile_calibration/L36_GATE_DECISION.json), [same-noise visual comparison](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l34_structured_geometry_diffusion/l36_quantile_calibration/L36_raw_vs_calibrated_grid.png), and [fitted tables](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l34_structured_geometry_diffusion/l36_quantile_calibration/L36_CALIBRATION_TABLES.json) are saved. L35 can use it with `--geometry-calibration` while keeping the renderer and event physics unchanged. The optional calibrated collision sample took **0.63 seconds** total and passed the same 12-case collision checks; see [calibrated video](outputs/cpu_event_motion_l35/two_red_circles_6s_calibrated.gif) and [metrics](outputs/cpu_event_motion_l35/two_red_circles_6s_calibrated.json).

### L37 joint geometry comparison

The L32 procedural generator draws center, latent size, and triangle rotation independently. I compared the learned L34 output, L36 marginal calibration, and a factorized empirical sampler fitted only to the 1,200 training rows. Across the same three L34 checkpoints, the factorized sampler had the lowest mean held-out feature KS (**0.1357**), joint energy distance (**0.05417**), and correlation-matrix error (**0.1398**). It beat L36 on joint energy distance for all three seeds. No neural weights changed; this sampler draws independent geometry attributes from training-split marginals and should only be used for this procedural curriculum.

The factorized option is available in [sample_geometry_collision.py](video_foundation/inference/sample_geometry_collision.py) as `--geometry-sampler factorized_train_prior`. It skips checkpoint loading and generated a six-second 72-frame clip in **0.32 seconds** on CPU; all 12 seeded motion cases still had one collision, no overlap, and no wall escape. See the [L37 report](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l34_structured_geometry_diffusion/l37_joint_geometry/L37_RESULTS.json), [gate decision](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l34_structured_geometry_diffusion/l37_joint_geometry/L37_GATE_DECISION.json), [comparison grid](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l34_structured_geometry_diffusion/l37_joint_geometry/L37_joint_geometry_comparison.png), [factorized collision video](outputs/cpu_event_motion_l35/two_red_circles_6s_factorized.gif), and [metrics](outputs/cpu_event_motion_l35/two_red_circles_6s_factorized.json).

### L38 learned speed-style variation

L38 connects the previously trained v37 `motion_velocity` table to the L35 physics renderer. Its learned `slowly` and `quickly` rows select different safe collision tempos; the event solver still calculates exact impact, rebound, and wall events. At 12 fps, the first impact is **2.33 seconds** for slow and **2.08 seconds** for quick, with applied speeds of **21.86** and **24.48 px/s**. Across 12 geometry seeds per style, every clip had exactly one circle collision, no overlap, and no wall escape. Both six-second 72-frame clips rendered in under **0.76 seconds** on CPU.

The learned checkpoint's raw speed ratio is compressed into the range that preserves a visible single collision and rebound in six seconds. This is learned tempo selection plus explicit physics, not a neural trajectory generator or raw RGB diffusion. No weights were retrained. The [slow clip](outputs/cpu_event_motion_l38/two_red_circles_6s_slow.gif), [quick clip](outputs/cpu_event_motion_l38/two_red_circles_6s_quick.gif), [per-run metrics](outputs/cpu_event_motion_l38/two_red_circles_6s_slow.json), [gate decision](outputs/cpu_event_motion_l38/L38_GATE_DECISION.json), and the quick [metrics](outputs/cpu_event_motion_l38/two_red_circles_6s_quick.json) preserve the results.

### L39 learned acceleration/deceleration profiles

L39 adds a 1,345-parameter CPU MLP that predicts a bounded positive speed scale over normalized time. Three models were trained on independently generated synthetic analytical profiles (1,800 train and 400 validation profiles/model, 32 points/profile, 3,000 steps). Best held-out RMSEs were **0.05855**, **0.05724**, and **0.05702**, versus constant-speed baselines **0.09870**, **0.09953**, and **0.09528**. Mean RMSE reduction is **41.1%**. Combined profile training took about **22 seconds** on four CPU threads.

The learned speed curves feed the exact event solver; positions and collision responses are not learned. Across **600** scenarios (100 each style/model seed), all had one circle collision, no overlap, and no wall escape. Maximum impact-time error was below 6e-15 seconds. Mean within-clip speed variation was 6.3% (coefficient of variation). The slow/quick six-second demo clips rendered in **0.48/0.60 seconds** on CPU, including the profile model, sprite sampling, and GIF save. This is a renderer-backed motion pilot trained from a synthetic teacher, not a learned-from-video motion result, and it does not pass the raw RGB video gate.

See the [L39 profile trainer](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l39_learned_speed_profiles/train_speed_profile.py), [robustness evaluator](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l39_learned_speed_profiles/evaluate_speed_profile_robustness.py), [three-seed gate](experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l39_learned_speed_profiles/L39_MULTI_SEED_GATE.json), and [motion gate decision](outputs/cpu_event_motion_l39/L39_GATE_DECISION.json). The [slow profile clip](outputs/cpu_event_motion_l39/two_red_circles_6s_slow_profile.gif) and [quick profile clip](outputs/cpu_event_motion_l39/two_red_circles_6s_quick_profile.gif) show the result; per-run metrics preserve profile scales and event logs.

## Keep generation paths separate

- **Raw diffusion:** learned 64x64 pixels; measured by the baseline above.
- **Palette-projected learned-mask video:** diffusion supplies the appearance and contour; CPU mask cleanup removes speckle and translates the learned silhouette. It does not draw a canonical primitive.
- **Direct image renderer:** `sample_image.py --refine_geometry`; exact requested shape/color, no diffusion output.
- **Model-guided image hybrid:** `sample_image.py --model_guided_geometry`; diffusion estimates placement and the renderer supplies crisp shape edges.
- **Renderer-assisted video:** the six-second clip below renders canonical geometry and applies checkpoint-predicted motion offsets; it skips pixel diffusion. Do not compare its edge quality with raw diffusion.
- **Crisp triangle reference:** [six-second 512x512 MP4](runs/raw_diffusion_fix01/crisp_renderer_assisted_large_blue_triangle_s2345.mp4), [frame grid](runs/raw_diffusion_fix01/crisp_renderer_assisted_triangle_contact.png), and [provenance metrics](runs/raw_diffusion_fix01/crisp_renderer_assisted_large_blue_triangle_s2345.metrics.json). It uses antialiased CPU triangle rasterization and checkpoint-predicted motion, runs in 3.44 seconds, and skips pixel diffusion.
- **L34/L35 structured geometry video:** L34 samples shape geometry; an explicit CPU primitive renderer and event-driven physics produce the L35 circle-collision clip. It skips raw pixel diffusion and has no learned motion component.

## Current raw triangle routing

Raw triangle generation now has size-specific measured checkpoints:

- Small: `checkpoints/cpu_triangle_small_area_loss_v65/checkpoint_final.pt`
- Medium: `checkpoints/cpu_triangle_balanced_contour_v69/checkpoint_final.pt`
- Large: `checkpoints/cpu_triangle_large_centroid_v70/checkpoint_final.pt`
- Other shapes or requests without specialist arguments: calibrated v54 base checkpoint

Pass `--small_triangle_checkpoint`, `--medium_triangle_checkpoint`, and `--large_triangle_checkpoint` to `sample_image.py` to activate these routes. The medium/large flags were added because the prior image sampler could only select a shared v59 checkpoint for both sizes; this was an inference routing limitation, not a weight or architecture incompatibility. The [historical four-checkpoint small comparison](outputs/cpu_triangle_routed_v65_small/seed1234_comparison.png) shows same-seed 64x64 samples; the images are enlarged with nearest-neighbor scaling for visibility.

The raw output remains visually soft because its learned image grid is only 64x64. For a crisp diagram while retaining learned placement, use the new `--model_guided_geometry` mode: diffusion predicts the foreground location at checkpoint resolution, then the CPU renderer creates an antialiased canonical shape at 256x256. This is a hybrid path, not raw pixel diffusion. [Raw versus model-guided output](outputs/cpu_triangle_model_guided_v67_small/raw_vs_hybrid.png).

## Selected small-shape result: 12 paired seeds

The selected small-triangle checkpoint was compared directly against its v64 parent with the same 12 noise seeds. Each score is the mean over one generated sample per seed. Centered IoU and boundary F1 translate the detected mask to the target centroid to isolate silhouette from placement; boundary F1 allows a one-pixel tolerance. The area ratio is raw foreground area divided by target area.

| Small-triangle checkpoint | Centered IoU | Boundary F1 | Foreground area ratio |
|---|---:|---:|---:|
| v64 parent | 0.7194 ± 0.0509 | 0.9216 ± 0.0685 | 1.282 ± 0.156 |
| **v65 selected** | **0.7619 ± 0.0413** | **0.9643 ± 0.0273** | **1.051 ± 0.145** |

That is a **0.043 IoU** increase, **0.043 boundary-F1** increase, and area much closer to target. The selected sample is also [available as a 4x nearest-neighbor preview](outputs/cpu_triangle_routed_v65_small/sample_0000_4x.png).

The earlier four-seed model-selection history is:

| Small-triangle checkpoint | Centered IoU | Boundary F1 | Area ratio |
|---|---:|---:|---:|
| v54 calibrated baseline | 0.5499 ± 0.1357 | 0.7404 ± 0.1426 | 0.832 |
| v59 specialist | 0.5313 ± 0.0259 | 0.5489 ± 0.1028 | 1.881 |
| v62 edge specialist, step 600 | 0.6486 ± 0.0157 | 0.8163 ± 0.0498 | 1.509 |
| v63 stronger structure, final | 0.6617 ± 0.0181 | 0.7999 ± 0.0471 | 1.485 |
| v64 isolated area loss, final | 0.6780 ± 0.0224 | 0.8456 ± 0.0583 | 1.427 |

The v59 medium/large scores **0.7878 IoU / 0.7889 boundary F1 / 1.134 area ratio** for medium and **0.8776 / 0.8844 / 0.946** for large on the original four-seed comparison; these are historical checkpoint-selection metrics. Current route assignments and the matched 12-seed results are in the refreshed raw baseline above.

## Training steps

- **v62 edge pass:** 800 CPU steps, learning rate 2e-6, 64 small-triangle training examples balanced across four colors, structure weight 0.3, edge-gradient weight 0.25. Step 600 had the best edge F1 among saved checkpoints.
- **v63 structure pass:** 400 CPU steps with stronger total structure weighting. It slightly improved area and IoU but reduced boundary F1, so it was not selected.
- **v64 area pass:** 400 CPU steps from v62 step 600. Dice and edge weights remained fixed; the structure-area coefficient increased from 0.25 to 0.75. It improved all three metrics over v62.
- **v65 area pass:** 400 CPU steps from v64, with the area coefficient increased to 1.5 and learning rate reduced to 5e-7. Its 12-seed result improved contour, overlap, and area together, so it is selected for small triangles.

v65 is a focused small-shape specialist. Its small held-out denoising v-MSE rose from 0.048234 on v64 to 0.052472 on v65 while generated geometry improved. Keep v64 as a fallback and continue monitoring this tradeoff. On this machine, the optimization loops took about 2m21s for v62 and 1m42–1m50s for each 400-step pass, with a little extra time for sample generation and checkpoint I/O. All training used CPU with four PyTorch threads.

## Reproducible artifacts

- v54 versus v59, four paired seeds: [metrics](outputs/paired_edge_baseline_v61_v54_vs_v59/paired_generation_metrics.json), [manifest](outputs/paired_edge_baseline_v61_v54_vs_v59/run_manifest.json), [sample grid](outputs/paired_edge_baseline_v61_v54_vs_v59/paired_seed_1234.png)
- v62 step 600: [metrics](outputs/cpu_triangle_small_edge_v62_eval_s600/paired_generation_metrics.json), [manifest](outputs/cpu_triangle_small_edge_v62_eval_s600/run_manifest.json)
- v64 versus v65, 12 paired seeds: [metrics](outputs/cpu_triangle_small_area_loss_v65_eval_12seed/paired_generation_metrics.json), [manifest](outputs/cpu_triangle_small_area_loss_v65_eval_12seed/run_manifest.json)
- v65 training log: [training_metrics.csv](outputs/cpu_triangle_small_area_loss_v65/training_metrics.csv)
- Selected v65 training settings: [cpu_triangle_small_area_loss_v65.yaml](configs/cpu_triangle_small_area_loss_v65.yaml)

Evaluation manifests record checkpoint and embedded-config hashes, validation-directory content hash, arguments, CPU/software details, Git revision and dirty state, and hashes of the model, diffusion, evaluation, rendering, and training source files. Evaluations save paired per-seed geometry metrics and one-pixel boundary precision/recall/F1.

Reproduce the 12-seed v64/v65 comparison with:

```powershell
.\venv\Scripts\python.exe -m video_foundation.training.evaluate_size_conditioned `
  --data_dir data\cpu_size_conditioned_v1\val `
  --baseline_checkpoint checkpoints\cpu_triangle_small_area_loss_v64\checkpoint_final.pt `
  --size_checkpoint checkpoints\cpu_triangle_small_area_loss_v65\checkpoint_final.pt `
  --output_dir outputs\cpu_triangle_small_area_loss_v65_eval_12seed `
  --samples 128 --sampling_steps 24 --seed 1234 `
  --generation_seeds 1234 2345 3456 4567 5678 6789 7890 8901 9012 11223 22334 33445
```

Generate a raw small blue triangle with the selected routes using:

```powershell
.\venv\Scripts\python.exe -m video_foundation.inference.sample_image `
  --checkpoint checkpoints\cpu_size_embedding_calibrated_v54\checkpoint_final.pt `
  --small_triangle_checkpoint checkpoints\cpu_triangle_small_area_loss_v65\checkpoint_final.pt `
  --medium_triangle_checkpoint checkpoints\cpu_triangle_balanced_contour_v69\checkpoint_final.pt `
  --large_triangle_checkpoint checkpoints\cpu_triangle_large_centroid_v70\checkpoint_final.pt `
  --output_dir outputs\cpu_triangle_routed_v65_small `
  --num_samples 1 --sampling_steps 24 --seed 1234 `
  --class_label 2 --color_label 2 --size small --raw_generation
```

For the crisp model-guided diagram, replace `--raw_generation` with `--model_guided_geometry`. Keep `--output_image_size 256` (the default) for the supersampled render.

## Separate generation paths and six-second CPU reference

The paths have different quality/runtime meanings and their metrics must not be pooled:

- **Raw image diffusion:** learned 64x64 RGB output, evaluated in the current 12-seed table above.
- **Direct image renderer:** exact CPU rasterization of requested shape/color/size; the denoiser is skipped. It is a deterministic renderer reference, not a learned image sample.
- **Model-guided image hybrid:** raw diffusion predicts location and an antialiased CPU renderer supplies the requested primitive. See [the v67 raw-versus-hybrid image](outputs/cpu_triangle_model_guided_v67_small/raw_vs_hybrid.png).
- **Six-second renderer-assisted video:** a canonical CPU shape renderer draws frames and a checkpoint predicts motion offsets. The diffusion pixel sampler is skipped in this mode.

The older v60 clip reports **3.23 seconds** but did not preserve which checkpoint generated it. A fresh, fully identified v68 reference now records a six-second, 72-frame medium blue triangle clip at 64x64 in **3.18 seconds on CPU with four threads**. Its mode is `direct_geometry_video`, it used 0 diffusion steps, and its renderer-to-applied-offset residual was 0.27 px with no cropping. The motion offsets come from checkpoint v37 step 8,500; the visible shape and color come from canonical rendering. This is not raw pixel-video generation.

- [v68 six-second triangle GIF](outputs/cpu_video_direct_geometry_v68/triangle_medium_6s.gif)
- [Video metrics](outputs/cpu_video_direct_geometry_v68/triangle_medium_6s.metrics.json)
- [Video provenance manifest](outputs/cpu_video_direct_geometry_v68/triangle_medium_6s.metrics_manifest.json), including checkpoint/config hashes and full config

Reproduce that CPU video reference with:

```powershell
.\venv\Scripts\python.exe -m video_foundation.inference.sample_cpu_video `
  --checkpoint checkpoints\cpu_video_canonical_position_aug24_v37\checkpoint_step_008500.pt `
  --output outputs\cpu_video_direct_geometry_v68\triangle_medium_6s.gif `
  --shape 2 --color 2 --size medium --duration_seconds 6 --fps 12 `
  --threads 4 --seed 7 --motion_mode rigid --rigid_appearance palette `
  --rigid_geometry canonical --motion left_to_right --speed quickly
```

The existing v60 report remains available as a historical runtime result, but it should not be used for checkpoint-quality comparisons because it lacks checkpoint provenance: [v60 metrics](outputs/cpu_video_direct_geometry_v60/triangle_medium_6s_metrics.json).

The remaining image limitation is 64x64 raw diffusion softness; the 4x preview does not add resolution. Continue using boundary F1, IoU, area ratio, paired seeds, and the separate video-runtime gate for the next iteration.

## Latest raw RGB triangle investigation

The four-sample consensus remains rejected because it reduced background speckle by softening the triangle edges. The selected raw route remains 64px v70 (large centered IoU 0.903, boundary F1 0.905 in its recorded evaluation); none of the 128px candidates below is promoted.

The v81 128px fine-tune used learning rate 1e-7 for 2,000 steps and changed only 0.12% of the starting weights. It also filtered to large triangles, so the other size labels were not trained. v82 used all triangle size labels at 1e-6, but its training data normalized large triangles only. Its 12-seed centered IoU was 0.342/0.464/0.604 for small/medium/large; area ratios were 2.748/1.764/1.192. Its RGB preview remained visibly malformed.

An all-size canonicalization path was then added to `canonicalize_size_conditioned_data.py`. v83 trained for 1,500 steps on 128px triangles with the same 1e-6 rate. A data audit confirmed that all labeled triangle extents now match their intended 30/58/84px targets, with 64/71/48 training examples per size. Its 12-seed raw RGB results were:

- Small: centered IoU 0.498, unaligned boundary F1 0.080, area ratio 1.936.
- Medium: centered IoU 0.483, unaligned boundary F1 0.129, area ratio 1.508.
- Large: centered IoU 0.600, unaligned boundary F1 0.246, area ratio 1.081.

This improves the size targets and large-edge score over v82, but the generated triangle still has a broken, lumpy contour. The preview is [here](outputs/cpu_triangle_large_raw_rgb_v83_128_all_sizes_canonical/raw_eval_12seed/candidate_seed_2345_large_preview_4x.png); the metrics and provenance are in [evaluation_metrics.json](outputs/cpu_triangle_large_raw_rgb_v83_128_all_sizes_canonical/raw_eval_12seed/evaluation_metrics.json). v83 completed 1,500 steps in 18m03s on CPU with four threads and is not promoted.

The next learning-rate test, v84 at 5e-6, visibly regressed by step 750 and was stopped at step 806. A 12-seed evaluation of the saved step-500 checkpoint was effectively tied with v83 (centered IoU 0.500/0.477/0.600; area ratio 2.018/1.504/1.099), while its held-out v-MSE was slightly worse at all sizes. Do not resume/promote it. Its incomplete training checkpoint and diagnostics are under `checkpoints/cpu_triangle_large_raw_rgb_v84_128_lr5e6` and `outputs/cpu_triangle_large_raw_rgb_v84_128_lr5e6`.

The data mismatch is fixed, but the raw 128px denoiser still cannot produce a reliable triangle boundary. The next experiment should target model capacity/receptive field or the diffusion representation, not more loss-weight tuning. Keep raw diffusion samples, direct renders, and renderer-assisted videos separate in all reports. The relevant configs are [v82](configs/cpu_triangle_large_raw_rgb_v82_128_all_sizes.yaml), [v83](configs/cpu_triangle_large_raw_rgb_v83_128_all_sizes_canonical.yaml), and [v84](configs/cpu_triangle_large_raw_rgb_v84_128_lr5e6.yaml).

### Receptive-field experiment: v85/v86

Added an optional dilated middle residual block (`middle_dilation`) with dilation 2. It preserves all parameter tensor shapes and the 224,579-parameter footprint, so a v83 checkpoint can warm-start it without attention or a wider CPU model. Raw sampling and CPU-video model constructors now read this field from checkpoint configs.

The v85 first 500-step checkpoint improved 12-seed centered IoU over v83 from 0.498 to 0.455 small, 0.483 to 0.599 medium, and 0.600 to 0.699 large. Large unaligned boundary F1 rose from 0.246 to 0.284, and held-out v-MSE decreased at all sizes. Small shapes remain oversized (mean area ratio 2.153), and the raw preview still has a secondary top lobe. Later v85 training preview regressed, so only the measured step-500 checkpoint was retained for refinement.

The v86 500-step low-rate refinement retained the medium/large gains: centered IoU 0.469 small / 0.601 medium / 0.702 large; large area ratio 1.051 and centered boundary F1 0.219. Unaligned boundary F1 was 0.141 / 0.191 / 0.266. The sample remains visibly imperfect, especially on small and medium triangles, so v86 is experimental and not yet a dependable raw-triangle solution. Its 12-seed report is [here](outputs/cpu_triangle_large_raw_rgb_v86_128_dilated_refine/raw_eval_12seed/evaluation_metrics.json), with a raw RGB preview [here](outputs/cpu_triangle_large_raw_rgb_v86_128_dilated_refine/raw_eval_12seed/candidate_seed_2345_large_preview_4x.png). Do not describe the renderer-assisted six-second clip benchmark as a raw-diffusion video result.

### Six-second raw-diffusion video comparison

For an apples-to-apples runtime and visual reference, the video pipeline was run with the selected 64px v70 raw triangle prior and the v37 motion checkpoint. The generated clip uses one 24-step raw RGB diffusion anchor and rigidly translates that learned anchor for 72 frames; no canonical shape renderer or mask projection is used. At 512x512, encoding the six-second clip took 7.38 seconds on CPU with four threads. Added `--display_scale` to control the final enlargement without changing the 64px diffusion sample. The matched 384x384 run completed in 5.71 seconds, including 0.81 seconds of diffusion sampling, with 0.002% mean foreground crop and 0.970 centroid-aligned adjacent-mask IoU. This meets the observed under-six-second target on this machine, but the margin is small and the 512px preset does not. The v86 128px anchor also completed at 384x384 in 4.21 seconds, but its split apex makes it the lower-quality choice despite its faster render; see its [video](runs/raw_diffusion_fix01/raw_rgb_triangle_large_v86_seed2345_6s72f_384.mp4) and [metrics](runs/raw_diffusion_fix01/raw_rgb_triangle_large_v86_seed2345_6s72f_384.metrics.json).

The v70 output has a single triangular apex and is visibly cleaner than the v86 128px sample's split apex. Fine edge speckles remain, so this is a better raw result rather than a final quality fix. The v70 12-seed large-triangle evaluation measured centered IoU 0.903, area ratio 0.977, and centered one-pixel boundary F1 0.944; its unaligned boundary F1 is only 0.147, reflecting localization error as well as contour quality. Keep those metrics distinct from temporal video metrics.

- [v70 six-second raw-diffusion video, 384x384](runs/raw_diffusion_fix02/raw_rgb_triangle_large_v70_seed2345_6s72f_384.mp4)
- [Video metrics](runs/raw_diffusion_fix02/raw_rgb_triangle_large_v70_seed2345_6s72f_384.metrics.json) and [provenance manifest](runs/raw_diffusion_fix02/raw_rgb_triangle_large_v70_seed2345_6s72f_384.manifest.json)
- [Three-frame preview](runs/raw_diffusion_fix02/raw_rgb_triangle_large_v70_contact_384.png)
- [v70 paired-seed raw image evaluation](outputs/cpu_triangle_large_centroid_v70/raw_eval_12seed/evaluation_metrics.json)

Reproduce with `--spatial_checkpoint checkpoints/cpu_triangle_large_centroid_v70/checkpoint_final.pt --display_scale 6 --steps 24 --threads 4`, using the v37 motion checkpoint and the same prompt labels recorded in the manifest. The default 512px scale remains available by omitting `--display_scale`; runtime depends on output size, CPU, thread count, and encoder load.

### Follow-up for the soft triangle output

The 64px anchor looked softer after video export because the rigid denoised path enlarged it with Lanczos and shifted it with bicubic interpolation. Added `--display_resampling` so raw anchors can use nearest-neighbor for both steps. This preserves the model's pixel boundaries; it does not alter or repair the diffusion sample. A same-seed comparison showed a more distinct triangle outline with nearest-neighbor, at the cost of visibly stair-stepped edges.

- [Seed 5678 raw-diffusion video, 384x384, nearest-neighbor](runs/raw_diffusion_fix04/raw_rgb_triangle_large_v70_seed5678_6s72f_384_nearest.mp4)
- [Three-frame preview](runs/raw_diffusion_fix04/raw_rgb_triangle_large_v70_contact_nearest.png)
- [Video metrics](runs/raw_diffusion_fix04/raw_rgb_triangle_large_v70_seed5678_6s72f_384_nearest.metrics.json) and [provenance manifest](runs/raw_diffusion_fix04/raw_rgb_triangle_large_v70_seed5678_6s72f_384_nearest.manifest.json)

The clip is 6 seconds, 72 frames at 12fps, 384x384, and took 1.02 seconds on CPU with four threads in this run. Foreground crop was 0 and centroid-aligned adjacent-mask IoU was 0.971. The selected seed was the strongest of the prior 12-seed v70 large-triangle evaluation (centered IoU 0.930, centered boundary F1 0.981); this is seed selection from the recorded evaluation set, not an average or a renderer-backed shape.

Two additional learned-detail experiments did not fix the native-128 raw shape and are not promoted. v87 transferred 64px v70 weights to a 128px model and trained only the identity-initialized 4,976-parameter refinement block for 1,000 steps; large centered IoU moved 0.587 to 0.592 and boundary F1 0.127 to 0.130, while v-MSE slightly worsened. v88 started from native-128 v86, enabled the detail block, and fine-tuned all weights for 1,000 steps. Against v86 on the same 12 seeds, large IoU changed 0.706 to 0.708 and boundary F1 0.263 to 0.277, but area ratio worsened 1.038 to 1.111 and preview samples remained lumpy. Training took 5m20s on CPU/4 threads; the paired evaluation took 45.77s. Neither passes the visual quality gate.

Reproduce the sharper export with `--display_scale 6 --display_resampling nearest --steps 24 --threads 4`. The native image model still needs a real resolution/shape-quality improvement; nearest-neighbor only makes the current 64px output easier to see.
