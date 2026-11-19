# CPU-First Local Video Foundation

This repository is a small research model for generating simple colored shapes and short clips on a CPU. The current strongest raw triangle route uses **v93 for small triangles** and **v90 for medium and large triangles**. The spatial model is raw RGB diffusion; the six-second video path samples one diffusion anchor and moves it with a separate compact CPU motion model.

This is a closed-curriculum model, not general text-to-video. Supported shapes are circles, squares, and triangles; supported colors are red, green, blue, and yellow; sizes are small, medium, and large. Motion prompts map to the supported synthetic directions and speeds.

## Current results and limits

- Spatial model: compact 2D U-Net, 128×128 RGB, 224,579 parameters. It has no VAE, 3D convolution, or broad temporal attention.
- v93 is the current small-triangle specialist. Against v92 on 12 paired seeds, centered IoU rose **0.757→0.874** (10/12 wins), one-pixel boundary F1 **0.666→0.970** (11/12), and area ratio **1.332→0.998** (closer to target on 11/12). Raw unaligned IoU changed **0.625→0.621** (5/12 wins); its paired bootstrap interval includes zero, so the small mean decrease is inconclusive.
- Across v90→v91→v92→v93, small centered IoU moved **0.625→0.718→0.757→0.874**, boundary F1 **0.267→0.561→0.666→0.970**, and area ratio **1.605→1.401→1.332→0.998**. The latest pass aligned the training foreground threshold with the evaluator’s blue-pixel threshold and steepened the occupancy sigmoid so black stays near zero.
- Keep v90 for medium and large triangles; v93 is only evaluated as a small-size specialist. Its small held-out v-MSE improved from 0.01801 to 0.01536, matching the generated-quality gain.
- A 72-frame, six-second 384×384 small-triangle clip took **1.03 seconds** on the recorded Windows CPU with four threads. The large-triangle clip took **2.82 seconds**. Times depend on CPU, thread settings, and video encoding.
- The video path uses rigid translation of a single raw diffusion anchor. It is not independent diffusion of every frame. Do not describe renderer-assisted outputs as raw diffusion.

Detailed measurements and decisions are in [CURRENT_MODEL_STATUS.md](CURRENT_MODEL_STATUS.md). After extraction, the bundle's per-size means, standard deviations, checkpoint hashes, and evaluation notes are in `provenance/routed_metrics.json`.

After extracting the bundle, preview the packaged raw outputs in [the four-color small-triangle grid](examples/small_triangle_raw_diffusion_grid.png), [the six-second small-triangle clip](examples/small_triangle_raw_anchor_6s.mp4), and [its generation metrics](examples/small_triangle_raw_anchor_metrics.json).

## Download and install

The compact model bundle is [release/vidzy_cpu_raw_triangle_bundle.zip](release/vidzy_cpu_raw_triangle_bundle.zip). It contains v93 and v90 inference checkpoints, v92/v91 fine-tuning ancestors, the v86 initializer, the v37 motion checkpoint, configs, example outputs, routed metrics, and SHA-256 hashes. Optimizer state is omitted to keep the download small; model weights, configs, and training step metadata are preserved. The bundle is built locally by `python tools/build_cpu_model_bundle.py`; a source-only Git checkout does not include weights, so a published source release should attach this ZIP and its `.sha256` sidecar as separate release assets.

If the bundle is downloaded separately from the source repository, put it at `release/vidzy_cpu_raw_triangle_bundle.zip` and extract it from the repository root. The archive recreates the `checkpoints/`, `configs/`, `examples/`, and `provenance/` paths.

```powershell
python tools\verify_cpu_model_bundle.py
Expand-Archive -LiteralPath .\release\vidzy_cpu_raw_triangle_bundle.zip -DestinationPath . -Force
Get-FileHash .\release\vidzy_cpu_raw_triangle_bundle.zip -Algorithm SHA256
Get-Content .\release\vidzy_cpu_raw_triangle_bundle.sha256
```

The verifier checks the ZIP checksum and every bundled file against `model_bundle_manifest.json` before extraction. The archive's internal manifest also lists a SHA-256 digest for each model checkpoint, config, metric, and example.

Use Python 3.11 or 3.12 and PowerShell. Create an isolated environment, install a CPU build of PyTorch, then install the project dependencies and commands:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On the [official PyTorch install selector](https://pytorch.org/get-started/locally/), select Windows, Pip, Python, and CPU, and run the displayed command. Then, from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The last command should report `CUDA available: False` for a CPU-only PyTorch build. Image sampling and image training also accept `--device cpu`; the video generator runs on CPU and defaults to four threads.

## First run: generate a raw diffusion image

From the repository root, this makes four raw small blue/red/green/yellow triangle samples using the v93 specialist route. Output images and a contact sheet are written to `outputs/first_run_small_triangles/`.

```powershell
python -m video_foundation.inference.sample_image `
  --checkpoint checkpoints\cpu_triangle_raw_rgb_v90_native_centered_lr5e5\checkpoint_final.pt `
  --small_triangle_checkpoint checkpoints\cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold\checkpoint_final.pt `
  --medium_large_checkpoint checkpoints\cpu_triangle_raw_rgb_v90_native_centered_lr5e5\checkpoint_final.pt `
  --output_dir outputs\first_run_small_triangles `
  --num_samples 4 --sampling_steps 50 --seed 5678 `
  --class_label 2 --size small --vary_colors --raw_generation --device cpu
```

Shape labels are `0=circle`, `1=square`, `2=triangle`; color labels are `0=red`, `1=green`, `2=blue`, `3=yellow`. Size labels are `small`, `medium`, and `large`. Use `--size medium` or `--size large` to sample the v90 route. The output stays in the raw diffusion path; options that request canonical geometry or renderer refinement are not raw diffusion.

## Generate a six-second CPU video

This makes a six-second small blue triangle clip. v93 supplies the small raw spatial anchor, v90 is the medium/large spatial fallback, and v37 supplies motion. `--rigid_appearance denoised` and `--rigid_geometry learned` preserve the learned/raw appearance and silhouette.

```powershell
python -m video_foundation.inference.sample_cpu_video `
  --checkpoint checkpoints\cpu_video_v37_motion_v90_spatial_composite.pt `
  --small_spatial_checkpoint checkpoints\cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold\checkpoint_final.pt `
  --medium_large_spatial_checkpoint checkpoints\cpu_triangle_raw_rgb_v90_native_centered_lr5e5\checkpoint_final.pt `
  --output outputs\first_run_small_triangle_6s.mp4 `
  --shape 2 --color 2 --size small `
  --duration_seconds 6 --fps 12 --display_scale 3 `
  --display_resampling nearest --steps 24 --threads 4 --seed 2345 `
  --motion_mode rigid --rigid_appearance denoised --rigid_geometry learned `
  --motion left_to_right --speed quickly `
  --metrics_json outputs\first_run_small_triangle_6s.metrics.json
```

This outputs 72 frames at 384×384. The metrics JSON records frame count, elapsed CPU time, and temporal-shape measurements; a sibling manifest records checkpoint hashes and generation settings. Change shape, color, size, motion, and speed to select other supported conditions. The generator supports clips up to six seconds.

## Continue training or fine-tune

The included weights are ready for inference. To repeat the v90 training run, first generate the deterministic antialiased triangle dataset, then fine-tune from the included v86 initializer:

```powershell
python -m video_foundation.training.generate_native_triangle_data `
  --output_dir data\cpu_native_centered_antialiased_triangles_128_v1 `
  --image_size 128 --train_per_combo 256 --val_per_combo 64 `
  --seed 88090 --supersample 8

python -m video_foundation.training.train_image `
  --config configs\cpu_triangle_raw_rgb_v90_native_antialias_lr5e5.yaml `
  --data_dir data\cpu_native_centered_antialiased_triangles_128_v1\train `
  --init_checkpoint checkpoints\cpu_triangle_large_raw_rgb_v86_128_dilated_refine\checkpoint_final.pt `
  --checkpoint_dir checkpoints\my_v90_reproduction `
  --output_dir outputs\my_v90_reproduction `
  --target_shape_class 2 --seed 70090 --device cpu
```

To reproduce the v91 small-triangle edge refinement, use the same generated training data and initialize from v90:

```powershell
python -m video_foundation.training.train_image `
  --config configs\cpu_triangle_small_raw_rgb_v91_area_edge_refine.yaml `
  --data_dir data\cpu_native_centered_antialiased_triangles_128_v1\train `
  --init_checkpoint checkpoints\cpu_triangle_raw_rgb_v90_native_centered_lr5e5\checkpoint_final.pt `
  --checkpoint_dir checkpoints\my_v91_small_refinement `
  --output_dir outputs\my_v91_small_refinement `
  --target_shape_class 2 --target_size_class 0 --seed 70091 --device cpu
```

The v91 config runs 2,000 steps with batch size 4, learning rate `1e-5`, and four CPU threads. A comparable run took about 18 minutes on the recorded machine. It is retained as the initialization point for v92 and as a historical comparison.

The v92 area-calibration stage is reproduced from v91 with a lower learning rate and a stronger log-area penalty:

```powershell
python -m video_foundation.training.train_image `
  --config configs\cpu_triangle_small_raw_rgb_v92_area_calibration.yaml `
  --data_dir data\cpu_native_centered_antialiased_triangles_128_v1\train `
  --init_checkpoint checkpoints\cpu_triangle_small_raw_rgb_v91_area_edge_refine\checkpoint_final.pt `
  --checkpoint_dir checkpoints\my_v92_small_area_calibration `
  --output_dir outputs\my_v92_small_area_calibration `
  --target_shape_class 2 --target_size_class 0 --seed 70092 --device cpu
```

This measured run used 1,000 steps, batch size 4, learning rate `5e-6`, and four CPU threads; it took about 7.5 minutes in this workspace.

The current v93 small-triangle fine-tune starts from v92 and aligns the occupancy target with the evaluator’s foreground threshold. The config sets both `foreground_threshold=-0.5686` and `occupancy_threshold=-0.5686` (the normalized value for a 55/255 blue pixel), plus `foreground_occupancy_scale=12`. This keeps black near zero occupancy while supervising the matching silhouette boundary:

```powershell
python -m video_foundation.training.train_image `
  --config configs\cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold.yaml `
  --data_dir data\cpu_native_centered_antialiased_triangles_128_v1\train `
  --init_checkpoint checkpoints\cpu_triangle_small_raw_rgb_v92_area_calibration\checkpoint_final.pt `
  --checkpoint_dir checkpoints\my_v93_threshold_aligned `
  --output_dir outputs\my_v93_threshold_aligned `
  --target_shape_class 2 --target_size_class 0 --seed 70093 --device cpu
```

This measured run used 1,000 steps, batch size 4, learning rate `5e-6`, and four CPU threads; it took 9 minutes 42 seconds. Keep v93 on the small-triangle route; use v90 for medium and large triangles.

## Evaluate raw triangle quality

The evaluator uses fixed paired seeds and reports each size independently. This command compares v86 with v90; use the same seed list and data when comparing another checkpoint.

```powershell
python -m video_foundation.training.evaluate_size_conditioned `
  --data_dir data\cpu_native_centered_antialiased_triangles_128_v1\val `
  --baseline_checkpoint checkpoints\cpu_triangle_large_raw_rgb_v86_128_dilated_refine\checkpoint_final.pt `
  --size_checkpoint checkpoints\cpu_triangle_raw_rgb_v90_native_centered_lr5e5\checkpoint_final.pt `
  --output_dir outputs\my_v90_evaluation `
  --samples 128 --sampling_steps 24 --seed 1234 `
  --generation_seeds 1234 2345 3456 4567 5678 6789 7890 8901 9012 11223 22334 33445
```

For a v92-to-v93 comparison, use v92 as `--baseline_checkpoint` and v93 as `--size_checkpoint`. Keep runs with different checkpoints, seeds, sizes, or generation modes separate. Evaluation summaries and the model bundle manifest record the checkpoint/config hashes and route.

## Repository map

- `video_foundation/models/` — compact 2D spatial and CPU video networks.
- `video_foundation/diffusion/` — diffusion schedule, training process, and DDIM sampler.
- `video_foundation/training/` — training, synthetic data generation, and evaluation.
- `video_foundation/inference/` — raw image and CPU video generation.
- `configs/` — versioned training configurations.
- `checkpoints/` — extracted model weights; excluded from Git source checkouts.
- `outputs/` — generated media and run records; excluded from Git source checkouts.
- `release/` — compact downloadable model bundle and checksum.
- `CURRENT_MODEL_STATUS.md` — detailed results, decisions, and known limitations.

## Scope and limitations

The model generates simple synthetic geometry, not arbitrary scenes or open-ended text-to-video. Triangle quality depends on size and random seed; small triangles remain oversized and imperfect. Six-second duration and sub-six-second generation are measured for the recorded CPU/settings, not guaranteed for other machines. A source-only repository checkout does not contain weights: obtain the model bundle and extract it before running the inference commands.
