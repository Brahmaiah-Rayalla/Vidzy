# Local Video Foundation - Stage 1

A lightweight diffusion model implementation for generating synthetic geometric images at 64×64 resolution. Built from scratch with no pretrained weights or external APIs.

## Overview

This is Stage 1 of a project to build a locally runnable AI generation engine. This milestone implements:
- A synthetic dataset of geometric shapes (circles, squares, triangles)
- A compact pixel-space DDPM (Denoising Diffusion Probabilistic Model)
- A convolutional U-Net architecture built from scratch
- DDIM sampling for efficient image generation
- Complete training and inference pipeline

**Key Constraints:**
- No pretrained model weights, VAEs, or encoders
- No external generative model APIs
- Designed for CPU execution on Windows laptops
- Synthetic dataset generated entirely offline

## System Requirements

- Python 3.8 or higher
- Windows 10/11 (PowerShell 5.1+)
- 16 GB RAM (tested on Intel i7 with Intel Iris Xe graphics)
- ~2 GB disk space for dataset and checkpoints

## Installation

### Windows PowerShell

```powershell
# Navigate to the project directory
cd local-video-foundation

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Linux/macOS (bash)

```bash
# Navigate to the project directory
cd local-video-foundation

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
local-video-foundation/
├── configs/
│   └── image_v0.yaml          # Model and training configuration
├── video_foundation/
│   ├── data/
│   │   ├── synthetic.py       # Synthetic dataset generator
│   │   └── dataset.py         # PyTorch Dataset wrapper
│   ├── models/
│   │   └── unet.py            # U-Net architecture
│   ├── diffusion/
│   │   ├── schedule.py        # Noise schedule
│   │   ├── process.py         # Forward/reverse diffusion
│   │   └── sampler.py         # High-level sampler
│   ├── training/
│   │   └── train_image.py     # Training script
│   ├── inference/
│   │   └── sample_image.py    # Inference script
│   └── utils/
│       ├── reproducibility.py # Seed and device utilities
│       └── checkpoint.py      # Checkpoint management
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_diffusion.py
│   └── test_checkpoint.py
├── data/                      # Synthetic dataset (generated locally)
├── checkpoints/               # Model checkpoints
├── outputs/                   # Training outputs and samples
├── run_tests.py               # Automated test runner
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

### 1. Generate Synthetic Dataset

```powershell
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Generate 5,000 synthetic images at 64x64
python -m video_foundation.data.synthetic --num_images 5000 --image_size 64 --output_dir data/synthetic --seed 42
```

This creates:
- 5,000 PNG images in `data/synthetic/`
- A preview grid showing sample images

### 2. Run Automated Tests

```powershell
# Run all tests to verify the implementation
python run_tests.py
```

Expected output:
```
============================================================
Running Automated Tests
============================================================

Dataset Tests
------------------------------------------------------------
✓ Synthetic generation test passed
✓ Dataset loading test passed
✓ Reproducibility test passed

Model Tests
------------------------------------------------------------
✓ Model output shape test passed
✓ Model initialization test passed
✓ Gradient flow test passed

Diffusion Tests
------------------------------------------------------------
✓ Schedule computation test passed
✓ Forward diffusion test passed
✓ Training loss test passed
✓ DDIM sampling test passed

Checkpoint Tests
------------------------------------------------------------
✓ Checkpoint save/load test passed
✓ Config save/load test passed

============================================================
All tests passed successfully!
============================================================
```

### 3. Run Smoke Test (Short Training)

```powershell
# Run a very short training to verify the pipeline
python -m video_foundation.training.train_image --config configs/image_v0.yaml --num_steps 10 --batch_size 2 --seed 42
```

This runs:
- 10 training steps
- Batch size of 2
- Saves a checkpoint
- Generates sample images

**Note:** This is a smoke test to verify the pipeline works. The model will not converge in 10 steps.

### 4. Full Training

```powershell
# Full training with default settings (2,000 steps)
python -m video_foundation.training.train_image --config configs/image_v0.yaml
```

Default configuration:
- Resolution: 64×64
- Training steps: 2,000
- Batch size: 8
- Diffusion timesteps: 1,000
- Learning rate: 0.0002
- Checkpoint interval: 500 steps
- Sample interval: 250 steps

**Expected duration on CPU:** Several hours (varies by hardware)

To customize training:
```powershell
# Custom training with specific parameters
python -m video_foundation.training.train_image `
    --config configs/image_v0.yaml `
    --num_steps 5000 `
    --batch_size 4 `
    --seed 123 `
    --output_dir outputs/my_run
```

### 5. Generate Samples from Checkpoint

```powershell
# Generate 4 samples using the final checkpoint
python -m video_foundation.inference.sample_image `
    --checkpoint checkpoints/checkpoint_final.pt `
    --output_dir outputs/samples `
    --num_samples 4 `
    --sampling_steps 50 `
    --seed 42
```

This creates:
- Individual PNG images in `outputs/samples/`
- A grid image showing all samples

## Configuration

Edit `configs/image_v0.yaml` to customize:

```yaml
# Model architecture
model:
  image_size: 64
  in_channels: 3
  model_channels: 64
  channel_mult: [1, 2, 2, 2]
  num_res_blocks: 2
  dropout: 0.0

# Diffusion process
diffusion:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  beta_schedule: "linear"

# Training
training:
  num_steps: 2000
  batch_size: 8
  learning_rate: 0.0002
  weight_decay: 0.01
  seed: 42
  save_interval: 500
  log_interval: 10
  sample_interval: 250

# Dataset
dataset:
  num_images: 5000
  image_size: 64
  seed: 42
```

## Device Selection

The training script automatically detects and uses the best available device:
- CUDA (NVIDIA GPU) if available
- MPS (Apple Silicon) if available
- CPU otherwise

Check your detected device:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

## Training Metrics

Training metrics are saved to `outputs/training_metrics.csv`:
- Step number
- Loss value

Sample images are saved periodically to `outputs/`:
- `samples_step_000250.png` (every 250 steps)
- `samples_final.png` (after training completes)

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors on CPU:
- Reduce batch size: `--batch_size 4` or `--batch_size 2`
- Reduce model channels in config: `model_channels: 32`
- Reduce image size in config: `image_size: 32`

### Slow Training on CPU

Training on CPU is expected to be slow. Options:
- Reduce training steps for experimentation: `--num_steps 500`
- Reduce batch size: `--batch_size 4`
- Use a smaller model by editing `configs/image_v0.yaml`

### Import Errors

Ensure you're in the `local-video-foundation` directory and the virtual environment is activated:
```powershell
cd local-video-foundation
.\venv\Scripts\Activate.ps1
```

## Current Limitations

- CPU-only training (CUDA not available in current environment)
- Small image resolution (64×64)
- Simple geometric shapes only
- No attention mechanisms
- No text conditioning
- No video generation (Stage 2)

## Next Steps (Stage 2)

Future improvements planned:
- Add self-attention layers to U-Net
- Implement latent-space diffusion for efficiency
- Add text conditioning for text-to-image
- Scale to higher resolutions
- Implement temporal layers for video generation
- Add more complex synthetic datasets

## Architecture Notes

### U-Net Architecture
- Sinusoidal timestep embeddings
- Residual convolution blocks with group normalization
- Skip connections between encoder and decoder
- 4 downsampling/upsampling levels with channel multipliers [1, 2, 2, 2]

### Diffusion Process
- Linear noise schedule from β=0.0001 to β=0.02 over 1,000 timesteps
- Noise prediction objective: MSE(predicted_noise, actual_noise)
- DDIM sampling for efficient generation (configurable steps)

### Dataset
- 5,000 synthetic images with random geometric shapes
- Each image contains 1-3 shapes (circles, squares, triangles)
- Random colors, positions, and sizes
- Fixed seed for reproducibility

## License

This is a research project. All code is written from scratch with no pretrained weights.

## Acknowledgments

Built following the DDPM paper: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
