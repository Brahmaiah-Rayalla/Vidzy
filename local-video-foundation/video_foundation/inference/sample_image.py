"""
Inference script for generating images from a trained checkpoint.
"""

import os
import sys
import argparse
from PIL import Image
import torch
import numpy as np

# Add project root to path for package imports when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from video_foundation.models.unet import UNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device
from video_foundation.utils.checkpoint import CheckpointManager


def save_image_grid(images: torch.Tensor, save_path: str, nrow: int = 4):
    """
    Save a grid of images.

    Args:
        images: Tensor of shape [batch, channels, height, width] in range [-1, 1]
        save_path: Path to save the grid
        nrow: Number of images per row
    """
    # Denormalize from [-1, 1] to [0, 255]
    images = (images + 1) / 2.0
    images = torch.clamp(images, 0, 1)
    images = (images * 255).byte()

    # Create grid
    from torchvision.utils import make_grid
    grid = make_grid(images, nrow=nrow, padding=2)

    # Convert to PIL Image and save
    grid = grid.permute(1, 2, 0).cpu().numpy()
    grid = Image.fromarray(grid)
    grid.save(save_path)


def save_individual_images(images: torch.Tensor, output_dir: str):
    """
    Save individual images from a batch.

    Args:
        images: Tensor of shape [batch, channels, height, width] in range [-1, 1]
        output_dir: Directory to save images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Denormalize from [-1, 1] to [0, 255]
    images = (images + 1) / 2.0
    images = torch.clamp(images, 0, 1)
    images = (images * 255).byte()

    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        img_path = os.path.join(output_dir, f"sample_{i:04d}.png")
        img.save(img_path)


def sample(args: argparse.Namespace):
    """
    Generate images from a trained checkpoint.

    Args:
        args: Command-line arguments
    """
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using seed: {args.seed}")

    # Get device
    device = get_device()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint_data = torch.load(args.checkpoint, map_location=device)

    # Extract config from checkpoint
    config = checkpoint_data.get('config', {})
    if not config:
        print("Warning: No config found in checkpoint, using defaults")
        config = {
            'model': {
                'in_channels': 3,
                'model_channels': 64,
                'channel_mult': [1, 2, 2, 2],
                'num_res_blocks': 2,
                'attention_resolutions': [],
                'dropout': 0.0,
                'image_size': 64
            },
            'diffusion': {
                'num_timesteps': 1000,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'linear'
            }
        }

    # Reconstruct model
    print("Reconstructing model...")
    model = UNet(
        in_channels=config['model']['in_channels'],
        model_channels=config['model']['model_channels'],
        channel_mult=config['model']['channel_mult'],
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=config['model']['attention_resolutions'],
        dropout=config['model']['dropout'],
        num_timesteps=config['diffusion']['num_timesteps'],
        num_classes=config['model'].get('num_classes', 0),
        num_color_classes=config['model'].get('num_color_classes', 0)
    )
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create diffusion sampler
    diffusion = DiffusionSampler(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule=config['diffusion']['beta_schedule']
    )

    # Sampling parameters
    num_samples = args.num_samples
    num_sampling_steps = args.sampling_steps
    image_size = config['model'].get('image_size', 64)
    channels = config['model']['in_channels']

    print("\n" + "="*60)
    print("Sampling Configuration")
    print("="*60)
    print(f"Device: {device}")
    print(f"Number of samples: {num_samples}")
    print(f"Sampling steps: {num_sampling_steps}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Checkpoint step: {checkpoint_data.get('step', 'unknown')}")
    print("="*60 + "\n")

    # Progress callback
    def progress_callback(step, total):
        if (step + 1) % 10 == 0 or step == total - 1:
            print(f"Sampling step {step + 1}/{total}")

    # Generate samples
    print("Generating samples...")
    with torch.no_grad():
        class_labels = None
        color_labels = None
        if model.num_classes > 0:
            if args.class_label < 0 or args.class_label >= model.num_classes:
                raise ValueError(f"--class_label must be in [0, {model.num_classes - 1}]")
            class_labels = torch.full((num_samples,), args.class_label, device=device, dtype=torch.long)
        if model.num_color_classes > 0:
            if args.vary_colors:
                color_labels = torch.arange(num_samples, device=device, dtype=torch.long) % model.num_color_classes
            elif args.color_label < 0 or args.color_label >= model.num_color_classes:
                raise ValueError(f"--color_label must be in [0, {model.num_color_classes - 1}]")
            else:
                color_labels = torch.full((num_samples,), args.color_label, device=device, dtype=torch.long)
        samples = diffusion.sample(
            model=model,
            batch_size=num_samples,
            channels=channels,
            image_size=image_size,
            num_sampling_steps=num_sampling_steps,
            device=device,
            progress_callback=progress_callback,
            prediction_type=config.get('diffusion', {}).get('prediction_type', 'epsilon'),
            recompute_noise_after_clipping=True,
            class_labels=class_labels,
            color_labels=color_labels
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save individual images
    print("Saving individual images...")
    save_individual_images(samples, args.output_dir)

    # Save grid
    print("Saving image grid...")
    grid_path = os.path.join(args.output_dir, "grid.png")
    save_image_grid(samples, grid_path, nrow=min(4, num_samples))

    print("\n" + "="*60)
    print("Sampling Complete")
    print("="*60)
    print(f"Individual images saved to: {args.output_dir}")
    print(f"Grid saved to: {grid_path}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate images from a trained checkpoint")

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")

    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="outputs/samples", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--sampling_steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--class_label", type=int, default=0, help="Class ID for class-conditional checkpoints (0 circle, 1 square, 2 triangle)")
    parser.add_argument("--color_label", type=int, default=0, help="Palette ID for color-conditional checkpoints (0 red, 1 green, 2 blue, 3 yellow)")
    parser.add_argument("--vary_colors", action="store_true", help="Cycle color IDs across samples in one batch")

    args = parser.parse_args()

    # Sample
    sample(args)


if __name__ == "__main__":
    main()
