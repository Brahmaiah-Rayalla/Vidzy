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
from video_foundation.inference.shape_rendering import (
    render_antialiased_shape, render_canonical_shape,
)
from video_foundation.utils.reproducibility import set_seed, get_device
from video_foundation.utils.checkpoint import CheckpointManager


PALETTE_RGB = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
SIZE_EXTENTS_64 = (15, 29, 42)


def _palette_mask(rgb: np.ndarray, color_id: int) -> np.ndarray:
    """Find pixels that most likely belong to the requested curriculum color."""
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if color_id == 0:
        return (red > green * 1.2) & (red > blue * 1.2) & (red > 45)
    if color_id == 1:
        return (green > red * 1.2) & (green > blue * 1.2) & (green > 45)
    if color_id == 2:
        return (blue > red * 1.2) & (blue > green * 1.1) & (blue > 45)
    return (red > blue * 1.2) & (green > blue * 1.1) & (red > 45) & (green > 45)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected palette component using a small CPU flood fill."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    largest = []
    for seed_y, seed_x in zip(*np.where(mask)):
        if visited[seed_y, seed_x]:
            continue
        visited[seed_y, seed_x] = True
        stack = [(int(seed_y), int(seed_x))]
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(component) > len(largest):
            largest = component
    clean = np.zeros_like(mask, dtype=bool)
    if largest:
        ys, xs = zip(*largest)
        clean[np.asarray(ys), np.asarray(xs)] = True
    return clean


def refine_curriculum_geometry(samples: torch.Tensor, shape_id: int, color_ids,
                               size_id: int, image_size: int,
                               output_image_size: int = None) -> torch.Tensor:
    """Use diffusion for placement, then render a clean high-resolution primitive."""
    if shape_id not in (0, 1, 2):
        raise ValueError("Canonical refinement supports shape IDs 0-2 and palette IDs 0-3")
    if size_id not in range(len(SIZE_EXTENTS_64)):
        raise ValueError("Canonical refinement requires a size ID in [0, 2]")
    if isinstance(color_ids, int):
        color_ids = [color_ids] * len(samples)
    if len(color_ids) != len(samples) or any(color_id not in range(len(PALETTE_RGB)) for color_id in color_ids):
        raise ValueError("Canonical refinement requires one palette ID in [0, 3] per sample")

    output_image_size = int(output_image_size or image_size)
    if output_image_size < 1:
        raise ValueError("output_image_size must be positive")
    center_scale = output_image_size / float(image_size)
    extent = max(1, round(SIZE_EXTENTS_64[size_id] * output_image_size / 64))
    refined = []
    for sample, color_id in zip(samples.detach().cpu(), color_ids):
        rgb = ((sample.clamp(-1, 1).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
        component = _largest_component(_palette_mask(rgb, color_id))
        ys, xs = np.where(component)
        if len(xs):
            center = (
                (float(xs.mean()) + 0.5) * center_scale - 0.5,
                (float(ys.mean()) + 0.5) * center_scale - 0.5,
            )
        else:
            center = (
                (image_size / 2.0) * center_scale - 0.5,
                (image_size / 2.0) * center_scale - 0.5,
            )

        coverage = render_antialiased_shape(
            shape_id, center, extent, output_image_size, supersample=4
        )
        clean_rgb = np.rint(
            coverage[..., None] * np.asarray(PALETTE_RGB[color_id], dtype=np.float32)
        ).astype(np.uint8)
        refined.append(torch.from_numpy(clean_rgb).permute(2, 0, 1).float() / 127.5 - 1.0)
    return torch.stack(refined).to(device=samples.device, dtype=samples.dtype)


def render_curriculum_geometry(num_samples: int, shape_id: int, color_ids,
                               size_id: int, image_size: int, device) -> torch.Tensor:
    """Render requested primitives directly, without running pixel diffusion."""
    if shape_id not in (0, 1, 2):
        raise ValueError("Canonical rendering supports shape IDs 0-2")
    if size_id not in range(len(SIZE_EXTENTS_64)):
        raise ValueError("Canonical rendering requires a size ID in [0, 2]")
    if isinstance(color_ids, int):
        color_ids = [color_ids] * num_samples
    if len(color_ids) != num_samples or any(color_id not in range(len(PALETTE_RGB)) for color_id in color_ids):
        raise ValueError("Canonical rendering requires one palette ID in [0, 3] per sample")

    extent = max(1, round(SIZE_EXTENTS_64[size_id] * image_size / 64))
    center = ((image_size - 1) / 2.0, (image_size - 1) / 2.0)
    rendered = []
    for color_id in color_ids:
        coverage = render_antialiased_shape(
            shape_id, center, extent, image_size, supersample=4
        )
        rgb = np.rint(coverage[..., None] * np.asarray(PALETTE_RGB[color_id])).astype(np.uint8)
        rendered.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1.0)
    return torch.stack(rendered).to(device=device)


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
    device = get_device(args.device)

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

    model_guided_geometry = args.model_guided_geometry
    refine_geometry = args.refine_geometry
    if model_guided_geometry:
        refine_geometry = False
    elif refine_geometry is None:
        model_config = config.get('model', {})
        refine_geometry = (
            int(model_config.get('num_classes', 0)) >= 3
            and int(model_config.get('num_color_classes', 0)) >= 4
            and int(model_config.get('num_size_classes', 0)) >= 3
        )

    # Route raw triangle requests to size-specific specialists only when they
    # have been measured against the calibrated base checkpoint. Keep the base
    # model as the fallback for any size without an explicitly supplied route.
    specialist_path = None
    if not refine_geometry and args.class_label == 2:
        if args.size == "small" and args.small_triangle_checkpoint:
            specialist_path = args.small_triangle_checkpoint
        elif args.size == "medium":
            specialist_path = args.medium_triangle_checkpoint or args.medium_large_checkpoint
        elif args.size == "large":
            specialist_path = args.large_triangle_checkpoint or args.medium_large_checkpoint
    if specialist_path:
        specialist_data = torch.load(
            specialist_path, map_location=device, weights_only=False
        )
        specialist_config = specialist_data.get('config', {})
        if not specialist_config:
            raise ValueError("The specialist checkpoint must include its model config")
        architecture_keys = (
            "in_channels", "model_channels", "channel_mult", "num_res_blocks",
            "num_classes", "num_color_classes", "num_size_classes", "image_size",
            "spatial_size_conditioning", "high_resolution_refinement", "middle_dilation",
        )
        base_model_config = config["model"]
        specialist_model_config = specialist_config["model"]
        boolean_model_defaults = {"spatial_size_conditioning": False,
                                  "high_resolution_refinement": False}
        base_model_config.setdefault("middle_dilation", 1)
        specialist_model_config.setdefault("middle_dilation", 1)
        mismatches = [
            key for key in architecture_keys
            if base_model_config.get(key, boolean_model_defaults.get(key))
            != specialist_model_config.get(key, boolean_model_defaults.get(key))
        ]
        if mismatches:
            raise ValueError(
                "Base and specialist checkpoints have incompatible model fields: "
                + ", ".join(mismatches)
            )
        if specialist_config.get("diffusion") != config.get("diffusion"):
            raise ValueError("Base and specialist checkpoints use different diffusion configs")
        checkpoint_data = specialist_data
        config = specialist_config
        print(
            f"Routing {args.size} triangle request to specialist checkpoint {specialist_path}"
        )

    model_config = config['model']
    num_classes = int(model_config.get('num_classes', 0))
    num_color_classes = int(model_config.get('num_color_classes', 0))
    num_size_classes = int(model_config.get('num_size_classes', 0))
    model = None
    if not refine_geometry:
        # Raw generation is the only path that needs the neural network.
        print("Reconstructing diffusion model...")
        model = UNet(
            in_channels=model_config['in_channels'],
            model_channels=model_config['model_channels'],
            channel_mult=model_config['channel_mult'],
            num_res_blocks=model_config['num_res_blocks'],
            attention_resolutions=model_config['attention_resolutions'],
            dropout=model_config['dropout'],
            num_timesteps=config['diffusion']['num_timesteps'],
            num_classes=num_classes,
            num_color_classes=num_color_classes,
            num_size_classes=num_size_classes,
            spatial_size_conditioning=model_config.get('spatial_size_conditioning', False),
            high_resolution_refinement=model_config.get('high_resolution_refinement', False),
            size_embedding_scale=model_config.get('size_embedding_scale', 1.0),
            middle_dilation=int(model_config.get('middle_dilation', 1))
        )
        model.load_state_dict(checkpoint_data['model_state_dict'])
        model = model.to(device)
        model.eval()
    else:
        print("Skipping diffusion model loading for exact canonical rendering")

    # Count parameters
    if model is not None:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

    # Exact curriculum rendering is the fast production decoder; construct the
    # diffusion sampler only for users who explicitly request raw model output.
    diffusion = None
    if not refine_geometry:
        diffusion = DiffusionSampler(
            num_timesteps=config['diffusion']['num_timesteps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            schedule=config['diffusion']['beta_schedule']
        )

    # Sampling parameters
    num_samples = args.num_samples
    num_sampling_steps = args.sampling_steps
    checkpoint_image_size = config['model'].get('image_size', 64)
    image_size = max(checkpoint_image_size, 256) if refine_geometry else checkpoint_image_size
    output_image_size = (
        max(checkpoint_image_size, 256) if model_guided_geometry else image_size
    )
    if args.output_image_size is not None:
        if not (refine_geometry or model_guided_geometry):
            raise ValueError("--output_image_size requires exact or model-guided geometry rendering")
        if args.output_image_size < 16 or args.output_image_size > 1024:
            raise ValueError("--output_image_size must be between 16 and 1024")
        output_image_size = args.output_image_size
        if refine_geometry:
            image_size = output_image_size
    channels = config['model']['in_channels']

    print("\n" + "="*60)
    print("Sampling Configuration")
    print("="*60)
    print(f"Device: {device}")
    print(f"Number of samples: {num_samples}")
    print(f"Sampling steps: {num_sampling_steps}")
    mode = (
        "exact CPU renderer" if refine_geometry
        else "model-guided geometry (diffusion + renderer)" if model_guided_geometry
        else "raw diffusion"
    )
    print(f"Generation mode: {mode}")
    print(f"Image size: {image_size}x{image_size}")
    if output_image_size != image_size:
        print(f"Rendered output size: {output_image_size}x{output_image_size}")
    print(f"Checkpoint step: {checkpoint_data.get('step', 'unknown')}")
    print("="*60 + "\n")

    # Progress callback
    def progress_callback(step, total):
        if (step + 1) % 10 == 0 or step == total - 1:
            print(f"Sampling step {step + 1}/{total}")

    # Generate samples
    print("Generating samples...")
    # Model construction advances PyTorch's RNG even when checkpoint weights
    # are loaded afterward. Reset here so --seed controls the actual diffusion
    # noise, independent of architecture initialization and checkpoint routing.
    if args.seed is not None:
        set_seed(args.seed)
    with torch.no_grad():
        class_labels = None
        color_labels = None
        size_labels = None
        if num_classes > 0:
            if args.class_label < 0 or args.class_label >= num_classes:
                raise ValueError(f"--class_label must be in [0, {num_classes - 1}]")
            class_labels = torch.full((num_samples,), args.class_label, device=device, dtype=torch.long)
        if num_color_classes > 0:
            if args.vary_colors:
                color_labels = torch.arange(num_samples, device=device, dtype=torch.long) % num_color_classes
            elif args.color_label < 0 or args.color_label >= num_color_classes:
                raise ValueError(f"--color_label must be in [0, {num_color_classes - 1}]")
            else:
                color_labels = torch.full((num_samples,), args.color_label, device=device, dtype=torch.long)
        if num_size_classes > 0:
            size_ids = {"small": 0, "medium": 1, "large": 2}
            requested_size_id = size_ids[args.size]
            size_labels = torch.full(
                (num_samples,), requested_size_id, device=device, dtype=torch.long
            )
        else:
            requested_size_id = {"small": 0, "medium": 1, "large": 2}[args.size]
        if refine_geometry:
            sample_color_ids = (
                color_labels.detach().cpu().tolist()
                if color_labels is not None else [args.color_label] * num_samples
            )
            samples = render_curriculum_geometry(
                num_samples, args.class_label, sample_color_ids,
                requested_size_id, image_size, device,
            )
        else:
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
                color_labels=color_labels,
                size_labels=size_labels
            )
            if model_guided_geometry:
                sample_color_ids = (
                    color_labels.detach().cpu().tolist()
                    if color_labels is not None else [args.color_label] * num_samples
                )
                samples = refine_curriculum_geometry(
                    samples, args.class_label, sample_color_ids,
                    requested_size_id, image_size, output_image_size,
                )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if refine_geometry:
        print("Rendered exact CPU canonical geometry; diffusion sampling was skipped")
    elif model_guided_geometry:
        print("Rendered learned diffusion placement as an antialiased canonical primitive")

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
    parser.add_argument("--small_triangle_checkpoint", type=str, default=None,
                        help="Optional measured triangle specialist used for small raw or model-guided requests")
    parser.add_argument("--medium_large_checkpoint", type=str, default=None,
                        help="Optional fallback triangle specialist for medium and large raw or model-guided requests")
    parser.add_argument("--medium_triangle_checkpoint", type=str, default=None,
                        help="Optional medium-only triangle specialist; overrides --medium_large_checkpoint for medium requests")
    parser.add_argument("--large_triangle_checkpoint", type=str, default=None,
                        help="Optional large-only triangle specialist; overrides --medium_large_checkpoint for large requests")

    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="outputs/samples", help="Output directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="Execution device; choose cpu to force native CPU inference")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--sampling_steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--output_image_size", type=int, default=None,
                        help="Override exact or model-guided render resolution (default 256, 16-1024); raw diffusion stays at checkpoint resolution")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--class_label", type=int, default=0, help="Class ID for class-conditional checkpoints (0 circle, 1 square, 2 triangle)")
    parser.add_argument("--color_label", type=int, default=0, help="Palette ID for color-conditional checkpoints (0 red, 1 green, 2 blue, 3 yellow)")
    parser.add_argument("--vary_colors", action="store_true", help="Cycle color IDs across samples in one batch")
    parser.add_argument("--size", choices=("small", "medium", "large"), default="medium",
                        help="Size condition for size-aware image checkpoints")
    geometry_group = parser.add_mutually_exclusive_group()
    geometry_group.add_argument("--refine_geometry", dest="refine_geometry", action="store_true",
                                help="Force exact CPU rendering of the requested primitive")
    geometry_group.add_argument("--raw_generation", dest="refine_geometry", action="store_false",
                                help="Keep the raw diffusion output without geometry refinement")
    geometry_group.add_argument("--model_guided_geometry", action="store_true",
                                help="Use diffusion for object placement, then render crisp canonical geometry")
    parser.set_defaults(refine_geometry=None)

    args = parser.parse_args()

    # Sample
    sample(args)


if __name__ == "__main__":
    main()
