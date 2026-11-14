"""
Training script for Stage 1 image generation model.
"""

import os
import sys
import argparse
import csv
from datetime import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.optim as optim

# Add project root to path for package imports when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from video_foundation.data.synthetic import SyntheticShapeGenerator
from video_foundation.data.dataset import SyntheticImageDataset
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


def train(config: dict, args: argparse.Namespace):
    """
    Train the diffusion model.

    Args:
        config: Configuration dictionary
        args: Command-line arguments
    """
    # Set seed for reproducibility
    seed = args.seed if args.seed is not None else config['training']['seed']
    set_seed(seed)

    cpu_threads = int(config.get('training', {}).get('cpu_threads', min(4, os.cpu_count() or 1)))
    if cpu_threads < 1:
        raise ValueError("training.cpu_threads must be at least 1")
    torch.set_num_threads(cpu_threads)

    # Get device
    device = get_device()

    # Print configuration
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Device: {device}")
    print(f"Image size: {config['model']['image_size']}")
    print(f"Batch size: {args.batch_size if args.batch_size else config['training']['batch_size']}")
    print(f"Training steps: {args.num_steps if args.num_steps else config['training']['num_steps']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Seed: {seed}")
    print("="*60 + "\n")

    # Generate or load dataset
    data_dir = args.data_dir if args.data_dir else "data/synthetic"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"Generating synthetic dataset in {data_dir}...")
        generator = SyntheticShapeGenerator(
            image_size=config['dataset']['image_size'],
            seed=config['dataset']['seed']
        )
        generator.generate_dataset(
            num_images=config['dataset']['num_images'],
            output_dir=data_dir,
            save_preview=True
        )

    # Create dataset and dataloader
    print("Loading dataset...")
    image_size = args.image_size if args.image_size is not None else config['model']['image_size']
    model_channels = args.model_channels if args.model_channels is not None else config['model']['model_channels']
    num_classes = int(config['model'].get('num_classes', 0))
    num_color_classes = int(config['model'].get('num_color_classes', 0))
    dataset = SyntheticImageDataset(
        data_dir=data_dir,
        image_size=image_size,
        normalize=True,
        cache_in_memory=config.get('dataset', {}).get('cache_in_memory', len(os.listdir(data_dir)) <= 1024),
        return_label=num_classes > 0,
        return_color_label=num_color_classes > 0
    )

    if args.overfit_samples is not None:
        if args.overfit_samples < 1 or args.overfit_samples > len(dataset):
            raise ValueError(f"--overfit_samples must be in [1, {len(dataset)}]")
        dataset.image_files = dataset.image_files[:args.overfit_samples]
        print(f"Overfit mode: restricting training to {len(dataset)} fixed images")

    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )

    # Create model
    print("Creating model...")
    prediction_type = config.get('diffusion', {}).get('prediction_type', 'epsilon')
    if prediction_type not in ('epsilon', 'v_prediction'):
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")
    model = UNet(
        in_channels=config['model']['in_channels'],
        model_channels=model_channels,
        num_classes=num_classes,
        num_color_classes=num_color_classes,
        channel_mult=config['model']['channel_mult'],
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=config['model']['attention_resolutions'],
        dropout=config['model']['dropout'],
        num_timesteps=config['diffusion']['num_timesteps']
    )
    if args.init_checkpoint:
        init_data = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        state = dict(init_data.get('model_state_dict', init_data))
        # Allow curriculum expansion (for example, circle/square to add
        # triangle) while preserving learned rows of conditioning embeddings.
        target_state = model.state_dict()
        for key in ('class_embed.weight', 'color_embed.weight'):
            if key in state and key in target_state and state[key].shape != target_state[key].shape:
                expanded = target_state[key].clone()
                rows = min(state[key].shape[0], expanded.shape[0])
                cols = min(state[key].shape[1], expanded.shape[1])
                expanded[:rows, :cols] = state[key][:rows, :cols]
                state[key] = expanded
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = {key for key in incompatible.missing_keys if key.startswith(('class_embed.', 'color_embed.'))}
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Initialization checkpoint is structurally incompatible: "
                f"missing={incompatible.missing_keys[:5]}, "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )
        previous_prediction_type = init_data.get('config', {}).get('diffusion', {}).get('prediction_type', 'epsilon')
        if previous_prediction_type != prediction_type:
            raise ValueError(
                f"Cannot initialize {prediction_type} training from a "
                f"{previous_prediction_type} checkpoint"
            )
        print(f"Initialized model weights from {args.init_checkpoint}")
    model = model.to(device)

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

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create checkpoint directory
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create output directory
    output_dir = args.output_dir if args.output_dir else "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Setup CSV logging
    csv_path = os.path.join(output_dir, "training_metrics.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'low_t_loss', 'mid_t_loss', 'high_t_loss', 'grad_norm'])

    # Training loop
    num_steps = args.num_steps if args.num_steps else config['training']['num_steps']
    save_interval = config['training']['save_interval']
    log_interval = config['training']['log_interval']
    sample_interval = config['training']['sample_interval']

    print(f"\nStarting training for {num_steps} steps...")
    print(f"Save interval: {save_interval} steps")
    print(f"Log interval: {log_interval} steps")
    print(f"Sample interval: {sample_interval} steps\n")

    step = 0
    epoch = 0
    losses = []

    progress_bar = tqdm(total=num_steps, desc="Training")

    sample_generator = torch.Generator(device=device).manual_seed(seed + 2024)
    fixed_sample_noise = torch.randn(
        4, config['model']['in_channels'], image_size, image_size,
        generator=sample_generator, device=device
    )
    fixed_sample_labels = (
        torch.arange(fixed_sample_noise.shape[0], device=device) % num_classes
        if num_classes > 0 else None
    )
    fixed_sample_colors = (
        torch.arange(fixed_sample_noise.shape[0], device=device) % num_color_classes
        if num_color_classes > 0 else None
    )

    def save_fixed_samples(filename):
        model.eval()
        with torch.no_grad():
            samples = diffusion.process.ddim_sample_loop(
                model=model,
                batch_size=fixed_sample_noise.shape[0],
                channels=config['model']['in_channels'],
                image_size=image_size,
                num_sampling_steps=50,
                device=device,
                initial_noise=fixed_sample_noise.clone(),
                prediction_type=prediction_type,
                recompute_noise_after_clipping=True,
                class_labels=fixed_sample_labels,
                color_labels=fixed_sample_colors
            )
        path = os.path.join(output_dir, filename)
        save_image_grid(samples, path, nrow=2)
        model.train()
        return path

    # Save initial samples (step 0, before training)
    print("Generating initial samples (step 0)...")
    model.eval()
    sample_path = save_fixed_samples(f"samples_step_{step:06d}.png")
    print(f"Initial samples saved to {sample_path}")
    model.train()

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            # Move batch to device
            if isinstance(batch, (tuple, list)):
                x_0 = batch[0].to(device)
                cursor = 1
                class_labels = batch[cursor].to(device=device, dtype=torch.long) if num_classes > 0 else None
                cursor += int(num_classes > 0)
                color_labels = batch[cursor].to(device=device, dtype=torch.long) if num_color_classes > 0 else None
            else:
                x_0 = batch.to(device)
                class_labels = None
                color_labels = None

            # Sample random timesteps
            batch_size = x_0.shape[0]
            timesteps = torch.randint(
                0,
                config['diffusion']['num_timesteps'],
                (batch_size,),
                device=device
            )

            # One forward pass provides both the standard objective and
            # timestep-band diagnostics without adding CPU inference cost.
            x_t, target_noise = diffusion.process.q_sample(x_0, timesteps)
            if prediction_type == 'v_prediction':
                _, alpha_bar, one_minus_alpha_bar = diffusion.schedule.get(timesteps, device)
                target_noise = torch.sqrt(alpha_bar) * target_noise - torch.sqrt(one_minus_alpha_bar) * x_0
            prediction = model(x_t, timesteps, class_labels, color_labels)
            per_sample_loss = (prediction - target_noise).square().flatten(1).mean(1)
            loss = per_sample_loss.mean()
            bands = [timesteps < 333, (timesteps >= 333) & (timesteps < 666), timesteps >= 666]
            band_losses = [
                per_sample_loss[mask].mean().item() if mask.any() else float('nan')
                for mask in bands
            ]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()

            # Log loss
            losses.append(loss.item())

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

            # Log to CSV
            if step % log_interval == 0:
                csv_writer.writerow([step, loss.item(), *band_losses, grad_norm])
                csv_file.flush()

            # Save samples
            if step % sample_interval == 0 and step > 0:
                print(f"\nGenerating samples at step {step}...")
                model.eval()
                with torch.no_grad():
                    samples = diffusion.process.ddim_sample_loop(
                        model=model,
                        batch_size=fixed_sample_noise.shape[0],
                        channels=config['model']['in_channels'],
                        image_size=image_size,
                        num_sampling_steps=50,
                        device=device,
                        initial_noise=fixed_sample_noise.clone(),
                        prediction_type=prediction_type,
                        recompute_noise_after_clipping=True,
                        class_labels=fixed_sample_labels,
                        color_labels=fixed_sample_colors
                    )
                sample_path = os.path.join(output_dir, f"samples_step_{step:06d}.png")
                save_image_grid(samples, sample_path, nrow=2)
                print(f"Samples saved to {sample_path}")
                model.train()

            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pt")
                CheckpointManager.save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                    seed=seed
                )

            step += 1

        epoch += 1

    progress_bar.close()

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    CheckpointManager.save_checkpoint(
        checkpoint_path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=step,
        config=config,
        seed=seed
    )

    # Save final samples
    print("Generating final samples...")
    model.eval()
    with torch.no_grad():
        samples = diffusion.process.ddim_sample_loop(
            model=model,
            batch_size=fixed_sample_noise.shape[0],
            channels=config['model']['in_channels'],
            image_size=image_size,
            num_sampling_steps=50,
            device=device,
            initial_noise=fixed_sample_noise.clone(),
            prediction_type=prediction_type,
            recompute_noise_after_clipping=True,
            class_labels=fixed_sample_labels,
            color_labels=fixed_sample_colors
        )
    final_sample_path = os.path.join(output_dir, "samples_final.png")
    save_image_grid(samples, final_sample_path, nrow=2)
    print(f"Final samples saved to {final_sample_path}")

    # Close CSV file
    csv_file.close()

    # Print summary
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Total steps: {step}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Average loss (last 100 steps): {np.mean(losses[-100:]):.6f}")
    print(f"Checkpoint saved to: {final_checkpoint_path}")
    print(f"Metrics saved to: {csv_path}")
    print(f"Samples saved to: {final_sample_path}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 image generation model")

    # Configuration
    parser.add_argument("--config", type=str, default="configs/image_v0.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    # Training parameters
    parser.add_argument("--num_steps", type=int, default=None, help="Override number of training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--overfit_samples", type=int, default=None, help="Train repeatedly on this many initial dataset images")
    parser.add_argument("--image_size", type=int, default=None, help="Override image resolution")
    parser.add_argument("--model_channels", type=int, default=None, help="Override base U-Net width (must be divisible by 8)")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Initialize model weights from a compatible checkpoint (fresh optimizer)")

    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Train
    train(config, args)


if __name__ == "__main__":
    main()
