"""
Training script for conditional text-to-image diffusion model.
"""

import os
import sys
import argparse
import csv
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.optim as optim

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.data.text_image_dataset import TextImageShapeGenerator, create_train_val_test_splits
from video_foundation.data.text_image_loader import TextImageDataset
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device


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
    Train the conditional diffusion model.

    Args:
        config: Configuration dictionary
        args: Command-line arguments
    """
    # Set seed for reproducibility
    seed = args.seed if args.seed is not None else config['training']['seed']
    set_seed(seed)

    # Get device
    device = get_device()

    # Print configuration
    print("\n" + "="*60)
    print("Conditional Training Configuration")
    print("="*60)
    print(f"Device: {device}")
    print(f"Image size: {config['model']['image_size']}")
    print(f"Batch size: {args.batch_size if args.batch_size else config['training']['batch_size']}")
    print(f"Training steps: {args.num_steps if args.num_steps else config['training']['num_steps']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Seed: {seed}")
    print("="*60 + "\n")

    # Generate or load dataset
    data_dir = args.data_dir if args.data_dir else "data/text_image"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"Generating synthetic text-image dataset in {data_dir}...")
        generator = TextImageShapeGenerator(
            image_size=config['dataset']['image_size'],
            seed=config['dataset']['seed'],
            max_objects=config['dataset'].get('max_objects', 2)
        )
        metadata = generator.generate_dataset(
            num_images=config['dataset']['num_images'],
            output_dir=data_dir,
            save_preview=True
        )

        # Create splits
        create_train_val_test_splits(
            metadata,
            data_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )

    # Build tokenizer from training captions
    print("\nBuilding tokenizer...")
    train_metadata_path = os.path.join(data_dir, 'train_metadata.json')
    with open(train_metadata_path, 'r') as f:
        train_metadata = json.load(f)

    captions = [m['caption'] for m in train_metadata]
    tokenizer = SimpleTokenizer(
        vocab_size=config['text_encoder']['vocab_size'],
        min_freq=config['text_encoder']['min_freq']
    )
    tokenizer.build_vocab(captions)

    # Save tokenizer
    tokenizer_dir = os.path.join(data_dir, 'tokenizer')
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, 'tokenizer.json'))

    # Create datasets
    print("Loading datasets...")
    train_dataset = TextImageDataset(
        metadata_path=train_metadata_path,
        image_dir=data_dir,
        image_size=config['model']['image_size'],
        normalize=True
    )

    val_metadata_path = os.path.join(data_dir, 'val_metadata.json')
    val_dataset = TextImageDataset(
        metadata_path=val_metadata_path,
        image_dir=data_dir,
        image_size=config['model']['image_size'],
        normalize=True
    )

    def collate_fn(batch):
        """Custom collate function to handle dictionary batches."""
        images = torch.stack([item['image'] for item in batch])
        captions = [item['caption'] for item in batch]
        return {'image': images, 'caption': captions}

    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )

    # Create text encoder
    print("Creating text encoder...")
    text_encoder = TextEncoder(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=config['text_encoder']['d_model'],
        n_heads=config['text_encoder']['n_heads'],
        n_layers=config['text_encoder']['n_layers'],
        d_ff=config['text_encoder']['d_ff'],
        max_seq_len=config['text_encoder']['max_seq_len'],
        dropout=config['text_encoder']['dropout']
    )
    text_encoder = text_encoder.to(device)

    # Create conditional U-Net
    print("Creating conditional U-Net...")
    model = ConditionalUNet(
        in_channels=config['model']['in_channels'],
        model_channels=config['model']['model_channels'],
        channel_mult=config['model']['channel_mult'],
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=config['model']['attention_resolutions'],
        dropout=config['model']['dropout'],
        num_timesteps=config['diffusion']['num_timesteps'],
        text_dim=config['text_encoder']['d_model'],
        n_heads=config.get('n_heads', 4),
        use_cross_attn=config['model']['use_cross_attn']
    )
    model = model.to(device)

    # Load unconditional weights if specified
    if args.load_unconditional and os.path.exists(args.load_unconditional):
        print(f"Loading unconditional weights from {args.load_unconditional}...")
        checkpoint = torch.load(args.load_unconditional, map_location=device)
        model.load_unconditional_weights(checkpoint['model_state_dict'])

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    text_encoder_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Text encoder parameters: {text_encoder_params:,}")
    print(f"Total parameters: {num_params + text_encoder_params:,}")

    # Create diffusion sampler
    diffusion = DiffusionSampler(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule=config['diffusion']['beta_schedule']
    )

    # Create optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(text_encoder.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create checkpoint directory
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else "checkpoints_conditional"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create output directory
    output_dir = args.output_dir if args.output_dir else "outputs_conditional"
    os.makedirs(output_dir, exist_ok=True)

    # Setup CSV logging
    csv_path = os.path.join(output_dir, "training_metrics.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'train_loss', 'val_loss'])

    # Training loop
    num_steps = args.num_steps if args.num_steps else config['training']['num_steps']
    save_interval = config['training']['save_interval']
    log_interval = config['training']['log_interval']
    sample_interval = config['training']['sample_interval']
    val_interval = config['training'].get('val_interval', save_interval)

    print(f"\nStarting training for {num_steps} steps...")
    print(f"Save interval: {save_interval} steps")
    print(f"Log interval: {log_interval} steps")
    print(f"Sample interval: {sample_interval} steps")
    print(f"Validation interval: {val_interval} steps\n")

    step = 0
    epoch = 0
    train_losses = []

    progress_bar = tqdm(total=num_steps, desc="Training")

    while step < num_steps:
        # Training epoch
        model.train()
        text_encoder.train()

        for batch in train_loader:
            if step >= num_steps:
                break

            # Move batch to device
            x_0 = batch['image'].to(device)
            captions = batch['caption']

            # Tokenize captions
            max_len = config['text_encoder']['max_seq_len']
            token_ids_list = []
            attention_masks_list = []

            for caption in captions:
                token_ids, attention_mask = tokenizer.encode(caption, max_length=max_len)
                token_ids_list.append(token_ids)
                attention_masks_list.append(attention_mask)

            token_ids = torch.tensor(token_ids_list, device=device)
            attention_mask = torch.tensor(attention_masks_list, device=device).float()

            # Encode text
            text_emb, _ = text_encoder(token_ids, attention_mask)

            # Sample random timesteps
            batch_size = x_0.shape[0]
            timesteps = torch.randint(
                0,
                config['diffusion']['num_timesteps'],
                (batch_size,),
                device=device
            )

            # Compute loss
            noise = torch.randn_like(x_0)
            x_t, noise = diffusion.process.q_sample(x_0, timesteps, noise)

            # Predict noise with conditional model
            predicted_noise = model(x_t, timesteps, text_emb, attention_mask)

            # Compute MSE loss
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss
            train_losses.append(loss.item())

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

            # Log to CSV
            if step % log_interval == 0:
                csv_writer.writerow([step, loss.item(), ''])
                csv_file.flush()

            # Validation
            if step % val_interval == 0 and step > 0:
                model.eval()
                text_encoder.eval()
                val_losses = []

                with torch.no_grad():
                    for val_batch in val_loader:
                        x_0_val = val_batch['image'].to(device)
                        captions_val = val_batch['caption']

                        # Tokenize
                        token_ids_list = []
                        attention_masks_list = []
                        for caption in captions_val:
                            token_ids, attention_mask = tokenizer.encode(caption, max_length=max_len)
                            token_ids_list.append(token_ids)
                            attention_masks_list.append(attention_mask)

                        token_ids = torch.tensor(token_ids_list, device=device)
                        attention_mask = torch.tensor(attention_masks_list, device=device).float()

                        # Encode
                        text_emb_val, _ = text_encoder(token_ids, attention_mask)

                        # Sample timesteps
                        batch_size_val = x_0_val.shape[0]
                        timesteps_val = torch.randint(
                            0,
                            config['diffusion']['num_timesteps'],
                            (batch_size_val,),
                            device=device
                        )

                        # Compute loss
                        noise_val = torch.randn_like(x_0_val)
                        x_t_val, noise_val = diffusion.process.q_sample(x_0_val, timesteps_val, noise_val)
                        predicted_noise_val = model(x_t_val, timesteps_val, text_emb_val, attention_mask)
                        loss_val = torch.nn.functional.mse_loss(predicted_noise_val, noise_val)
                        val_losses.append(loss_val.item())

                avg_val_loss = np.mean(val_losses)
                print(f"\nStep {step}: Validation loss = {avg_val_loss:.6f}")

                # Update CSV with validation loss
                # Find the last row and update it
                # (Simplified: just append a new row)
                csv_writer.writerow([step, '', avg_val_loss])
                csv_file.flush()

                model.train()
                text_encoder.train()

            # Save samples
            if step % sample_interval == 0 and step > 0:
                print(f"\nGenerating samples at step {step}...")
                model.eval()
                text_encoder.eval()

                # Use fixed prompts for comparison
                test_prompts = [
                    "A small red circle on a blue background.",
                    "A large green triangle near the center.",
                    "A yellow square on a white background.",
                ]

                with torch.no_grad():
                    samples_list = []
                    for prompt in test_prompts:
                        # Tokenize
                        token_ids, attention_mask = tokenizer.encode(prompt, max_length=max_len)
                        token_ids = torch.tensor([token_ids], device=device)
                        attention_mask = torch.tensor([attention_mask], device=device).float()

                        # Encode
                        text_emb_prompt, _ = text_encoder(token_ids, attention_mask)

                        # Sample using conditional model with text conditioning
                        samples = diffusion.sample_ddpm(
                            model=model,
                            batch_size=1,
                            channels=config['model']['in_channels'],
                            image_size=config['model']['image_size'],
                            device=device,
                            text_emb=text_emb_prompt,
                            text_mask=attention_mask
                        )
                        samples_list.append(samples)

                samples = torch.cat(samples_list, dim=0)
                sample_path = os.path.join(output_dir, f"samples_step_{step:06d}.png")
                save_image_grid(samples, sample_path, nrow=3)
                print(f"Samples saved to {sample_path}")

                model.train()
                text_encoder.train()

            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'text_encoder_state_dict': text_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'seed': seed,
                }, checkpoint_path)
                print(f"\nCheckpoint saved to {checkpoint_path}")

            step += 1

        epoch += 1

    progress_bar.close()

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'text_encoder_state_dict': text_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'seed': seed,
    }, final_checkpoint_path)

    # Save text encoder separately
    text_encoder_path = os.path.join(checkpoint_dir, "text_encoder.pt")
    text_encoder.save(text_encoder_path, tokenizer)

    # Close CSV file
    csv_file.close()

    # Print summary
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Total steps: {step}")
    print(f"Final loss: {train_losses[-1]:.6f}")
    print(f"Average loss (last 100 steps): {np.mean(train_losses[-100:]):.6f}")
    print(f"Checkpoint saved to: {final_checkpoint_path}")
    print(f"Text encoder saved to: {text_encoder_path}")
    print(f"Metrics saved to: {csv_path}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train conditional text-to-image diffusion model")

    # Configuration
    parser.add_argument("--config", type=str, default="configs/conditional_v0.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    # Training parameters
    parser.add_argument("--num_steps", type=int, default=None, help="Override number of training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--load_unconditional", type=str, default=None, help="Path to unconditional checkpoint")

    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Train
    train(config, args)


if __name__ == "__main__":
    main()
