"""
Text-to-image inference CLI for conditional diffusion model.
"""

import os
import argparse
import json
from PIL import Image
import torch
import yaml
from torchvision.utils import make_grid

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device


def save_image_grid(images: torch.Tensor, save_path: str, nrow: int = 4):
    """Save a grid of images."""
    images = (images + 1) / 2.0
    images = torch.clamp(images, 0, 1)
    images = (images * 255).byte()
    grid = make_grid(images, nrow=nrow, padding=2)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    grid = Image.fromarray(grid)
    grid.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Text-to-image inference")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/conditional_v0.yaml", help="Path to config file")

    # Generation
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sampling_steps", type=int, default=100, help="Number of DDPM sampling steps")
    parser.add_argument("--fast_sampling", action="store_true", help="Use fast sampling (every 10th step)")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs_conditional/inference", help="Output directory")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    set_seed(args.seed)

    print("="*70)
    print("TEXT-TO-IMAGE INFERENCE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")
    print(f"Number of images: {args.num_images}")
    print(f"Sampling steps: {args.sampling_steps}")
    print(f"Seed: {args.seed}")
    print("="*70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load tokenizer
    tokenizer_dir = os.path.dirname(args.checkpoint)
    tokenizer_path = os.path.join(tokenizer_dir, 'text_encoder_tokenizer.json')
    if not os.path.exists(tokenizer_path):
        # Try alternative location
        tokenizer_path = os.path.join(os.path.dirname(args.checkpoint), 'tokenizer', 'tokenizer.json')

    if os.path.exists(tokenizer_path):
        tokenizer = SimpleTokenizer.load(tokenizer_path)
    else:
        print("Warning: Tokenizer not found in checkpoint directory.")
        print("Building vocabulary from a default set...")
        tokenizer = SimpleTokenizer(vocab_size=1000, min_freq=1)
        tokenizer.build_vocab([args.prompt])

    # Load text encoder
    text_encoder = TextEncoder(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=config['text_encoder']['d_model'],
        n_heads=config['text_encoder']['n_heads'],
        n_layers=config['text_encoder']['n_layers'],
        d_ff=config['text_encoder']['d_ff'],
        max_seq_len=config['text_encoder']['max_seq_len'],
        dropout=0.0  # No dropout during inference
    )
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    # Load model
    model = ConditionalUNet(
        in_channels=config['model']['in_channels'],
        model_channels=config['model']['model_channels'],
        channel_mult=config['model']['channel_mult'],
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=config['model']['attention_resolutions'],
        dropout=0.0,
        num_timesteps=config['diffusion']['num_timesteps'],
        text_dim=config['text_encoder']['d_model'],
        n_heads=config['n_heads'],
        use_cross_attn=config['model']['use_cross_attn']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")

    # Create diffusion sampler
    diffusion = DiffusionSampler(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule=config['diffusion']['beta_schedule']
    )

    # Tokenize prompt
    print("\nTokenizing prompt...")
    max_len = config['text_encoder']['max_seq_len']
    token_ids, attention_mask = tokenizer.encode(args.prompt, max_length=max_len)
    token_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device).float()

    # Encode text
    print("Encoding text...")
    with torch.no_grad():
        text_emb, _ = text_encoder(token_ids, attention_mask)

    # Repeat for batch
    text_emb = text_emb.repeat(args.num_images, 1, 1)
    attention_mask = attention_mask.repeat(args.num_images, 1)

    # Generate images
    print(f"\nGenerating {args.num_images} images...")
    if args.fast_sampling:
        print("Using fast sampling (every 10th timestep)...")

        # Manual fast sampling
        batch_size = args.num_images
        x = torch.randn((batch_size, 3, 64, 64), device=device)
        timesteps = torch.arange(999, -1, -10, device=device, dtype=torch.long)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)
            with torch.no_grad():
                predicted_noise = model(x, t_batch, text_emb, attention_mask)

            # Get schedule values
            alpha, alpha_bar, one_minus_alpha_bar = diffusion.schedule.get(t_batch, device)
            beta = 1 - alpha

            sqrt_alpha = torch.sqrt(alpha)
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)

            # Predict x_0
            x_0_pred = (x - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar

            # Get previous alpha_bar
            if t > 0:
                alpha_bar_prev = diffusion.schedule.alphas_cumprod[t - 1].to(device)
            else:
                alpha_bar_prev = torch.tensor(1.0).to(device)

            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)

            # Compute posterior mean
            posterior_mean = (sqrt_alpha_bar_prev * beta / one_minus_alpha_bar) * x_0_pred + \
                           (sqrt_alpha * (1 - alpha_bar_prev) / one_minus_alpha_bar) * x

            # Compute posterior variance
            if t > 0:
                posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
            else:
                posterior_variance = torch.tensor(0.0).to(device)

            # Sample from posterior
            if t > 0:
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                x = posterior_mean

        samples = torch.clamp(x, -1.0, 1.0)

    else:
        # Full sampling
        with torch.no_grad():
            samples = diffusion.sample_ddpm(
                model=model,
                batch_size=args.num_images,
                channels=config['model']['in_channels'],
                image_size=config['model']['image_size'],
                device=device,
                text_emb=text_emb,
                text_mask=attention_mask
            )

    print(f"Generated images shape: {samples.shape}")
    print(f"Generated images range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"Generated images mean: {samples.mean():.4f}, std: {samples.std():.4f}")

    # Save images
    print("\nSaving images...")
    output_path = os.path.join(args.output_dir, f"prompt_{args.seed}.png")
    save_image_grid(samples, output_path, nrow=min(4, args.num_images))
    print(f"Saved to {output_path}")

    # Save metadata
    metadata = {
        'prompt': args.prompt,
        'num_images': args.num_images,
        'seed': args.seed,
        'sampling_steps': args.sampling_steps,
        'fast_sampling': args.fast_sampling,
        'config': config
    }
    metadata_path = os.path.join(args.output_dir, f"prompt_{args.seed}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
