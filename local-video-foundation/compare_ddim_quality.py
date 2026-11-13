"""
Image Quality Comparison: DDPM vs DDIM for Stage 2.3.2

Compares image quality between DDPM (1000 steps) and DDIM (25/50/100/200 steps)
using controlled prompts and identical seeds where possible.
"""
import os
import sys
import torch
import yaml
import json
from PIL import Image
from torchvision.utils import make_grid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device


def compute_statistics(images):
    """Compute basic statistics for image quality assessment."""
    images_np = images.cpu().numpy()
    stats = {
        'mean': float(images_np.mean()),
        'std': float(images_np.std()),
        'min': float(images_np.min()),
        'max': float(images_np.max()),
        'per_channel_std': [float(images_np[:, i].std()) for i in range(images_np.shape[1])]
    }
    return stats


def run_quality_comparison():
    # Load config
    with open('configs/conditional_v0.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    set_seed(42)

    print("="*70)
    print("IMAGE QUALITY COMPARISON - STAGE 2.3.2")
    print("="*70)
    print(f"Device: {device}")
    print("="*70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    train_metadata_path = "data/text_image/train_metadata.json"
    with open(train_metadata_path, 'r') as f:
        train_metadata = json.load(f)
    captions = [m['caption'] for m in train_metadata]
    tokenizer = SimpleTokenizer(
        vocab_size=config['text_encoder']['vocab_size'],
        min_freq=config['text_encoder']['min_freq']
    )
    tokenizer.build_vocab(captions)

    # Load models
    print("\nLoading models...")
    text_encoder = TextEncoder(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=config['text_encoder']['d_model'],
        n_heads=config['text_encoder']['n_heads'],
        n_layers=config['text_encoder']['n_layers'],
        d_ff=config['text_encoder']['d_ff'],
        max_seq_len=config['text_encoder']['max_seq_len'],
        dropout=0.0
    ).to(device)

    model = ConditionalUNet(
        in_channels=config['model']['in_channels'],
        model_channels=config['model']['model_channels'],
        channel_mult=config['model']['channel_mult'],
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=config['model']['attention_resolutions'],
        dropout=0.0,
        num_timesteps=config['diffusion']['num_timesteps'],
        text_dim=config['text_encoder']['d_model'],
        n_heads=config.get('n_heads', 4),
        use_cross_attn=config['model']['use_cross_attn']
    ).to(device)

    checkpoint = torch.load("checkpoints_conditional/checkpoint_final.pt", map_location=device)
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    text_encoder.eval()
    model.eval()

    # Create diffusion sampler
    diffusion = DiffusionSampler(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule=config['diffusion']['beta_schedule']
    )

    # Test prompts covering different attributes
    test_prompts = [
        "A small red circle on a blue background.",
        "A large green triangle near the center.",
        "A yellow square on a white background.",
        "A blue circle on a black background.",
        "A small red square in the left corner.",
    ]

    # Configurations to test
    configs = [
        {'method': 'DDPM', 'steps': 1000},
        {'method': 'DDIM', 'steps': 25},
        {'method': 'DDIM', 'steps': 50},
        {'method': 'DDIM', 'steps': 100},
        {'method': 'DDIM', 'steps': 200},
    ]

    # Generate samples
    results = {}
    all_images = []

    for config_data in configs:
        method = config_data['method']
        steps = config_data['steps']

        print(f"\n{'='*70}")
        print(f"GENERATING: {method} {steps} steps")
        print(f"{'='*70}")

        config_key = f"{method}_{steps}"
        config_images = []
        config_stats = []

        for i, prompt in enumerate(test_prompts):
            print(f"\nPrompt {i+1}/{len(test_prompts)}: {prompt}")

            # Encode prompt
            max_len = config['text_encoder']['max_seq_len']
            token_ids, attention_mask = tokenizer.encode(prompt, max_length=max_len)
            token_ids = torch.tensor([token_ids], device=device)
            attention_mask = torch.tensor([attention_mask], device=device).float()

            with torch.no_grad():
                text_emb, _ = text_encoder(token_ids, attention_mask)

            # Generate with same seed for comparison
            set_seed(42 + i + steps)  # Unique seed per config and prompt

            with torch.no_grad():
                if method == 'DDPM':
                    samples = diffusion.sample_ddpm(
                        model=model,
                        batch_size=1,
                        channels=config['model']['in_channels'],
                        image_size=config['model']['image_size'],
                        device=device,
                        text_emb=text_emb,
                        text_mask=attention_mask
                    )
                else:
                    samples = diffusion.sample(
                        model=model,
                        batch_size=1,
                        channels=config['model']['in_channels'],
                        image_size=config['model']['image_size'],
                        num_sampling_steps=steps,
                        device=device,
                        clip_x0=False,
                        text_emb=text_emb,
                        text_mask=attention_mask
                    )

            stats = compute_statistics(samples)
            config_stats.append(stats)
            config_images.append(samples)

            print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")

        results[config_key] = {
            'method': method,
            'steps': steps,
            'stats': config_stats,
            'prompts': test_prompts
        }

        # Stack images for grid
        config_images_tensor = torch.cat(config_images, dim=0)
        all_images.append(config_images_tensor)

    # Create comparison grids
    print("\n" + "="*70)
    print("CREATING COMPARISON GRIDS")
    print("="*70)

    os.makedirs("outputs_conditional/quality_comparison", exist_ok=True)

    # Grid 1: All configurations side by side (for each prompt)
    for i, prompt in enumerate(test_prompts):
        prompt_images = [all_images[j][i:i+1] for j in range(len(all_images))]
        prompt_grid = torch.cat(prompt_images, dim=0)

        # Convert to image
        grid = (prompt_grid + 1) / 2.0
        grid = torch.clamp(grid, 0, 1)
        grid = (grid * 255).byte()
        grid = make_grid(grid, nrow=len(configs), padding=2)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = Image.fromarray(grid)

        # Add labels
        grid_path = f"outputs_conditional/quality_comparison/prompt_{i+1}_comparison.png"
        grid.save(grid_path)
        print(f"Saved: {grid_path}")

    # Grid 2: All prompts for each configuration
    for j, config_data in enumerate(configs):
        method = config_data['method']
        steps = config_data['steps']
        config_key = f"{method}_{steps}"

        config_grid = make_grid(all_images[j], nrow=len(test_prompts), padding=2)
        config_grid = (config_grid + 1) / 2.0
        config_grid = torch.clamp(config_grid, 0, 1)
        config_grid = (config_grid * 255).byte()
        config_grid = config_grid.permute(1, 2, 0).cpu().numpy()
        config_grid = Image.fromarray(config_grid)

        grid_path = f"outputs_conditional/quality_comparison/{config_key}_grid.png"
        config_grid.save(grid_path)
        print(f"Saved: {grid_path}")

    # Save statistics
    stats_path = "outputs_conditional/quality_comparison/statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")

    # Print summary statistics
    print("\n" + "="*70)
    print("QUALITY STATISTICS SUMMARY")
    print("="*70)
    print(f"{'Config':<15} {'Mean Std':<12} {'Mean Range':<15}")
    print("-"*70)

    for config_key, data in results.items():
        avg_std = np.mean([s['std'] for s in data['stats']])
        avg_range = np.mean([s['max'] - s['min'] for s in data['stats']])
        print(f"{config_key:<15} {avg_std:<12.3f} {avg_range:<15.3f}")

    print("="*70)

    return results


if __name__ == "__main__":
    results = run_quality_comparison()
