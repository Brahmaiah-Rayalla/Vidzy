"""
DDIM Inference Script for Stage 2.3 - CPU-Optimized Text-to-Image Generation

Supports configurable DDIM sampling steps with thread optimization.
"""
import os
import sys
import argparse
import time
import json
import psutil
import torch
import yaml
from PIL import Image
from torchvision.utils import make_grid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device

# DDIM Sampling Presets (Stage 2.3.2 recommendations)
DDIM_PRESETS = {
    'fast': {'steps': 25, 'description': 'FAST: ~3s/img, interactive use'},
    'balanced': {'steps': 50, 'description': 'BALANCED: ~6s/img, recommended default'},
    'quality': {'steps': 100, 'description': 'QUALITY: ~12s/img, best quality'},
    'reference': {'steps': 200, 'description': 'REFERENCE: ~24s/img, marginal quality gain'},
}


def optimize_cpu_threads(num_threads=None):
    """Optimize PyTorch CPU thread configuration."""
    if num_threads is None:
        num_threads = psutil.cpu_count(logical=True)
    
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, num_threads // 2))
    
    print(f"CPU threads configured: {num_threads} main, {max(1, num_threads // 2)} interop")


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
    parser = argparse.ArgumentParser(description="DDIM inference for text-to-image generation")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/conditional_v0.yaml", help="Path to config")
    
    # Inference
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--preset", type=str, default="balanced", choices=list(DDIM_PRESETS.keys()),
                        help="DDIM preset (fast/balanced/quality/reference)")
    parser.add_argument("--steps", type=int, default=None, help="Override DDIM steps")
    parser.add_argument("--use_ddpm", action="store_true", help="Use DDPM instead of DDIM")
    
    # Optimization
    parser.add_argument("--threads", type=int, default=None, help="CPU thread count")
    parser.add_argument("--no_clip", action="store_true", help="Disable x0 clipping")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs_conditional", help="Output directory")
    
    args = parser.parse_args()
    
    # Optimize CPU threads
    optimize_cpu_threads(args.threads)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    set_seed(args.seed)
    
    print("="*70)
    print("DDIM INFERENCE - STAGE 2.3")
    print("="*70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt}")
    print(f"Images: {args.num_images}")
    
    # Determine sampling method
    if args.use_ddpm:
        sampling_method = "DDPM"
        num_steps = config['diffusion']['num_timesteps']
    else:
        sampling_method = "DDIM"
        if args.steps:
            num_steps = args.steps
        else:
            num_steps = DDIM_PRESETS[args.preset]['steps']
            print(f"Preset: {args.preset} ({DDIM_PRESETS[args.preset]['description']})")
    
    print(f"Sampling: {sampling_method} ({num_steps} steps)")
    print("="*70)
    
    # Load tokenizer
    tokenizer_dir = os.path.dirname(args.checkpoint)
    tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer', 'tokenizer.json')
    
    if os.path.exists(tokenizer_path):
        tokenizer = SimpleTokenizer.load(tokenizer_path)
    else:
        # Fallback: build from training data
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
    )
    text_encoder = text_encoder.to(device)
    
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
    )
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
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
    
    # Tokenize prompt
    max_len = config['text_encoder']['max_seq_len']
    token_ids, attention_mask = tokenizer.encode(args.prompt, max_length=max_len)
    token_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device).float()
    
    # Encode text
    with torch.no_grad():
        text_emb, _ = text_encoder(token_ids, attention_mask)
    
    # Expand for batch
    text_emb = text_emb.repeat(args.num_images, 1, 1)
    attention_mask = attention_mask.repeat(args.num_images, 1)
    
    # Generate
    print(f"\nGenerating {args.num_images} images...")
    mem_before = psutil.Process().memory_info().rss / (1024**2)
    start_time = time.time()
    
    with torch.no_grad():
        if args.use_ddpm:
            samples = diffusion.sample_ddpm(
                model=model,
                batch_size=args.num_images,
                channels=config['model']['in_channels'],
                image_size=config['model']['image_size'],
                device=device,
                text_emb=text_emb,
                text_mask=attention_mask
            )
        else:
            samples = diffusion.sample(
                model=model,
                batch_size=args.num_images,
                channels=config['model']['in_channels'],
                image_size=config['model']['image_size'],
                num_sampling_steps=num_steps,
                device=device,
                clip_x0=not args.no_clip,
                text_emb=text_emb,
                text_mask=attention_mask
            )
    
    elapsed = time.time() - start_time
    mem_after = psutil.Process().memory_info().rss / (1024**2)
    mem_used = mem_after - mem_before
    
    print(f"Generation complete in {elapsed:.2f}s")
    print(f"Peak memory: {mem_after:.2f} MB")
    print(f"Memory used: {mem_used:.2f} MB")
    print(f"Time per image: {elapsed/args.num_images:.2f}s")
    print(f"Throughput: {args.num_images/elapsed:.3f} images/sec")
    
    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save grid
    grid_path = os.path.join(args.output_dir, f"ddim_{args.preset if not args.use_ddpm else 'ddpm'}_grid.png")
    save_image_grid(samples, grid_path, nrow=4)
    print(f"Saved grid to {grid_path}")
    
    # Save individual images
    for i in range(args.num_images):
        img = (samples[i] + 1) / 2.0
        img = torch.clamp(img, 0, 1)
        img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        img_path = os.path.join(args.output_dir, f"ddim_{args.preset if not args.use_ddpm else 'ddpm'}_image_{i}.png")
        img.save(img_path)
    
    # Save metadata
    metadata = {
        'prompt': args.prompt,
        'sampling_method': sampling_method,
        'num_steps': num_steps,
        'num_images': args.num_images,
        'seed': args.seed,
        'elapsed_time_s': elapsed,
        'peak_memory_mb': mem_after,
        'throughput_imgs_per_sec': args.num_images / elapsed,
        'clip_x0': not args.no_clip,
        'preset': args.preset if not args.use_ddpm else None,
    }
    
    metadata_path = os.path.join(args.output_dir, f"ddim_{args.preset if not args.use_ddpm else 'ddpm'}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
