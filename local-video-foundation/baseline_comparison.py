"""
Baseline comparison for Stage 3.2.

Compares frame-independent spatial model against spatial model with trained temporal modules.

Tests both models with identical prompts, sampling configurations, and initial noise.
Determines whether the temporal model demonstrates meaningful improvement in:
- Direction control
- Trajectory accuracy
- Frame consistency
- Temporal coherence
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


def ddim_sample_video(
    model,
    diffusion,
    text_emb,
    text_mask,
    shape,
    num_sampling_steps=25,
    device='cpu',
    clip_range=(-1.0, 1.0),
    progress_callback=None,
    initial_noise=None
):
    """
    DDIM sampling for video generation.

    Args:
        model: Video UNet model
        diffusion: Diffusion process
        text_emb: Text embeddings
        text_mask: Text attention mask
        shape: Output shape
        num_sampling_steps: Number of DDIM steps
        device: Device
        clip_range: Clipping range
        progress_callback: Progress callback
        initial_noise: Optional initial noise for reproducibility

    Returns:
        Generated video tensor
    """
    batch_size, channels, num_frames, height, width = shape

    # Use provided initial noise or generate random
    if initial_noise is not None:
        x = initial_noise.to(device)
    else:
        x = torch.randn(shape, device=device)

    # Create subsampled timesteps
    total_timesteps = diffusion.schedule.num_timesteps
    timesteps = torch.linspace(total_timesteps - 1, 0, num_sampling_steps, dtype=torch.long, device=device)

    # Reverse diffusion loop
    for i, t in enumerate(timesteps):
        if progress_callback:
            progress_callback(i, num_sampling_steps)

        # Predict noise at current timestep
        t_batch = t.expand(batch_size)
        predicted_noise = model(x, t_batch, text_emb, text_mask)

        # Get schedule values for current and previous timesteps
        _, alpha_bar_t, _ = diffusion.schedule.get(t_batch, device)

        # Previous timestep (or 0 for the last step)
        if i < num_sampling_steps - 1:
            t_prev = timesteps[i + 1]
            t_prev_batch = t_prev.expand(batch_size)
            _, alpha_bar_t_prev, _ = diffusion.schedule.get(t_prev_batch, device)
        else:
            alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

        # DDIM update
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)

        # Predict x_0
        x_0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

        # Compute x_{t-1}
        x = sqrt_alpha_bar_t_prev * x_0_pred + sqrt_one_minus_alpha_bar_t_prev * predicted_noise

    # Final clipping
    x = torch.clamp(x, clip_range[0], clip_range[1])

    return x


def load_model(checkpoint_path, num_frames=16, image_size=64, device='cpu', use_temporal=True):
    """Load the video model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load tokenizer
    tokenizer_path = "data/text_image/tokenizer/tokenizer.json"
    with open(tokenizer_path, 'r') as f:
        data = json.load(f)

    idx_to_word = {}
    word_to_idx = {}
    for key, value in data['idx_to_word'].items():
        if key.startswith('<'):
            idx = int(value)
            token = key
            idx_to_word[idx] = token
            word_to_idx[token] = idx
        else:
            idx = int(key)
            token = value
            idx_to_word[idx] = token
            word_to_idx[token] = idx

    vocab_size = len(word_to_idx)

    # Create spatial U-Net
    spatial_unet = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=1000,
        text_dim=256,
        n_heads=4,
        use_cross_attn=True
    )

    # Create text encoder
    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Create video U-Net
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=True,
        use_temporal=use_temporal
    )

    # Load model state
    video_unet.load_state_dict(checkpoint['model_state_dict'])
    video_unet = video_unet.to(device)
    video_unet.eval()

    # Create tokenizer
    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        )
    })()

    return video_unet


def compare_models(
    temporal_checkpoint_path,
    output_dir,
    prompts,
    num_frames=16,
    image_size=64,
    num_sampling_steps=10,
    device='cpu',
    seed=42
):
    """
    Compare temporal model against frame-independent baseline.

    Args:
        temporal_checkpoint_path: Path to temporal model checkpoint
        output_dir: Output directory
        prompts: List of text prompts to test
        num_frames: Number of frames
        image_size: Image size
        num_sampling_steps: Number of DDIM steps
        device: Device
        seed: Random seed
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directories
    temporal_dir = os.path.join(output_dir, "temporal")
    baseline_dir = os.path.join(output_dir, "baseline")
    os.makedirs(temporal_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)

    # Load temporal model
    print("=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print(f"Loading temporal model from {temporal_checkpoint_path}")
    temporal_model = load_model(temporal_checkpoint_path, num_frames, image_size, device, use_temporal=True)

    # Load baseline model (same checkpoint but disable temporal)
    print("Loading baseline model (frame-independent)")
    baseline_model = load_model(temporal_checkpoint_path, num_frames, image_size, device, use_temporal=False)

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    shape = (1, 3, num_frames, image_size, image_size)

    results = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nGenerating for prompt {prompt_idx + 1}/{len(prompts)}: {prompt}")

        # Encode prompt for temporal model
        token_ids, mask = temporal_model.tokenizer.encode(prompt, max_length=32)
        max_len = len(token_ids)
        token_ids = token_ids + [0] * (32 - max_len)
        mask = mask + [0] * (32 - len(mask))

        text_emb_tokens = torch.tensor([token_ids]).to(device)
        text_mask = torch.tensor([mask]).to(device)

        with torch.no_grad():
            text_emb, _ = temporal_model.text_encoder(text_emb_tokens, text_mask)

        # Generate initial noise (same for both models)
        initial_noise = torch.randn(shape, device=device)

        # Sample with temporal model
        print("  Sampling with temporal model...")
        start_time = time.time()

        with torch.no_grad():
            video_temporal = ddim_sample_video(
                model=temporal_model,
                diffusion=diffusion,
                text_emb=text_emb,
                text_mask=text_mask,
                shape=shape,
                num_sampling_steps=num_sampling_steps,
                device=device,
                initial_noise=initial_noise
            )

        temporal_time = time.time() - start_time
        print(f"    Temporal sampling time: {temporal_time:.2f}s")

        # Sample with baseline model
        print("  Sampling with baseline model...")
        start_time = time.time()

        with torch.no_grad():
            video_baseline = ddim_sample_video(
                model=baseline_model,
                diffusion=diffusion,
                text_emb=text_emb,
                text_mask=text_mask,
                shape=shape,
                num_sampling_steps=num_sampling_steps,
                device=device,
                initial_noise=initial_noise
            )

        baseline_time = time.time() - start_time
        print(f"    Baseline sampling time: {baseline_time:.2f}s")

        # Save temporal frames
        video_temporal_np = video_temporal.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        video_temporal_np = ((video_temporal_np + 1) * 127.5).clip(0, 255).astype(np.uint8)

        prompt_dir_name = f"prompt_{prompt_idx:03d}"
        temporal_prompt_dir = os.path.join(temporal_dir, prompt_dir_name)
        os.makedirs(temporal_prompt_dir, exist_ok=True)

        for i, frame in enumerate(video_temporal_np):
            frame_img = Image.fromarray(frame)
            frame_path = os.path.join(temporal_prompt_dir, f"frame_{i:04d}.png")
            frame_img.save(frame_path)

        # Save baseline frames
        video_baseline_np = video_baseline.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        video_baseline_np = ((video_baseline_np + 1) * 127.5).clip(0, 255).astype(np.uint8)

        baseline_prompt_dir = os.path.join(baseline_dir, prompt_dir_name)
        os.makedirs(baseline_prompt_dir, exist_ok=True)

        for i, frame in enumerate(video_baseline_np):
            frame_img = Image.fromarray(frame)
            frame_path = os.path.join(baseline_prompt_dir, f"frame_{i:04d}.png")
            frame_img.save(frame_path)

        # Compute difference between temporal and baseline
        diff = (video_temporal - video_baseline).abs().mean().item()

        results.append({
            'prompt': prompt,
            'temporal_time': temporal_time,
            'baseline_time': baseline_time,
            'mean_difference': diff,
            'temporal_dir': temporal_prompt_dir,
            'baseline_dir': baseline_prompt_dir
        })

        print(f"    Mean difference between models: {diff:.6f}")

    # Save results
    results_path = os.path.join(output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nComparison complete. Results saved to {results_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Baseline comparison")
    parser.add_argument("--checkpoint", type=str, required=True, help="Temporal model checkpoint")
    parser.add_argument("--output_dir", type=str, default="baseline_comparison_output", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--num_steps", type=int, default=10, help="DDIM steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Define test prompts
    prompts = [
        "A red circle moves left to right.",
        "A red circle moves right to left.",
        "A blue square moves diagonally.",
        "A yellow circle remains stationary."
    ]

    compare_models(
        temporal_checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        prompts=prompts,
        num_frames=args.num_frames,
        image_size=args.image_size,
        num_sampling_steps=args.num_steps,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
