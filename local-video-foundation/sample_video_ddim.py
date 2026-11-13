"""
Video DDIM sampling for Stage 3.1.

Implements DDIM sampling for video generation with temporal modules.
"""

import os
import sys
import argparse
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
    progress_callback=None
):
    """
    DDIM sampling for video generation.

    Args:
        model: Video UNet model
        diffusion: Diffusion process
        text_emb: Text embeddings [batch, seq_len, text_dim]
        text_mask: Text attention mask [batch, seq_len]
        shape: Output shape [batch, channels, frames, height, width]
        num_sampling_steps: Number of DDIM steps
        device: Device
        clip_range: Clipping range for final output
        progress_callback: Optional progress callback

    Returns:
        Generated video tensor
    """
    batch_size, channels, num_frames, height, width = shape

    # Start from pure Gaussian noise
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


def sample_video(
    checkpoint_path,
    prompt,
    output_dir,
    num_frames=16,
    image_size=64,
    num_sampling_steps=25,
    device='cpu',
    seed=42
):
    """
    Sample a video from a trained checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        prompt: Text prompt
        output_dir: Output directory
        num_frames: Number of frames
        image_size: Image size
        num_sampling_steps: Number of DDIM steps
        device: Device
        seed: Random seed
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Reconstruct models
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
        use_temporal=True
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

    # Encode prompt
    token_ids, mask = video_unet.tokenizer.encode(prompt, max_length=32)
    max_len = len(token_ids)
    token_ids = token_ids + [0] * (32 - max_len)
    mask = mask + [0] * (32 - len(mask))

    text_emb_tokens = torch.tensor([token_ids]).to(device)
    text_mask = torch.tensor([mask]).to(device)

    with torch.no_grad():
        text_emb, _ = video_unet.text_encoder(text_emb_tokens, text_mask)

    # Create diffusion process
    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # Sample video
    print(f"Sampling video with {num_sampling_steps} DDIM steps...")
    shape = (1, 3, num_frames, image_size, image_size)

    start_time = time.time()

    with torch.no_grad():
        video = ddim_sample_video(
            model=video_unet,
            diffusion=diffusion,
            text_emb=text_emb,
            text_mask=text_mask,
            shape=shape,
            num_sampling_steps=num_sampling_steps,
            device=device,
            progress_callback=lambda i, n: print(f"Step {i+1}/{n}")
        )

    sampling_time = time.time() - start_time
    print(f"Sampling complete in {sampling_time:.2f}s")

    # Save frames
    video = video.squeeze(0)  # [channels, frames, h, w]
    video = video.permute(1, 2, 3, 0)  # [frames, h, w, channels]

    # Denormalize from [-1, 1] to [0, 255]
    video = ((video + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)

    for i, frame in enumerate(video):
        frame_img = Image.fromarray(frame)
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        frame_img.save(frame_path)

    print(f"Saved {num_frames} frames to {output_dir}")

    # Save metadata
    metadata = {
        'prompt': prompt,
        'num_frames': num_frames,
        'image_size': image_size,
        'num_sampling_steps': num_sampling_steps,
        'sampling_time': sampling_time,
        'seed': seed,
        'checkpoint': checkpoint_path
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_path}")

    return output_dir


def main():
    import time

    parser = argparse.ArgumentParser(description="Sample video from trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--prompt", type=str, default="A red circle moves from left to right.", help="Text prompt")
    parser.add_argument("--output_dir", type=str, default="outputs_video", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--num_steps", type=int, default=25, help="Number of DDIM steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sample_video(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        num_sampling_steps=args.num_steps,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
