"""
Stage 3.2.9C - Matched Comparison Video Generation

Generate videos from all three checkpoints for comparison.

Author: Devin
Date: 2026-11-02
"""

import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List
import time

# Add paths
parent_dir = Path(__file__).parent.parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))

from video_foundation.models.conditional_unet import ConditionalUNet as OriginalConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
from models.conditional_unet import ConditionalUNet


class VideoGenerator:
    """Generate videos from checkpoints for comparison."""

    def __init__(self):
        self.parent_dir = parent_dir
        self.device = torch.device('cpu')

        # Checkpoint paths
        self.checkpoint_A = str(parent_dir / "checkpoints_conditional" / "checkpoint_final.pt")
        self.checkpoint_B = str(parent_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt")
        self.checkpoint_C = str(parent_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt")

        # Tokenizer config
        self.tokenizer_path = str(stage3_2_7_dir / "checkpoints" / "tokenizer_motion_expanded.json")

        # Output directory
        self.output_dir = Path(__file__).parent / "generated_videos"
        self.output_dir.mkdir(exist_ok=True)

        # Generation settings
        self.num_frames = 16
        self.resolution = 64
        self.num_inference_steps = 25
        self.guidance_scale = 1.0

        print("=" * 80)
        print("  STAGE 3.2.9C - VIDEO GENERATION")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Frames: {self.num_frames}")
        print(f"Resolution: {self.resolution}x{self.resolution}")
        print(f"Inference steps: {self.num_inference_steps}")

    def load_model_A(self):
        """Load Experiment A (spatial-only baseline)."""
        print("\nLoading Experiment A (spatial-only baseline)...")
        checkpoint = torch.load(self.checkpoint_A, map_location=self.device)

        # Use original ConditionalUNet for Stage 2.1 checkpoint
        # Model uses model_channels=64 based on checkpoint
        model = OriginalConditionalUNet(
            in_channels=3,
            model_channels=64,
            channel_mult=[1, 2, 2, 2],
            num_res_blocks=2,
            attention_resolutions=[16, 8],
            dropout=0.0,
            n_heads=4,
            use_cross_attn=True
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print("Experiment A loaded successfully")
        return model

    def load_model_B(self):
        """Load Experiment B (motion embeddings only)."""
        print("\nLoading Experiment B (motion embeddings only)...")
        checkpoint = torch.load(self.checkpoint_B, map_location=self.device)

        with open(self.tokenizer_path, 'r') as f:
            tokenizer_config = json.load(f)

        num_motion_tokens = tokenizer_config.get('num_tokens', 32)
        motion_token_dim = tokenizer_config.get('token_dim', 128)

        model = ConditionalUNet(
            in_channels=3,
            model_channels=64,  # Match checkpoint architecture
            channel_mult=[1, 2, 2, 2],
            num_res_blocks=2,
            attention_resolutions=[16, 8],
            dropout=0.0,
            n_heads=4,
            use_cross_attn=True,
            use_motion_tokens=True,
            num_motion_tokens=num_motion_tokens,
            motion_token_dim=motion_token_dim,
            use_temporal=False
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print("Experiment B loaded successfully")
        return model

    def load_model_C(self, enable_temporal=True):
        """Load Experiment C (motion + temporal)."""
        if enable_temporal:
            print("\nLoading Experiment C (motion + temporal enabled)...")
        else:
            print("\nLoading Experiment C (motion + temporal disabled)...")

        checkpoint = torch.load(self.checkpoint_C, map_location=self.device)

        with open(self.tokenizer_path, 'r') as f:
            tokenizer_config = json.load(f)

        num_motion_tokens = tokenizer_config.get('num_tokens', 32)
        motion_token_dim = tokenizer_config.get('token_dim', 128)

        model = ConditionalUNet(
            in_channels=3,
            model_channels=64,  # Match checkpoint architecture
            channel_mult=[1, 2, 2, 2],
            num_res_blocks=2,
            attention_resolutions=[16, 8],
            dropout=0.0,
            n_heads=4,
            use_cross_attn=True,
            use_motion_tokens=True,
            num_motion_tokens=num_motion_tokens,
            motion_token_dim=motion_token_dim,
            use_temporal=enable_temporal
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print(f"Experiment C loaded successfully (temporal={'enabled' if enable_temporal else 'disabled'})")
        return model

    def load_text_encoder(self):
        """Load text encoder."""
        print("\nLoading text encoder...")
        text_encoder = TextEncoder(
            model_name='openai/clip-vit-base-patch32',
            device=self.device
        )
        print("Text encoder loaded successfully")
        return text_encoder

    def generate_video(self, model, text_encoder, prompt: str, seed: int, config_name: str):
        """Generate a single video."""
        print(f"\nGenerating video for {config_name}: {prompt} (seed={seed})")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Encode text
        text_embedding = text_encoder.encode_text([prompt])[0]  # [D]

        # Create motion token (use token 0 for simplicity)
        motion_token = torch.zeros(1, 32, 128)
        motion_token[0, 0, :] = 1.0

        # DDIM sampling
        T = 1000
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Start from random noise
        x = torch.randn(1, 3, self.num_frames, self.resolution, self.resolution).to(self.device)

        # Sampling loop
        timesteps = torch.linspace(T-1, 0, self.num_inference_steps).long().to(self.device)

        for t in timesteps:
            t_tensor = torch.full((1,), t, device=self.device)

            with torch.no_grad():
                if config_name == 'A':
                    # Experiment A: no motion tokens
                    noise_pred = model(x, t_tensor, text_embedding=text_embedding.unsqueeze(0))
                else:
                    # Experiments B and C: with motion tokens
                    noise_pred = model(x, t_tensor, text_embedding=text_embedding.unsqueeze(0), motion_tokens=motion_token)

            # DDIM update
            alpha_t = alphas_cumprod[t]
            alpha_t_prev = alphas_cumprod[timesteps[max(0, timesteps.tolist().index(t) - 1)]]

            x_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            x = alpha_t_prev.sqrt() * x_pred + (1 - alpha_t_prev).sqrt() * noise_pred

        # Convert to [0, 1] range
        x = (x + 1) / 2
        x = x.clamp(0, 1)

        # Save as numpy
        video_np = x[0].cpu().numpy()  # [C, T, H, W]
        video_np = np.transpose(video_np, (1, 2, 3, 0))  # [T, H, W, C]
        video_np = (video_np * 255).astype(np.uint8)

        return video_np

    def generate_all_videos(self):
        """Generate videos for all configurations."""
        # Define prompts
        prompts = [
            "Move left",
            "Move right",
            "Move upward",
            "Move downward",
            "Move slowly",
            "Move quickly",
            "Bounce",
            "Remain stationary"
        ]

        # Load models
        model_A = self.load_model_A()
        model_B = self.load_model_B()
        model_C_enabled = self.load_model_C(enable_temporal=True)
        model_C_disabled = self.load_model_C(enable_temporal=False)

        # Load text encoder
        text_encoder = self.load_text_encoder()

        # Generate videos
        results = []

        for i, prompt in enumerate(prompts):
            print(f"\n{'='*80}")
            print(f"  PROMPT {i+1}/{len(prompts)}: {prompt}")
            print(f"{'='*80}")

            seed = 42 + i

            # Generate for each configuration
            configs = [
                ('A', model_A),
                ('B', model_B),
                ('C_enabled', model_C_enabled),
                ('C_disabled', model_C_disabled)
            ]

            for config_name, model in configs:
                start_time = time.time()
                video = self.generate_video(model, text_encoder, prompt, seed, config_name)
                duration = time.time() - start_time

                # Save video
                filename = f"{config_name}_{i:02d}_{prompt.replace(' ', '_')}.npy"
                output_path = self.output_dir / filename
                np.save(output_path, video)

                result = {
                    "prompt": prompt,
                    "config": config_name,
                    "seed": seed,
                    "video_path": str(output_path),
                    "generation_time": duration,
                    "frames": self.num_frames,
                    "resolution": self.resolution
                }
                results.append(result)

                print(f"Saved to {output_path} ({duration:.2f}s)")

        # Save results
        results_path = self.output_dir / "generation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"  GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total videos generated: {len(results)}")
        print(f"Results saved to {results_path}")


def main():
    generator = VideoGenerator()
    generator.generate_all_videos()


if __name__ == "__main__":
    main()
