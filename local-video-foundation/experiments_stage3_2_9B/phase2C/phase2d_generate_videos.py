"""
Stage 3.2.9B - Phase 2D: Video Generation for Evaluation

Generate prompt-controlled comparison videos for Experiments A, B, and C.

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
from typing import Dict, List, Optional
import time

# Add paths
parent_dir = Path(__file__).parent.parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))

from video_foundation.models.conditional_unet import ConditionalUNet as OriginalConditionalUNet
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed
from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder


class VideoGenerator:
    """Generate videos from trained checkpoints."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        # experiment_dir is experiments_stage3_2_9B, parent is local-video-foundation
        self.parent_dir = self.experiment_dir.parent
        self.phase2c_dir = self.experiment_dir / "phase2C"

        print(f"experiment_dir: {self.experiment_dir}")
        print(f"parent_dir: {self.parent_dir}")

        # Paths
        self.stage2_checkpoint = str(self.parent_dir / "checkpoints_conditional" / "checkpoint_final.pt")
        self.stage3_spatial_checkpoint = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_spatial.pt")
        self.tokenizer_path = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json")
        self.val_metadata = str(self.parent_dir / "experiments_stage3_2_7" / "data_synthetic" / "val_metadata.json")

        print(f"val_metadata path: {self.val_metadata}")
        print(f"val_metadata exists: {os.path.exists(self.val_metadata)}")

        # Verify paths exist
        if not os.path.exists(self.val_metadata):
            raise FileNotFoundError(f"val_metadata not found at {self.val_metadata}")

        # Checkpoint paths for Experiments B and C
        self.checkpoint_B = str(self.phase2c_dir / "checkpoints" / "experiment_B_final.pt")
        self.checkpoint_C = str(self.phase2c_dir / "checkpoints" / "experiment_C_final.pt")

        # Output directory
        self.output_dir = self.phase2c_dir / "generated_videos"
        self.output_dir.mkdir(exist_ok=True)

        # Device
        self.device = torch.device('cpu')

        # Generator (simple DDIM sampling)
        self.num_inference_steps = 20
        self.guidance_scale = 1.0

    def log_section(self, title: str):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def load_tokenizer(self):
        """Load motion tokenizer."""
        with open(self.tokenizer_path, 'r') as f:
            tokenizer_config = json.load(f)
        return tokenizer_config

    def load_model_A(self):
        """Load Experiment A (spatial-only baseline)."""
        print("Loading Experiment A (spatial-only baseline)...")

        # Load Stage 2 checkpoint as baseline
        checkpoint = torch.load(self.stage2_checkpoint, map_location=self.device)
        model = OriginalConditionalUNet(
            in_channels=3,
            model_channels=128,
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

        print(f"Loaded Experiment A from {self.stage2_checkpoint}")
        return model

    def load_model_B(self):
        """Load Experiment B (motion embeddings only)."""
        print("Loading Experiment B (motion embeddings only)...")

        checkpoint = torch.load(self.checkpoint_B, map_location=self.device)

        # Build model with motion embeddings but temporal disabled
        tokenizer_config = self.load_tokenizer()
        num_motion_tokens = tokenizer_config.get('num_tokens', 32)
        motion_token_dim = tokenizer_config.get('token_dim', 128)

        model = ConditionalUNet(
            in_channels=3,
            model_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_res_blocks=2,
            attention_resolutions=[16, 8],
            dropout=0.0,
            n_heads=4,
            use_cross_attn=True,
            use_motion_tokens=True,
            num_motion_tokens=num_motion_tokens,
            motion_token_dim=motion_token_dim,
            use_temporal=False  # Temporal disabled for Experiment B
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print(f"Loaded Experiment B from {self.checkpoint_B}")
        return model

    def load_model_C(self):
        """Load Experiment C (motion + temporal learning)."""
        print("Loading Experiment C (motion + temporal learning)...")

        checkpoint = torch.load(self.checkpoint_C, map_location=self.device)

        # Build model with full temporal learning
        tokenizer_config = self.load_tokenizer()
        num_motion_tokens = tokenizer_config.get('num_tokens', 32)
        motion_token_dim = tokenizer_config.get('token_dim', 128)

        model = ConditionalUNet(
            in_channels=3,
            model_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_res_blocks=2,
            attention_resolutions=[16, 8],
            dropout=0.0,
            n_heads=4,
            use_cross_attn=True,
            use_motion_tokens=True,
            num_motion_tokens=num_motion_tokens,
            motion_token_dim=motion_token_dim,
            use_temporal=True  # Temporal enabled for Experiment C
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print(f"Loaded Experiment C from {self.checkpoint_C}")
        return model

    def load_text_encoder(self):
        """Load text encoder."""
        print("Loading text encoder...")

        text_encoder = TextEncoder(
            model_name='openai/clip-vit-base-patch32',
            device=self.device
        )

        print("Loaded text encoder")
        return text_encoder

    def generate_video(self, model, text_encoder, caption: str, motion_token_id: int, experiment_name: str, video_id: int):
        """Generate a single video."""
        print(f"Generating video {video_id} for {experiment_name}: {caption}")

        # Encode text
        text_embedding = text_encoder.encode_text([caption])[0]  # [D]

        # Encode motion token
        motion_token = torch.zeros(1, 32, 128)  # [1, num_tokens, token_dim]
        motion_token[0, motion_token_id, :] = 1.0  # One-hot encoding

        # Simple DDIM sampling (simplified for speed)
        T = 1000
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Start from random noise
        num_frames = 8
        H, W = 64, 64
        x = torch.randn(1, 3, num_frames, H, W).to(self.device)

        # Sampling loop
        timesteps = torch.linspace(T-1, 0, self.num_inference_steps).long().to(self.device)

        for t in timesteps:
            t_tensor = torch.full((1,), t, device=self.device)

            # Predict noise
            with torch.no_grad():
                if experiment_name == 'A':
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

        # Save video
        video_np = x[0].cpu().numpy()  # [C, T, H, W]
        video_np = np.transpose(video_np, (1, 2, 3, 0))  # [T, H, W, C]
        video_np = (video_np * 255).astype(np.uint8)

        output_path = self.output_dir / f"{experiment_name}_video_{video_id:03d}.npy"
        np.save(output_path, video_np)

        print(f"Saved video to {output_path}")
        return str(output_path)

    def generate_all_videos(self):
        """Generate videos for all experiments."""
        self.log_section("PHASE 2D: VIDEO GENERATION")

        # Load validation metadata
        with open(self.val_metadata, 'r') as f:
            val_metadata = json.load(f)

        print(f"Loaded {len(val_metadata)} validation clips")

        # Load models
        model_A = self.load_model_A()
        model_B = self.load_model_B()
        model_C = self.load_model_C()

        # Load text encoder
        text_encoder = self.load_text_encoder()

        # Generate videos for first 5 clips per experiment (for speed)
        num_clips = 5

        results = {
            'experiment_A': [],
            'experiment_B': [],
            'experiment_C': []
        }

        for i, clip_data in enumerate(val_metadata[:num_clips]):
            print(f"\n--- Processing clip {i+1}/{num_clips} ---")
            caption = clip_data['caption']
            motion_type = clip_data.get('motion_type', 'stationary')

            # Map motion type to token ID (simplified)
            motion_token_map = {
                'stationary': 0,
                'horizontal': 1,
                'vertical': 2,
                'diagonal': 3
            }
            motion_token_id = motion_token_map.get(motion_type, 0)

            # Generate for each experiment
            video_A = self.generate_video(model_A, text_encoder, caption, motion_token_id, 'A', i)
            video_B = self.generate_video(model_B, text_encoder, caption, motion_token_id, 'B', i)
            video_C = self.generate_video(model_C, text_encoder, caption, motion_token_id, 'C', i)

            results['experiment_A'].append({
                'clip_id': clip_data['id'],
                'video_path': video_A,
                'caption': caption,
                'motion_type': motion_type
            })
            results['experiment_B'].append({
                'clip_id': clip_data['id'],
                'video_path': video_B,
                'caption': caption,
                'motion_type': motion_type
            })
            results['experiment_C'].append({
                'clip_id': clip_data['id'],
                'video_path': video_C,
                'caption': caption,
                'motion_type': motion_type
            })

        # Save results
        results_path = self.output_dir / "generation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved generation results to {results_path}")
        print(f"Generated {num_clips} videos per experiment")


def main():
    experiment_dir = Path(__file__).parent.parent
    generator = VideoGenerator(str(experiment_dir))
    generator.generate_all_videos()


if __name__ == "__main__":
    main()
