"""
Stage 3.2.9B - Phase 2A: CPU Training Smoke Test

Controlled smoke test for Experiments B and C with performance profiling.

Author: Devin
Date: 2026-11-02
"""

import sys
import os
import json
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import psutil

# Add paths
parent_dir = Path(__file__).parent.parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))

from video_foundation.models.conditional_unet import ConditionalUNet as OriginalConditionalUNet
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed
from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder


class SyntheticVideoDataset(Dataset):
    """Synthetic video dataset with metadata."""

    def __init__(self, metadata_path: str, split: str = "train"):
        self.metadata_path = metadata_path
        self.split = split

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"Loaded {len(self.metadata)} samples for {split} split")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # Load video
        video = np.load(item['video_path'])  # [T, H, W, C]
        video = torch.from_numpy(video).float() / 255.0 * 2 - 1  # Normalize to [-1, 1]
        video = video.permute(3, 0, 1, 2)  # [C, T, H, W]

        # Load trajectory if available
        trajectory = None
        if 'trajectory_path' in item and os.path.exists(item['trajectory_path']):
            trajectory = np.load(item['trajectory_path'])
            trajectory = torch.from_numpy(trajectory).float()

        return {
            'video': video,
            'caption': item['caption'],
            'motion_type': item.get('motion_type', 'unknown'),
            'trajectory': trajectory,
            'id': item['id']
        }


class SmokeTestRunner:
    """Smoke test runner with performance profiling."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.parent_dir = self.experiment_dir.parent.parent

        # Checkpoint paths
        self.stage2_checkpoint = str(self.parent_dir / "checkpoints_conditional" / "checkpoint_final.pt")
        self.stage3_spatial_checkpoint = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_spatial.pt")
        self.tokenizer_path = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json")

        # Dataset paths
        self.train_metadata = str(self.parent_dir / "experiments_stage3_2_7" / "data_synthetic" / "train_metadata.json")
        self.val_metadata = str(self.parent_dir / "experiments_stage3_2_7" / "data_synthetic" / "val_metadata.json")

        # Config
        self.config = {
            'image_size': 64,
            'num_frames': 16,
            'num_timesteps': 1000,
            'batch_size': 1,
            'learning_rate': 0.0002,
            'weight_decay': 0.01,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'seed': 42
        }

        # Results
        self.results = {
            'experiment_B': {},
            'experiment_C': {}
        }

    def log_section(self, title: str):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def get_parameter_snapshot(self, model, split_encoder=None) -> Dict:
        """Get snapshot of parameters for integrity verification."""
        snapshot = {}

        if split_encoder:
            snapshot['motion_emb'] = split_encoder.token_emb.motion_emb.weight.data.clone()
            snapshot['appearance_emb'] = split_encoder.token_emb.original_emb.weight.data.clone()

        if hasattr(model, 'spatial_unet'):
            snapshot['spatial_unet'] = {
                name: param.data.clone()
                for name, param in model.spatial_unet.named_parameters()
            }

        if hasattr(model, 'temporal_modules'):
            snapshot['temporal_modules'] = {
                name: param.data.clone()
                for name, param in model.temporal_modules.named_parameters()
            }

        if hasattr(model, 'fusion_modules'):
            snapshot['fusion_modules'] = {
                name: param.data.clone()
                for name, param in model.fusion_modules.named_parameters()
            }

        return snapshot

    def compare_parameters(self, snapshot1: Dict, snapshot2: Dict, tolerance: float = 1e-10) -> Dict:
        """Compare two parameter snapshots."""
        comparison = {}

        for key in snapshot1.keys():
            if key in snapshot2:
                if isinstance(snapshot1[key], dict):
                    # Nested dict (e.g., spatial_unet)
                    comparison[key] = {}
                    for subkey in snapshot1[key].keys():
                        if subkey in snapshot2[key]:
                            diff = torch.max(torch.abs(snapshot1[key][subkey] - snapshot2[key][subkey])).item()
                            unchanged = diff < tolerance
                            comparison[key][subkey] = {
                                'max_diff': diff,
                                'unchanged': unchanged
                            }
                else:
                    # Tensor (e.g., motion_emb)
                    diff = torch.max(torch.abs(snapshot1[key] - snapshot2[key])).item()
                    unchanged = diff < tolerance
                    comparison[key] = {
                        'max_diff': diff,
                        'unchanged': unchanged
                    }

        return comparison

    def load_models(self):
        """Load models for smoke test."""
        self.log_section("LOADING MODELS")

        # Load checkpoint
        checkpoint = torch.load(self.stage2_checkpoint, map_location='cpu')
        actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

        # Create text encoder
        text_encoder = TextEncoder(
            vocab_size=actual_vocab_size, d_model=256, n_heads=4,
            n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0
        )
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

        # Create spatial UNet
        spatial_unet_modified = ConditionalUNet(
            in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
            num_res_blocks=2, attention_resolutions=[], dropout=0.0,
            num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
        )
        spatial_unet_modified.load_state_dict(checkpoint['model_state_dict'])

        # Create split encoder
        split_encoder = load_from_standard_encoder(
            text_encoder, motion_vocab_size=40, freeze_transformer=True
        )

        # Create Experiment B model (temporal disabled)
        video_unet_B = VideoUNetFixed(
            spatial_unet=spatial_unet_modified,
            text_encoder=split_encoder,
            num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
            freeze_spatial=True, use_temporal=True, enable_temporal_processing=False
        )

        # Create Experiment C model (temporal enabled)
        video_unet_C = VideoUNetFixed(
            spatial_unet=spatial_unet_modified,
            text_encoder=split_encoder,
            num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
            freeze_spatial=True, use_temporal=True, enable_temporal_processing=True
        )

        print(f"Model B parameters: {sum(p.numel() for p in video_unet_B.parameters()):,}")
        print(f"Model C parameters: {sum(p.numel() for p in video_unet_C.parameters()):,}")

        return text_encoder, video_unet_B, video_unet_C, split_encoder

    def get_diffusion_schedule(self):
        """Get diffusion beta schedule."""
        num_timesteps = self.config['num_timesteps']
        betas = torch.linspace(self.config['beta_start'], self.config['beta_end'], num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars

    def add_noise(self, x0, t):
        """Add noise to clean video at timestep t."""
        betas, alphas, alpha_bars = self.get_diffusion_schedule()
        sqrt_alpha_bar = torch.sqrt(alpha_bars[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t])

        # Reshape for broadcasting
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1)

        noise = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise

    def run_smoke_test(self, experiment_name: str, model, optimizer, split_encoder,
                      train_loader, tokenizer, num_steps: int = 10):
        """Run smoke test for a single experiment."""
        self.log_section(f"SMOKE TEST: EXPERIMENT {experiment_name}")

        # Set random seed
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

        # Initial parameter snapshot
        initial_snapshot = self.get_parameter_snapshot(model, split_encoder)

        # Timing
        timing = {
            'data_loading': [],
            'forward_pass': [],
            'backward_pass': [],
            'optimizer_step': [],
            'total_step': [],
            'peak_ram_gb': 0
        }

        # Training loop
        model.train()
        losses = []
        optimizer_step_count = 0

        for step in range(num_steps):
            step_start = time.time()
            step_start_ram = psutil.virtual_memory().used / (1024**3)

            print(f"\nStep {step + 1}/{num_steps}")

            # Data loading
            data_start = time.time()
            batch = next(iter(train_loader))
            data_time = time.time() - data_start
            timing['data_loading'].append(data_time)
            print(f"  Data loading: {data_time:.3f}s")

            videos = batch['video']  # [B, C, T, H, W]
            captions = batch['caption']

            # Random timesteps
            batch_size = videos.shape[0]
            t = torch.randint(0, self.config['num_timesteps'], (batch_size,))

            # Add noise
            xt, noise = self.add_noise(videos, t)

            # Encode text
            text_start = time.time()
            token_ids_list = []
            mask_list = []
            for caption in captions:
                token_ids, mask = tokenizer.encode(caption, max_length=64)
                token_ids_list.append(token_ids)
                mask_list.append(mask)

            token_ids = torch.tensor(token_ids_list)
            mask = torch.tensor(mask_list, dtype=torch.float)
            text_time = time.time() - text_start
            print(f"  Text encoding: {text_time:.3f}s")

            # Forward pass
            forward_start = time.time()
            text_emb, text_mask = model.text_encoder(token_ids, mask)
            predicted_noise = model(xt, t, text_emb, text_mask)
            forward_time = time.time() - forward_start
            timing['forward_pass'].append(forward_time)
            print(f"  Forward pass: {forward_time:.3f}s")

            # Compute loss
            loss = nn.functional.mse_loss(predicted_noise, noise)
            losses.append(loss.item())
            print(f"  Loss: {loss.item():.6f}")

            # Backward pass
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            backward_time = time.time() - backward_start
            timing['backward_pass'].append(backward_time)
            print(f"  Backward pass: {backward_time:.3f}s")

            # Optimizer step
            opt_start = time.time()
            optimizer.step()
            opt_time = time.time() - opt_start
            timing['optimizer_step'].append(opt_time)
            print(f"  Optimizer step: {opt_time:.3f}s")

            optimizer_step_count += 1

            # Update peak RAM
            current_ram = psutil.virtual_memory().used / (1024**3)
            timing['peak_ram_gb'] = max(timing['peak_ram_gb'], current_ram)

            # Total step time
            step_time = time.time() - step_start
            timing['total_step'].append(step_time)
            print(f"  Total step time: {step_time:.3f}s")

        # Final parameter snapshot
        final_snapshot = self.get_parameter_snapshot(model, split_encoder)

        # Compare parameters
        parameter_comparison = self.compare_parameters(initial_snapshot, final_snapshot)

        # Save checkpoint
        checkpoint_path = self.experiment_dir / f"checkpoint_smoke_test_{experiment_name}.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'split_encoder_state_dict': split_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'step': num_steps,
            'loss': losses[-1] if losses else 0.0
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"\nSaved checkpoint to {checkpoint_path}")

        # Test checkpoint loading
        print("Testing checkpoint loading...")
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(loaded_checkpoint['model_state_dict'])
        split_encoder.load_state_dict(loaded_checkpoint['split_encoder_state_dict'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        print("Checkpoint loading successful")

        return {
            'losses': losses,
            'optimizer_step_count': optimizer_step_count,
            'timing': timing,
            'parameter_comparison': parameter_comparison,
            'checkpoint_path': str(checkpoint_path)
        }

    def run_all_smoke_tests(self):
        """Run smoke tests for both experiments."""
        self.log_section("STAGE 3.2.9B - SMOKE TEST")

        # Load models
        text_encoder, video_unet_B, video_unet_C, split_encoder = self.load_models()

        # Load datasets
        print("\nLoading datasets...")
        train_dataset = SyntheticVideoDataset(self.train_metadata, "train")
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)

        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = SimpleTokenizer.load(self.tokenizer_path)

        # Setup optimizers
        print("\nSetting up optimizers...")

        # Experiment B: Only motion embeddings
        motion_params = list(split_encoder.token_emb.motion_emb.parameters())
        optimizer_B = optim.AdamW(motion_params, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        print(f"Experiment B trainable parameters: {sum(p.numel() for p in motion_params):,}")

        # Experiment C: Motion + temporal + fusion
        trainable_params_C = list(split_encoder.token_emb.motion_emb.parameters())
        trainable_params_C.extend(list(video_unet_C.temporal_modules.parameters()))
        trainable_params_C.extend(list(video_unet_C.fusion_modules.parameters()))
        optimizer_C = optim.AdamW(trainable_params_C, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        print(f"Experiment C trainable parameters: {sum(p.numel() for p in trainable_params_C):,}")

        # Run Experiment B
        self.results['experiment_B'] = self.run_smoke_test(
            'B', video_unet_B, optimizer_B, split_encoder, train_loader, tokenizer, num_steps=10
        )

        # Run Experiment C
        self.results['experiment_C'] = self.run_smoke_test(
            'C', video_unet_C, optimizer_C, split_encoder, train_loader, tokenizer, num_steps=10
        )

        # Save results
        results_path = self.experiment_dir / "STAGE3_2_9B_SMOKE_TEST_RESULTS.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved results to {results_path}")

        return self.results


def main():
    experiment_dir = Path(__file__).parent
    runner = SmokeTestRunner(str(experiment_dir))
    results = runner.run_all_smoke_tests()

    return 0


if __name__ == "__main__":
    sys.exit(main())
