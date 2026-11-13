"""
Stage 3.2.9B - Controlled Temporal Learning Experiment

Training script for 3 experiment variants:
A: Spatial-only baseline (no training)
B: Motion-conditioning only (train motion embeddings)
C: Full temporal learning (train motion + temporal + fusion)

Author: Devin
Date: 2026-11-01
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
from collections import Counter
import time

# Add paths
parent_dir = Path(__file__).parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))

from video_foundation.models.conditional_unet import ConditionalUNet as OriginalConditionalUNet
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
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


class Stage3_2_9BExperiment:
    """Main experiment runner for Stage 3.2.9B."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.parent_dir = self.experiment_dir.parent

        # Checkpoint paths
        self.stage2_checkpoint = str(self.parent_dir / "checkpoints_conditional" / "checkpoint_final.pt")
        self.stage3_spatial_checkpoint = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_spatial.pt")
        self.stage3_motion_checkpoint = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_motion_expanded.pt")
        self.tokenizer_path = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json")

        # Dataset paths
        self.train_metadata = str(self.parent_dir / "experiments_stage3_2_7" / "data_synthetic" / "train_metadata.json")
        self.val_metadata = str(self.parent_dir / "experiments_stage3_2_7" / "data_synthetic" / "val_metadata.json")

        # Config
        self.config = {
            'image_size': 64,
            'num_frames': 16,
            'num_timesteps': 1000,
            'batch_size': 1,  # Reduced for CPU efficiency
            'learning_rate': 0.0002,
            'weight_decay': 0.01,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'seed': 42
        }

        # Set seed
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

        # Results storage
        self.results = {
            'experiment_A': {},
            'experiment_B': {},
            'experiment_C': {}
        }

    def log_section(self, title: str):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def get_file_hash(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return "FILE_NOT_FOUND"
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def record_checkpoint_hashes(self):
        """Record checkpoint hashes before training."""
        self.log_section("CHECKPOINT HASHES")

        hashes = {
            'stage2_checkpoint': self.get_file_hash(self.stage2_checkpoint),
            'stage3_spatial_checkpoint': self.get_file_hash(self.stage3_spatial_checkpoint),
            'stage3_motion_checkpoint': self.get_file_hash(self.stage3_motion_checkpoint),
            'tokenizer': self.get_file_hash(self.tokenizer_path)
        }

        for name, h in hashes.items():
            print(f"{name}: {h}")

        self.results['checkpoint_hashes'] = hashes
        return hashes

    def load_models(self) -> Tuple:
        """Load all required models."""
        self.log_section("LOADING MODELS")

        # Load checkpoint to get vocab size
        checkpoint = torch.load(self.stage2_checkpoint, map_location='cpu')
        actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

        # Create Stage 2.1 model (Experiment A)
        print("Creating Stage 2.1 model (Experiment A)...")
        spatial_unet_original = OriginalConditionalUNet(
            in_channels=3,
            model_channels=64,
            channel_mult=[1, 2, 2, 2],
            num_res_blocks=2,
            attention_resolutions=[],
            dropout=0.0,
            num_timesteps=self.config['num_timesteps'],
            text_dim=256,
            n_heads=4,
            use_cross_attn=True
        )

        text_encoder = TextEncoder(
            vocab_size=actual_vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=4,
            d_ff=512,
            max_seq_len=64,
            dropout=0.0
        )

        spatial_unet_original.load_state_dict(checkpoint['model_state_dict'])
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

        # Create Stage 3 models (Experiments B and C)
        print("Creating Stage 3 models (Experiments B and C)...")
        spatial_unet_modified = ConditionalUNet(
            in_channels=3,
            model_channels=64,
            channel_mult=[1, 2, 2, 2],
            num_res_blocks=2,
            attention_resolutions=[],
            dropout=0.0,
            num_timesteps=self.config['num_timesteps'],
            text_dim=256,
            n_heads=4,
            use_cross_attn=True
        )
        spatial_unet_modified.load_state_dict(checkpoint['model_state_dict'])

        # Create split embedding encoder
        split_encoder = load_from_standard_encoder(
            text_encoder,
            motion_vocab_size=40,
            freeze_transformer=True,
            original_vocab_size=42  # Original vocab size before motion expansion
        )

        # Create video models
        video_unet_B = VideoUNetFixed(
            spatial_unet=spatial_unet_modified,
            text_encoder=split_encoder,
            num_frames=self.config['num_frames'],
            image_size=self.config['image_size'],
            temporal_kernel=3,
            temporal_layers=2,
            freeze_spatial=True,
            use_temporal=True,
            enable_temporal_processing=False  # Temporal disabled for Experiment B
        )

        video_unet_C = VideoUNetFixed(
            spatial_unet=spatial_unet_modified,
            text_encoder=split_encoder,
            num_frames=self.config['num_frames'],
            image_size=self.config['image_size'],
            temporal_kernel=3,
            temporal_layers=2,
            freeze_spatial=True,
            use_temporal=True,
            enable_temporal_processing=True  # Temporal enabled for Experiment C
        )

        print(f"Model A parameters: {sum(p.numel() for p in spatial_unet_original.parameters()):,}")
        print(f"Model B parameters: {sum(p.numel() for p in video_unet_B.parameters()):,}")
        print(f"Model C parameters: {sum(p.numel() for p in video_unet_C.parameters()):,}")

        return spatial_unet_original, text_encoder, video_unet_B, video_unet_C, split_encoder

    def setup_optimizers(self, video_unet_B, video_unet_C, split_encoder):
        """Setup optimizers for Experiments B and C with hard assertions."""
        self.log_section("SETUP OPTIMIZERS")

        # Experiment B: Only motion embeddings trainable
        print("Experiment B: Motion embeddings only")
        motion_params = list(split_encoder.token_emb.motion_emb.parameters())
        optimizer_B = optim.AdamW(motion_params, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])

        # Hard assertion: No frozen spatial parameters in optimizer
        all_names_B = [name for name, _ in video_unet_B.named_parameters()]
        optimizer_param_ids_B = set(id(p) for p in optimizer_B.param_groups[0]['params'])
        for name, param in video_unet_B.named_parameters():
            if 'spatial' in name.lower() or 'unet' in name.lower():
                assert id(param) not in optimizer_param_ids_B, f"Frozen spatial parameter {name} in optimizer B"

        print(f"Experiment B trainable parameters: {sum(p.numel() for p in motion_params):,}")

        # Experiment C: Motion + temporal + fusion
        print("Experiment C: Motion + temporal + fusion")
        trainable_params_C = list(split_encoder.token_emb.motion_emb.parameters())
        trainable_params_C.extend(list(video_unet_C.temporal_modules.parameters()))
        trainable_params_C.extend(list(video_unet_C.fusion_modules.parameters()))

        optimizer_C = optim.AdamW(trainable_params_C, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])

        # Hard assertion: No frozen spatial parameters in optimizer
        all_names_C = [name for name, _ in video_unet_C.named_parameters()]
        optimizer_param_ids_C = set(id(p) for p in optimizer_C.param_groups[0]['params'])
        for name, param in video_unet_C.named_parameters():
            if 'spatial' in name.lower() or 'unet' in name.lower():
                assert id(param) not in optimizer_param_ids_C, f"Frozen spatial parameter {name} in optimizer C"

        print(f"Experiment C trainable parameters: {sum(p.numel() for p in trainable_params_C):,}")

        return optimizer_B, optimizer_C, motion_params, trainable_params_C

    def verify_integrity(self, models, initial_params):
        """Verify that frozen parameters remain unchanged."""
        # TODO: Implement integrity verification
        pass

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

        # Reshape to match video dimensions for broadcasting
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1)

        noise = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise

    def train_experiment(self, experiment_name: str, model, optimizer, num_steps: int, train_loader, val_loader, tokenizer):
        """Train a single experiment."""
        self.log_section(f"TRAINING EXPERIMENT {experiment_name}")

        if experiment_name == 'A':
            print("Experiment A: No training (baseline)")
            return {'loss': 0.0, 'steps': 0}

        model.train()
        losses = []
        val_losses = []
        gate_values = []

        for step in range(num_steps):
            step_start = time.time()
            epoch_losses = []

            for batch in train_loader:
                videos = batch['video']  # [B, C, T, H, W]
                captions = batch['caption']

                # Random timesteps
                batch_size = videos.shape[0]
                t = torch.randint(0, self.config['num_timesteps'], (batch_size,))

                # Add noise
                xt, noise = self.add_noise(videos, t)

                # Encode text
                token_ids_list = []
                mask_list = []
                for caption in captions:
                    token_ids, mask = tokenizer.encode(caption, max_length=64)
                    token_ids_list.append(token_ids)
                    mask_list.append(mask)

                token_ids = torch.tensor(token_ids_list)
                mask = torch.tensor(mask_list, dtype=torch.float)

                with torch.set_grad_enabled(experiment_name in ['B', 'C']):
                    # Forward pass
                    if experiment_name == 'A':
                        # Frame-independent - process each frame separately
                        B, C, T, H, W = xt.shape
                        outputs = []
                        for f in range(T):
                            xt_frame = xt[:, :, f, :, :]
                            t_expanded = t
                            text_emb, text_mask = text_encoder(token_ids, mask)
                            predicted_noise_frame = model(xt_frame, t_expanded, text_emb, text_mask)
                            outputs.append(predicted_noise_frame.unsqueeze(2))
                        predicted_noise = torch.cat(outputs, dim=2)
                    else:
                        text_emb, text_mask = model.text_encoder(token_ids, mask)
                        predicted_noise = model(xt, t, text_emb, text_mask)

                    # Compute loss
                    loss = nn.functional.mse_loss(predicted_noise, noise)
                    epoch_losses.append(loss.item())

                    if optimizer is not None:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Record gate values for Experiment C
                if experiment_name == 'C' and hasattr(model, 'get_gate_values'):
                    gate_values.append(model.get_gate_values())

                break  # One batch per step for CPU efficiency

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            losses.append(avg_loss)

            step_time = time.time() - step_start

            if step % 10 == 0:
                print(f"Step {step}/{num_steps}, Loss: {avg_loss:.6f}, Time: {step_time:.2f}s")
                if gate_values:
                    print(f"  Gate values: {gate_values[-1]}")

        # Validation
        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for batch in val_loader:
                videos = batch['video']
                captions = batch['caption']
                batch_size = videos.shape[0]
                t = torch.randint(0, self.config['num_timesteps'], (batch_size,))
                xt, noise = self.add_noise(videos, t)

                token_ids_list = []
                mask_list = []
                for caption in captions:
                    token_ids, mask = tokenizer.encode(caption, max_length=64)
                    token_ids_list.append(token_ids)
                    mask_list.append(mask)

                token_ids = torch.tensor(token_ids_list)
                mask = torch.tensor(mask_list, dtype=torch.float)

                if experiment_name == 'A':
                    B, C, T, H, W = xt.shape
                    outputs = []
                    for f in range(T):
                        xt_frame = xt[:, :, f, :, :]
                        t_expanded = t
                        text_emb, text_mask = text_encoder(token_ids, mask)
                        predicted_noise_frame = model(xt_frame, t_expanded, text_emb, text_mask)
                        outputs.append(predicted_noise_frame.unsqueeze(2))
                    predicted_noise = torch.cat(outputs, dim=2)
                else:
                    text_emb, text_mask = model.text_encoder(token_ids, mask)
                    predicted_noise = model(xt, t, text_emb, text_mask)

                loss = nn.functional.mse_loss(predicted_noise, noise)
                val_epoch_losses.append(loss.item())
                break

        avg_val_loss = np.mean(val_epoch_losses) if val_epoch_losses else 0.0
        val_losses.append(avg_val_loss)

        return {
            'loss': np.mean(losses) if losses else 0.0,
            'val_loss': avg_val_loss,
            'steps': num_steps,
            'losses': losses,
            'val_losses': val_losses,
            'gate_values': gate_values
        }

    def run_all_experiments(self):
        """Run all three experiments."""
        self.log_section("STAGE 3.2.9B - CONTROLLED TEMPORAL LEARNING")

        # Record checkpoint hashes
        self.record_checkpoint_hashes()

        # Load models
        spatial_unet_A, text_encoder, video_unet_B, video_unet_C, split_encoder = self.load_models()

        # Setup optimizers
        optimizer_B, optimizer_C, motion_params, trainable_params_C = self.setup_optimizers(
            video_unet_B, video_unet_C, split_encoder
        )

        # Load datasets
        print("\nLoading datasets...")
        train_dataset = SyntheticVideoDataset(self.train_metadata, "train")
        val_dataset = SyntheticVideoDataset(self.val_metadata, "val")

        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)

        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = SimpleTokenizer.load(self.tokenizer_path)

        # Run experiments
        self.results['experiment_A'] = self.train_experiment('A', spatial_unet_A, None, 0, train_loader, val_loader, tokenizer)
        self.results['experiment_B'] = self.train_experiment('B', video_unet_B, optimizer_B, 10, train_loader, val_loader, tokenizer)
        self.results['experiment_C'] = self.train_experiment('C', video_unet_C, optimizer_C, 10, train_loader, val_loader, tokenizer)

        # Save results
        self.save_results()

    def save_results(self):
        """Save experiment results."""
        results_path = self.experiment_dir / "STAGE3_2_9B_TRAINING_RESULTS.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved results to {results_path}")


def main():
    experiment_dir = Path(__file__).parent
    experiment = Stage3_2_9BExperiment(str(experiment_dir))
    experiment.run_all_experiments()


if __name__ == "__main__":
    main()
