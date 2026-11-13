"""
Stage 3.2.7B: Common Trainer

Shared training infrastructure for Experiments A, B, and C.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import time

# Add parent directory to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir.parent))

from experiments_stage3_2_7.training.parameter_groups import ParameterGroupManager
from experiments_stage3_2_7.training.checkpoint_manager import CheckpointManager


class SyntheticVideoDataset:
    """Load synthetic video dataset."""

    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir) / split
        # Allow absolute paths or resolve relative to script
        if not Path(data_dir).is_absolute():
            script_dir = Path(__file__).parent.parent
            self.data_dir = script_dir / data_dir / split
            self.metadata_path = script_dir / data_dir / f"{split}_metadata.json"
        else:
            self.metadata_path = Path(data_dir) / f"{split}_metadata.json"

        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.metadata[idx]

        # Load video
        video = np.load(item['video_path'])  # [frames, H, W, 3]
        video = torch.from_numpy(video).float() / 127.5 - 1.0  # Normalize to [-1, 1]
        video = video.permute(0, 3, 1, 2)  # [frames, C, H, W]

        # Load trajectory
        trajectory = np.load(item['trajectory_path'])  # [frames, 2]
        trajectory = torch.from_numpy(trajectory).float()

        return {
            'video': video,
            'trajectory': trajectory,
            'caption': item['caption'],
            'motion_type': item['motion_type'],
            'id': item['id']
        }


class DiffusionScheduler:
    """Simple diffusion scheduler."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1, 1)

        noisy_x = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        return noisy_x, noise


class CommonTrainer:
    """Common trainer for all experiments."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Dict,
        experiment_name: str,
        config: Dict[str, Any],
        device: str = 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.experiment_name = experiment_name
        self.config = config
        self.device = device

        # Initialize managers
        self.param_manager = ParameterGroupManager(model)
        checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_manager = CheckpointManager(experiment_name, checkpoint_dir)

        # Initialize diffusion scheduler
        self.diffusion = DiffusionScheduler(
            num_timesteps=config['num_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end']
        )

        # Setup datasets
        data_dir = config.get('data_dir', 'data_synthetic')
        self.train_dataset = SyntheticVideoDataset(data_dir, 'train')
        self.val_dataset = SyntheticVideoDataset(data_dir, 'val')

        # Training state
        self.step = 0
        self.best_val_loss = float('inf')

    def tokenize_caption(self, caption: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize caption using tokenizer."""
        word_to_idx = self.tokenizer['word_to_idx']

        # Handle both string and integer keys in word_to_idx
        unk_idx = word_to_idx.get('<UNK>', word_to_idx.get(1, 1))
        eos_idx = word_to_idx.get('<EOS>', word_to_idx.get(2, 2))
        pad_idx = word_to_idx.get('<PAD>', word_to_idx.get(0, 0))

        # Simple tokenization (split by spaces)
        words = caption.lower().replace('.', '').replace(',', '').split()
        tokens = [word_to_idx.get(word, unk_idx) for word in words]
        tokens = tokens[:self.config['max_seq_len'] - 1]  # Leave room for EOS
        tokens.append(eos_idx)

        # Pad
        tokens = tokens + [pad_idx] * (self.config['max_seq_len'] - len(tokens))
        tokens = tokens[:self.config['max_seq_len']]

        token_tensor = torch.tensor(tokens, dtype=torch.long)
        mask = (token_tensor != pad_idx).float()

        return token_tensor, mask

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move batch to device
        videos = batch['video'].to(self.device)  # [frames, C, H, W] for single item
        caption = batch['caption']

        # Add batch dimension and permute to [B, C, frames, H, W] for model
        videos = videos.unsqueeze(0).permute(0, 2, 1, 3, 4)

        batch_size = videos.shape[0]

        # Sample timesteps
        timesteps = torch.randint(0, self.config['num_timesteps'], (batch_size,), device=self.device)

        # Sample noise
        noise = torch.randn_like(videos)

        # Forward diffusion
        noisy_videos, _ = self.diffusion.q_sample(videos, timesteps, noise)

        # Tokenize caption
        tokens, mask = self.tokenize_caption(caption)
        text_emb = tokens.unsqueeze(0).to(self.device)
        text_mask = mask.unsqueeze(0).to(self.device)

        # Forward pass through text encoder
        text_enc, _ = self.model.text_encoder(text_emb, text_mask)

        # Predict noise
        predicted_noise = self.model(noisy_videos, timesteps, text_enc, text_mask)

        # Calculate loss
        loss = nn.functional.mse_loss(predicted_noise, noise)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def validate(self) -> Dict[str, float]:
        """Validation step."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataset:
                # Simplified: use single item for validation
                if num_batches >= 10:  # Limit validation batches
                    break

                # Process batch similarly to train_step
                video = batch['video'].to(self.device)  # [frames, C, H, W]
                caption = batch['caption']

                # Add batch dimension and permute to [B, C, frames, H, W]
                video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
                batch_size = video.shape[0]

                timesteps = torch.randint(0, self.config['num_timesteps'], (batch_size,), device=self.device)
                noise = torch.randn_like(video)
                noisy_videos, _ = self.diffusion.q_sample(video, timesteps, noise)

                tokens, mask = self.tokenize_caption(caption)
                text_emb, _ = self.model.text_encoder(tokens.unsqueeze(0).to(self.device), mask.unsqueeze(0).to(self.device))

                predicted_noise = self.model(noisy_videos, timesteps, text_emb, mask.unsqueeze(0).to(self.device))
                loss = nn.functional.mse_loss(predicted_noise, noise)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss}

    def train(self, num_steps: int, save_every: int = 100, val_every: int = 50):
        """Main training loop."""
        print(f"Starting training for {num_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Trainable parameters: {self.param_manager.count_parameters(trainable_only=True)}")

        for step in range(num_steps):
            # Sample random batch
            idx = torch.randint(0, len(self.train_dataset), (1,)).item()
            batch = self.train_dataset[idx]

            # Training step
            metrics = self.train_step(batch)
            self.step += 1

            # Logging
            if self.step % 10 == 0:
                print(f"Step {self.step}: loss = {metrics['loss']:.6f}")

            # Validation
            if self.step % val_every == 0:
                val_metrics = self.validate()
                print(f"Step {self.step}: val_loss = {val_metrics['val_loss']:.6f}")

                # Save best checkpoint
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, self.step, val_metrics['val_loss'],
                        is_best=True
                    )

            # Save checkpoint
            if self.step % save_every == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.step, metrics['loss']
                )

        print("Training complete!")

    def setup_optimizer(self, parameter_groups: Dict[str, nn.Parameter]):
        """Setup optimizer for trainable parameters."""
        trainable_params = list(parameter_groups.values())
        self.optimizer = optim.Adam(trainable_params, lr=self.config['learning_rate'])

    def setup_parameters(self, experiment_type: str):
        """Setup parameter groups based on experiment type."""
        if experiment_type == 'A':
            parameter_groups = self.param_manager.get_experiment_a_groups()
        elif experiment_type == 'B':
            parameter_groups = self.param_manager.get_experiment_b_groups()
        elif experiment_type == 'C':
            parameter_groups = self.param_manager.get_experiment_c_groups()
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        self.param_manager.freeze_all_except(parameter_groups)
        self.setup_optimizer(parameter_groups)

        # Print parameter summary
        summary = self.param_manager.get_parameter_summary()
        print("\nParameter Summary:")
        for module, counts in summary.items():
            print(f"  {module}: total={counts['total']}, trainable={counts['trainable']}")
