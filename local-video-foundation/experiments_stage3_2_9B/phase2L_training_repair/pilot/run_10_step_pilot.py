"""
Controlled 10-step pilot for Experiments B and C.

Configuration:
- Batch size: 1
- Resolution: 64x64
- Frames: 16
- CPU only
- Fixed random seed
- Exactly 10 optimizer steps per experiment
"""

import torch
import torch.optim as optim
import json
from pathlib import Path
import sys
import time
import numpy as np

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
stage3_2_7_dir = base_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation" / "models"))

from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed
from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder
from models.text_encoder import TextEncoder, SimpleTokenizer
from torch.utils.data import Dataset, DataLoader

class SyntheticVideoDataset(Dataset):
    """Synthetic video dataset for pilot training."""

    def __init__(self, metadata_path: str):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        video = np.load(item['video_path'])
        video = torch.from_numpy(video).float() / 255.0 * 2 - 1
        video = video.permute(3, 0, 1, 2)
        return {
            'video': video,
            'caption': item['caption']
        }

def get_diffusion_schedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    """Get diffusion beta schedule."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def add_noise(x0, t, alpha_bars):
    """Add noise to clean video at timestep t."""
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t])
    sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1)
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt, noise

def run_10_step_pilot():
    """Run controlled 10-step pilot for both experiments."""
    print("=" * 80)
    print("STAGE 3.2.9L.2 - CONTROLLED 10-STEP PILOT")
    print("=" * 80)

    # Configuration
    config = {
        'image_size': 64,
        'num_frames': 16,
        'num_timesteps': 1000,
        'batch_size': 1,
        'learning_rate': 0.0002,
        'weight_decay': 0.01,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'seed': 42,
        'num_steps': 10
    }

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Load checkpoint
    stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location='cpu')
    actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

    # Create standard text encoder
    text_encoder_standard = TextEncoder(
        vocab_size=actual_vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64,
        dropout=0.0
    )
    text_encoder_standard.load_state_dict(checkpoint['text_encoder_state_dict'])

    # Load tokenizer
    tokenizer_path = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    # Load dataset
    train_metadata = base_dir / "experiments_stage3_2_7/data_synthetic/train_metadata.json"
    dataset = SyntheticVideoDataset(str(train_metadata))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Get diffusion schedule
    betas, alphas, alpha_bars = get_diffusion_schedule(
        config['num_timesteps'],
        config['beta_start'],
        config['beta_end']
    )

    results = {
        'config': config,
        'experiment_B': {},
        'experiment_C': {}
    }

    # ============================================================================
    # EXPERIMENT B: Motion embeddings only
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT B - Motion Embeddings Only (10 steps)")
    print("=" * 80)

    # Create models
    spatial_unet_B = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=config['num_timesteps'],
        text_dim=256,
        n_heads=4,
        use_cross_attn=True
    )
    spatial_unet_B.load_state_dict(checkpoint['model_state_dict'])

    split_encoder_B = load_from_standard_encoder(
        text_encoder_standard,
        motion_vocab_size=40,
        freeze_transformer=True,
        original_vocab_size=42
    )

    video_unet_B = VideoUNetFixed(
        spatial_unet=spatial_unet_B,
        text_encoder=split_encoder_B,
        num_frames=config['num_frames'],
        image_size=config['image_size'],
        temporal_kernel=3,
        temporal_layers=2,
        freeze_spatial=True,
        use_temporal=True,
        enable_temporal_processing=False
    )

    # Setup optimizer
    motion_params_B = list(split_encoder_B.token_emb.motion_emb.parameters())
    optimizer_B = optim.AdamW(motion_params_B, lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Training loop
    video_unet_B.train()
    dataloader_iter = iter(dataloader)

    results['experiment_B']['steps'] = []

    for step in range(config['num_steps']):
        step_start = time.time()

        # Get batch
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        videos = batch['video']
        captions = batch['caption']

        # Random timesteps
        batch_size = videos.shape[0]
        t = torch.randint(0, config['num_timesteps'], (batch_size,))

        # Add noise
        xt, noise = add_noise(videos, t, alpha_bars)

        # Encode text
        token_ids_list = []
        mask_list = []
        for caption in captions:
            token_ids, mask = tokenizer.encode(caption, max_length=64)
            token_ids_list.append(token_ids)
            mask_list.append(mask)

        token_ids = torch.tensor(token_ids_list)
        mask = torch.tensor(mask_list, dtype=torch.float)

        # Forward pass
        text_emb, text_mask = video_unet_B.text_encoder(token_ids, mask)
        predicted_noise = video_unet_B(xt, t, text_emb, text_mask)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Backward pass
        optimizer_B.zero_grad()
        loss.backward()

        # Check gradients
        motion_grad_norm = sum(p.grad.norm().item() for p in motion_params_B if p.grad is not None)

        # Optimizer step
        optimizer_B.step()

        # Monitor gate values (not applicable for B)
        gate_values = {'bottleneck_gate': None, 'low_res_gate': None}

        step_time = time.time() - step_start

        # Record step
        step_result = {
            'step': step,
            'loss': loss.item(),
            'motion_grad_norm': motion_grad_norm,
            'gate_values': gate_values,
            'step_time': step_time
        }

        results['experiment_B']['steps'].append(step_result)

        print(f"Step {step}/{config['num_steps']}, Loss: {loss.item():.6f}, Motion grad norm: {motion_grad_norm:.6f}, Time: {step_time:.2f}s")

    # ============================================================================
    # EXPERIMENT C: Motion + Temporal + Fusion
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT C - Motion + Temporal + Fusion (10 steps)")
    print("=" * 80)

    # Recreate models
    spatial_unet_C = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=config['num_timesteps'],
        text_dim=256,
        n_heads=4,
        use_cross_attn=True
    )
    spatial_unet_C.load_state_dict(checkpoint['model_state_dict'])

    split_encoder_C = load_from_standard_encoder(
        text_encoder_standard,
        motion_vocab_size=40,
        freeze_transformer=True,
        original_vocab_size=42
    )

    video_unet_C = VideoUNetFixed(
        spatial_unet=spatial_unet_C,
        text_encoder=split_encoder_C,
        num_frames=config['num_frames'],
        image_size=config['image_size'],
        temporal_kernel=3,
        temporal_layers=2,
        freeze_spatial=True,
        use_temporal=True,
        enable_temporal_processing=True
    )

    # Setup optimizer
    motion_params_C = list(split_encoder_C.token_emb.motion_emb.parameters())
    temporal_params_C = list(video_unet_C.temporal_modules.parameters())
    fusion_params_C = list(video_unet_C.fusion_modules.parameters())
    trainable_params_C = motion_params_C + temporal_params_C + fusion_params_C

    optimizer_C = optim.AdamW(trainable_params_C, lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Training loop
    video_unet_C.train()
    dataloader_iter = iter(dataloader)

    results['experiment_C']['steps'] = []

    for step in range(config['num_steps']):
        step_start = time.time()

        # Get batch
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        videos = batch['video']
        captions = batch['caption']

        # Random timesteps
        batch_size = videos.shape[0]
        t = torch.randint(0, config['num_timesteps'], (batch_size,))

        # Add noise
        xt, noise = add_noise(videos, t, alpha_bars)

        # Encode text
        token_ids_list = []
        mask_list = []
        for caption in captions:
            token_ids, mask = tokenizer.encode(caption, max_length=64)
            token_ids_list.append(token_ids)
            mask_list.append(mask)

        token_ids = torch.tensor(token_ids_list)
        mask = torch.tensor(mask_list, dtype=torch.float)

        # Forward pass
        text_emb, text_mask = video_unet_C.text_encoder(token_ids, mask)
        predicted_noise = video_unet_C(xt, t, text_emb, text_mask)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Backward pass
        optimizer_C.zero_grad()
        loss.backward()

        # Check gradients
        motion_grad_norm = sum(p.grad.norm().item() for p in motion_params_C if p.grad is not None)
        temporal_grad_norm = sum(p.grad.norm().item() for p in temporal_params_C if p.grad is not None)
        fusion_grad_norm = sum(p.grad.norm().item() for p in fusion_params_C if p.grad is not None)

        # Optimizer step
        optimizer_C.step()

        # Monitor gate values
        gate_values = video_unet_C.get_gate_values()

        step_time = time.time() - step_start

        # Record step
        step_result = {
            'step': step,
            'loss': loss.item(),
            'motion_grad_norm': motion_grad_norm,
            'temporal_grad_norm': temporal_grad_norm,
            'fusion_grad_norm': fusion_grad_norm,
            'gate_values': gate_values,
            'step_time': step_time
        }

        results['experiment_C']['steps'].append(step_result)

        print(f"Step {step}/{config['num_steps']}, Loss: {loss.item():.6f}, Motion grad: {motion_grad_norm:.6f}, Temporal grad: {temporal_grad_norm:.6f}, Fusion grad: {fusion_grad_norm:.6f}, Time: {step_time:.2f}s")
        print(f"  Gate values: {gate_values}")

    # Save checkpoints
    print("\n" + "=" * 80)
    print("SAVING CHECKPOINTS")
    print("=" * 80)

    checkpoint_B = {
        'model_state_dict': video_unet_B.state_dict(),
        'optimizer_state_dict': optimizer_B.state_dict(),
        'config': config,
        'experiment': 'B'
    }

    checkpoint_C = {
        'model_state_dict': video_unet_C.state_dict(),
        'optimizer_state_dict': optimizer_C.state_dict(),
        'config': config,
        'experiment': 'C'
    }

    checkpoint_B_path = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/checkpoint_experiment_B_10steps.pt"
    checkpoint_C_path = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/checkpoint_experiment_C_10steps.pt"

    torch.save(checkpoint_B, checkpoint_B_path)
    torch.save(checkpoint_C, checkpoint_C_path)

    print(f"Saved Experiment B checkpoint to: {checkpoint_B_path}")
    print(f"Saved Experiment C checkpoint to: {checkpoint_C_path}")

    # Save results
    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/STAGE3_2_9L2_TRAINING_RESULTS.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results, checkpoint_B_path, checkpoint_C_path

if __name__ == "__main__":
    results, checkpoint_B_path, checkpoint_C_path = run_10_step_pilot()
