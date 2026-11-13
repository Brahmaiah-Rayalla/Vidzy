"""
Stage 3.2.9L.3 - Controlled 100-Step Training for Experiments B and C.

Configuration:
- Batch size: 1
- Resolution: 64x64
- Frames: 16
- CPU only, 8 threads
- Fixed random seed
- Exactly 100 optimizer steps per experiment
- Checkpoints at steps 25, 50, 75, 100
- Comprehensive monitoring and integrity assertions
"""

import torch
import torch.optim as optim
import json
from pathlib import Path
import sys
import time
import numpy as np
import gc
import psutil
import os

# Set PyTorch threads
torch.set_num_threads(8)

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
    """Synthetic video dataset for training."""

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

def count_parameters(model, trainable_only=True):
    """Count parameters by type."""
    param_info = {
        'num_tensors': 0,
        'num_scalars': 0,
        'num_trainable_tensors': 0,
        'num_trainable_scalars': 0
    }

    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue

        param_info['num_tensors'] += 1
        param_info['num_scalars'] += param.numel()

        if param.requires_grad:
            param_info['num_trainable_tensors'] += 1
            param_info['num_trainable_scalars'] += param.numel()

    return param_info

def verify_frozen_integrity(model, frozen_params_snapshot, current_step):
    """Verify that frozen parameters haven't changed."""
    violations = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            if name in frozen_params_snapshot:
                if not torch.allclose(param.data, frozen_params_snapshot[name], atol=1e-6):
                    violations.append(name)

    return violations

def record_parameter_state(model, trainable_only=False):
    """Record current parameter state for comparison."""
    state = {}
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        state[name] = param.data.clone()
    return state

def compute_parameter_delta(initial_state, current_state):
    """Compute parameter delta statistics."""
    deltas = {}

    for name in initial_state:
        if name in current_state:
            diff = (current_state[name] - initial_state[name]).abs()
            deltas[name] = {
                'max': diff.max().item(),
                'mean': diff.mean().item(),
                'std': diff.std().item()
            }

    return deltas

def verify_integrity_assertions(model, loss, optimizer_step_num):
    """Verify integrity assertions after optimizer update."""
    assertions = {
        'step': optimizer_step_num,
        'loss_finite': torch.isfinite(loss).item(),
        'gradients_finite': True,
        'unexpected_parameter_changes': []
    }

    # Check gradient finiteness
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                assertions['gradients_finite'] = False
                assertions['unexpected_parameter_changes'].append(f"{name}: gradient not finite")

    return assertions

def run_experiment_B(config, checkpoint, text_encoder_standard, tokenizer, dataloader, betas, alphas, alpha_bars, output_dir):
    """Run Experiment B: Motion embeddings only."""
    print("\n" + "=" * 80)
    print("EXPERIMENT B - Motion Embeddings Only (100 steps)")
    print("=" * 80)

    # Create models
    spatial_unet = ConditionalUNet(
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
    spatial_unet.load_state_dict(checkpoint['model_state_dict'])

    split_encoder = load_from_standard_encoder(
        text_encoder_standard,
        motion_vocab_size=40,
        freeze_transformer=True,
        original_vocab_size=42
    )

    video_unet = VideoUNetFixed(
        spatial_unet=spatial_unet,
        text_encoder=split_encoder,
        num_frames=config['num_frames'],
        image_size=config['image_size'],
        temporal_kernel=3,
        temporal_layers=2,
        freeze_spatial=True,
        use_temporal=True,
        enable_temporal_processing=False
    )

    # Record initial frozen state
    frozen_params_snapshot = record_parameter_state(video_unet, trainable_only=False)
    initial_motion_state = record_parameter_state(split_encoder.token_emb.motion_emb, trainable_only=True)

    # Setup optimizer
    motion_params = list(split_encoder.token_emb.motion_emb.parameters())
    optimizer = optim.AdamW(motion_params, lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Count parameters
    param_counts = count_parameters(video_unet, trainable_only=True)
    print(f"Trainable parameters: {param_counts}")

    # Training loop
    video_unet.train()
    dataloader_iter = iter(dataloader)

    results = {
        'experiment': 'B',
        'config': config,
        'parameter_counts': param_counts,
        'steps': [],
        'integrity_violations': []
    }

    checkpoint_steps = [25, 50, 75, 100]

    for step in range(config['num_steps']):
        step_start = time.time()
        peak_ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

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
        text_emb, text_mask = video_unet.text_encoder(token_ids, mask)
        predicted_noise = video_unet(xt, t, text_emb, text_mask)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        motion_grad_norm = sum(p.grad.norm().item() for p in motion_params if p.grad is not None)

        # Optimizer step
        optimizer.step()

        # Verify integrity
        frozen_violations = verify_frozen_integrity(video_unet, frozen_params_snapshot, step)
        integrity_assertions = verify_integrity_assertions(video_unet, loss, step)

        if frozen_violations:
            results['integrity_violations'].append({
                'step': step,
                'violations': frozen_violations
            })
            print(f"WARNING: Frozen parameters changed at step {step}: {frozen_violations}")

        if not integrity_assertions['loss_finite'] or not integrity_assertions['gradients_finite']:
            print(f"ERROR: Integrity violation at step {step}")
            raise RuntimeError(f"Integrity assertion failed: {integrity_assertions}")

        # Compute parameter delta
        current_motion_state = record_parameter_state(split_encoder.token_emb.motion_emb, trainable_only=True)
        motion_delta = compute_parameter_delta(initial_motion_state, current_motion_state)

        step_time = time.time() - step_start

        # Record step
        step_result = {
            'step': step,
            'loss': loss.item(),
            'motion_grad_norm': motion_grad_norm,
            'motion_delta': motion_delta.get('text_encoder.token_emb.motion_emb.weight', {}),
            'peak_ram_mb': peak_ram,
            'step_time': step_time,
            'learning_rate': config['learning_rate'],
            'integrity_assertions': integrity_assertions
        }

        results['steps'].append(step_result)

        print(f"Step {step+1}/{config['num_steps']}, Loss: {loss.item():.6f}, Motion grad: {motion_grad_norm:.6f}, Time: {step_time:.2f}s, RAM: {peak_ram:.1f}MB")

        # Save checkpoint
        if (step + 1) in checkpoint_steps:
            checkpoint_data = {
                'model_state_dict': video_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'experiment': 'B',
                'step': step + 1
            }
            checkpoint_path = output_dir / f"experiment_B/checkpoints/checkpoint_step_{(step+1):06d}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  Saved checkpoint to: {checkpoint_path}")

    return results

def run_experiment_C(config, checkpoint, text_encoder_standard, tokenizer, dataloader, betas, alphas, alpha_bars, output_dir):
    """Run Experiment C: Motion + Temporal + Fusion."""
    print("\n" + "=" * 80)
    print("EXPERIMENT C - Motion + Temporal + Fusion (100 steps)")
    print("=" * 80)

    # Create models
    spatial_unet = ConditionalUNet(
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
    spatial_unet.load_state_dict(checkpoint['model_state_dict'])

    split_encoder = load_from_standard_encoder(
        text_encoder_standard,
        motion_vocab_size=40,
        freeze_transformer=True,
        original_vocab_size=42
    )

    video_unet = VideoUNetFixed(
        spatial_unet=spatial_unet,
        text_encoder=split_encoder,
        num_frames=config['num_frames'],
        image_size=config['image_size'],
        temporal_kernel=3,
        temporal_layers=2,
        freeze_spatial=True,
        use_temporal=True,
        enable_temporal_processing=True
    )

    # Record initial frozen state
    frozen_params_snapshot = record_parameter_state(video_unet, trainable_only=False)
    initial_motion_state = record_parameter_state(split_encoder.token_emb.motion_emb, trainable_only=True)
    initial_temporal_state = record_parameter_state(video_unet.temporal_modules, trainable_only=True)
    initial_fusion_state = record_parameter_state(video_unet.fusion_modules, trainable_only=True)

    # Setup optimizer
    motion_params = list(split_encoder.token_emb.motion_emb.parameters())
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.fusion_modules.parameters())
    trainable_params = motion_params + temporal_params + fusion_params

    optimizer = optim.AdamW(trainable_params, lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Count parameters
    param_counts = count_parameters(video_unet, trainable_only=True)
    print(f"Trainable parameters: {param_counts}")

    # Training loop
    video_unet.train()
    dataloader_iter = iter(dataloader)

    results = {
        'experiment': 'C',
        'config': config,
        'parameter_counts': param_counts,
        'steps': [],
        'integrity_violations': []
    }

    checkpoint_steps = [25, 50, 75, 100]

    for step in range(config['num_steps']):
        step_start = time.time()
        peak_ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

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
        text_emb, text_mask = video_unet.text_encoder(token_ids, mask)
        predicted_noise = video_unet(xt, t, text_emb, text_mask)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        motion_grad_norm = sum(p.grad.norm().item() for p in motion_params if p.grad is not None)
        temporal_grad_norm = sum(p.grad.norm().item() for p in temporal_params if p.grad is not None)
        fusion_grad_norm = sum(p.grad.norm().item() for p in fusion_params if p.grad is not None)

        # Optimizer step
        optimizer.step()

        # Verify integrity
        frozen_violations = verify_frozen_integrity(video_unet, frozen_params_snapshot, step)
        integrity_assertions = verify_integrity_assertions(video_unet, loss, step)

        if frozen_violations:
            results['integrity_violations'].append({
                'step': step,
                'violations': frozen_violations
            })
            print(f"WARNING: Frozen parameters changed at step {step}: {frozen_violations}")

        if not integrity_assertions['loss_finite'] or not integrity_assertions['gradients_finite']:
            print(f"ERROR: Integrity violation at step {step}")
            raise RuntimeError(f"Integrity assertion failed: {integrity_assertions}")

        # Monitor gate values
        gate_values = video_unet.get_gate_values()

        # Compute parameter deltas
        current_motion_state = record_parameter_state(split_encoder.token_emb.motion_emb, trainable_only=True)
        current_temporal_state = record_parameter_state(video_unet.temporal_modules, trainable_only=True)
        current_fusion_state = record_parameter_state(video_unet.fusion_modules, trainable_only=True)

        motion_delta = compute_parameter_delta(initial_motion_state, current_motion_state)
        temporal_delta = compute_parameter_delta(initial_temporal_state, current_temporal_state)
        fusion_delta = compute_parameter_delta(initial_fusion_state, current_fusion_state)

        step_time = time.time() - step_start

        # Record step
        step_result = {
            'step': step,
            'loss': loss.item(),
            'motion_grad_norm': motion_grad_norm,
            'temporal_grad_norm': temporal_grad_norm,
            'fusion_grad_norm': fusion_grad_norm,
            'gate_values': gate_values,
            'motion_delta': motion_delta,
            'temporal_delta': temporal_delta,
            'fusion_delta': fusion_delta,
            'peak_ram_mb': peak_ram,
            'step_time': step_time,
            'learning_rate': config['learning_rate'],
            'integrity_assertions': integrity_assertions
        }

        results['steps'].append(step_result)

        print(f"Step {step+1}/{config['num_steps']}, Loss: {loss.item():.6f}, Motion grad: {motion_grad_norm:.6f}, Temporal grad: {temporal_grad_norm:.6f}, Fusion grad: {fusion_grad_norm:.6f}, Time: {step_time:.2f}s, RAM: {peak_ram:.1f}MB")
        print(f"  Gate values: {gate_values}")

        # Save checkpoint
        if (step + 1) in checkpoint_steps:
            checkpoint_data = {
                'model_state_dict': video_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'experiment': 'C',
                'step': step + 1
            }
            checkpoint_path = output_dir / f"experiment_C/checkpoints/checkpoint_step_{(step+1):06d}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  Saved checkpoint to: {checkpoint_path}")

    return results

def run_100_step_training():
    """Run controlled 100-step training for both experiments."""
    print("=" * 80)
    print("STAGE 3.2.9L.3 - CONTROLLED 100-STEP TRAINING")
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
        'num_steps': 100,
        'cpu_threads': 8
    }

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Output directory
    output_dir = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training"

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

    # Report dataset statistics
    print(f"\nDataset size: {len(dataset)}")

    # ============================================================================
    # RUN EXPERIMENTS
    # ============================================================================
    results_B = run_experiment_B(config, checkpoint, text_encoder_standard, tokenizer, dataloader, betas, alphas, alpha_bars, output_dir)
    gc.collect()

    results_C = run_experiment_C(config, checkpoint, text_encoder_standard, tokenizer, dataloader, betas, alphas, alpha_bars, output_dir)
    gc.collect()

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    combined_results = {
        'config': config,
        'experiment_B': results_B,
        'experiment_C': results_C
    }

    output_file = output_dir / "STAGE3_2_9L3_TRAINING_RESULTS.json"
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return combined_results

if __name__ == "__main__":
    results = run_100_step_training()
