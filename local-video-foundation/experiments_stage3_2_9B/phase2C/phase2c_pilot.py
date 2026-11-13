"""
Stage 3.2.9B - Phase 2C: Full 100-Step Pilot

Modified from Phase 2B smoke test to run 100 steps with comprehensive monitoring.

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

# Apply CPU optimizations
def apply_cpu_optimizations():
    logical_cores = psutil.cpu_count(logical=True)
    torch.set_num_threads(logical_cores)
    torch.set_num_interop_threads(logical_cores)
    os.environ['MKL_NUM_THREADS'] = str(logical_cores)
    os.environ['OMP_NUM_THREADS'] = str(logical_cores)
    return logical_cores

thread_count = apply_cpu_optimizations()

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
    def __init__(self, metadata_path: str):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"Loaded {len(self.metadata)} samples")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        video = np.load(item['video_path'])
        video = torch.from_numpy(video).float() / 255.0 * 2 - 1
        video = video.permute(3, 0, 1, 2)
        return {'video': video, 'caption': item['caption']}


def get_checkpoint_hash(checkpoint_path: Path) -> str:
    with open(checkpoint_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def record_baseline_hashes():
    print("\n" + "=" * 80)
    print("  RECORDING BASELINE CHECKPOINT HASHES")
    print("=" * 80)

    hashes = {}
    stage2_checkpoint = parent_dir / "checkpoints_conditional" / "checkpoint_final.pt"
    stage3_spatial_checkpoint = stage3_2_7_dir / "checkpoints" / "checkpoint_stage2_spatial.pt"
    tokenizer_path = stage3_2_7_dir / "checkpoints" / "tokenizer_motion_expanded.json"

    for path, name in [
        (stage2_checkpoint, 'stage2_checkpoint'),
        (stage3_spatial_checkpoint, 'stage3_spatial_checkpoint'),
        (tokenizer_path, 'tokenizer')
    ]:
        if path.exists():
            hash_value = get_checkpoint_hash(path)
            hashes[name] = hash_value
            print(f"{name}: {hash_value}")

    return hashes


def load_models():
    device = torch.device('cpu')
    stage2_checkpoint = parent_dir / "checkpoints_conditional" / "checkpoint_final.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location=device)
    actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

    text_encoder = TextEncoder(
        vocab_size=actual_vocab_size, d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0
    )
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
    )
    spatial_unet.load_state_dict(checkpoint['model_state_dict'])

    split_encoder = load_from_standard_encoder(
        text_encoder, motion_vocab_size=40, freeze_transformer=True
    )
    split_encoder.eval()
    split_encoder = split_encoder.to(device)

    return spatial_unet, split_encoder, device


def get_diffusion_schedule():
    num_timesteps = 1000
    betas = torch.linspace(0.0001, 0.02, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def add_noise(x0, t):
    betas, alphas, alpha_bars = get_diffusion_schedule()
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t])
    sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1)
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt, noise


def run_experiment(experiment_id: str, enable_temporal: bool, num_steps: int = 100, seed: int = 42):
    print("\n" + "=" * 80)
    print(f"  EXPERIMENT {experiment_id}: {'TEMPORAL ENABLED' if enable_temporal else 'TEMPORAL DISABLED'} - {num_steps} STEPS")
    print("=" * 80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    spatial_unet, split_encoder, device = load_models()

    # Record initial snapshots
    initial_spatial = {name: param.data.clone() for name, param in spatial_unet.named_parameters()}
    initial_motion = split_encoder.token_emb.motion_emb.weight.data.clone()

    video_unet = VideoUNetFixed(
        spatial_unet=spatial_unet,
        text_encoder=split_encoder,
        num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True, enable_temporal_processing=enable_temporal
    )
    video_unet = video_unet.to(device)

    if enable_temporal:
        trainable_params = list(split_encoder.token_emb.motion_emb.parameters())
        trainable_params.extend(list(video_unet.temporal_modules.parameters()))
        trainable_params.extend(list(video_unet.fusion_modules.parameters()))
    else:
        trainable_params = list(split_encoder.token_emb.motion_emb.parameters())

    optimizer = optim.AdamW(trainable_params, lr=0.0002, weight_decay=0.01)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    metadata_path = stage3_2_7_dir / "data_synthetic" / "train_metadata.json"
    dataset = SyntheticVideoDataset(metadata_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    tokenizer_path = stage3_2_7_dir / "checkpoints" / "tokenizer_motion_expanded.json"
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    results = {
        'losses': [],
        'timing': {
            'data_loading': [],
            'text_encoding': [],
            'forward_pass': [],
            'backward_pass': [],
            'optimizer_step': [],
            'total_step': []
        },
        'metrics': {
            'gradient_norms': [],
            'learning_rates': [],
            'fusion_gate_values': [],
            'cpu_utilization': [],
            'memory_consumption': []
        },
        'optimizer_step_count': 0,
        'config': {
            'enable_temporal': enable_temporal,
            'num_steps': num_steps,
            'seed': seed,
            'trainable_params': sum(p.numel() for p in trainable_params)
        }
    }

    print(f"\nStarting training for {num_steps} steps...")
    print(f"CPU frequency: {psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'} MHz")
    print(f"Thread count: {torch.get_num_threads()}")

    data_iter = iter(dataloader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        step_start = time.time()
        step_start_ram = psutil.virtual_memory().used / (1024**3)

        # Data loading
        data_start = time.time()
        videos = batch['video'].to(device)
        captions = batch['caption']
        data_end = time.time()

        # Text encoding
        text_start = time.time()
        token_ids_list = []
        mask_list = []
        for caption in captions:
            token_ids, mask = tokenizer.encode(caption, max_length=64)
            token_ids_list.append(token_ids)
            mask_list.append(mask)
        token_ids = torch.tensor(token_ids_list)
        mask = torch.tensor(mask_list, dtype=torch.float)
        text_end = time.time()

        # Forward pass
        forward_start = time.time()
        text_emb, text_mask = video_unet.text_encoder(token_ids, mask)
        t = torch.randint(0, 1000, (videos.shape[0],))
        xt, noise = add_noise(videos, t)
        output = video_unet(xt, t, text_emb, text_mask)
        forward_end = time.time()

        # Loss
        loss = nn.functional.mse_loss(output, noise)

        # Check for NaN/inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nERROR: Loss became {'NaN' if torch.isnan(loss) else 'infinite'} at step {step + 1}")
            results['error'] = f"Loss became {'NaN' if torch.isnan(loss) else 'infinite'} at step {step + 1}"
            return results

        # Backward pass
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_end = time.time()

        # Gradient norm
        total_norm = 0
        for p in trainable_params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        results['metrics']['gradient_norms'].append(total_norm)

        # Optimizer step
        opt_start = time.time()
        optimizer.step()
        opt_end = time.time()

        step_end = time.time()
        step_end_ram = psutil.virtual_memory().used / (1024**3)

        # Record metrics
        results['timing']['data_loading'].append(data_end - data_start)
        results['timing']['text_encoding'].append(text_end - text_start)
        results['timing']['forward_pass'].append(forward_end - forward_start)
        results['timing']['backward_pass'].append(backward_end - backward_start)
        results['timing']['optimizer_step'].append(opt_end - opt_start)
        results['timing']['total_step'].append(step_end - step_start)
        results['losses'].append(loss.item())
        results['optimizer_step_count'] += 1
        results['metrics']['learning_rates'].append(optimizer.param_groups[0]['lr'])
        results['metrics']['cpu_utilization'].append(psutil.cpu_percent())
        results['metrics']['memory_consumption'].append(step_end_ram - step_start_ram)

        if enable_temporal:
            gate_values = video_unet.get_gate_values()
            results['metrics']['fusion_gate_values'].append(gate_values)

        # Progress
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step + 1}/{num_steps}: Loss = {loss.item():.6f}, Time = {step_end - step_start:.2f}s, Grad Norm = {total_norm:.4f}")

    # Verify spatial parameters unchanged
    final_spatial = {name: param.data.clone() for name, param in spatial_unet.named_parameters()}
    spatial_changed = False
    for name in initial_spatial:
        if torch.max(torch.abs(initial_spatial[name] - final_spatial[name])) > 1e-10:
            spatial_changed = True
            print(f"\nWARNING: Spatial parameter {name} changed!")

    if spatial_changed:
        results['integrity_error'] = "Spatial parameters changed during training"
        print("ERROR: Frozen spatial parameters changed!")
        return results

    # Save checkpoint
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_phase2c_{experiment_id}.pt"

    torch.save({
        'video_unet': video_unet.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': num_steps,
        'loss': results['losses'][-1],
        'config': results['config']
    }, checkpoint_path)

    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Checkpoint hash: {get_checkpoint_hash(checkpoint_path)}")

    return results


def main():
    print("\n" + "=" * 80)
    print("  STAGE 3.2.9B - PHASE 2C: FULL 100-STEP PILOT")
    print("=" * 80)
    print(f"\nThread count: {thread_count}")
    print(f"CPU frequency: {psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'} MHz")

    # Record baseline hashes
    baseline_hashes = record_baseline_hashes()

    # Experiment A: Spatial-only baseline (no training)
    print("\n" + "=" * 80)
    print("  EXPERIMENT A: SPATIAL-ONLY BASELINE (NO TRAINING)")
    print("=" * 80)
    results_A = {
        'config': {'enable_temporal': False, 'num_steps': 0, 'seed': 42, 'trainable_params': 0},
        'losses': [], 'timing': {}, 'metrics': {}, 'optimizer_step_count': 0,
        'note': 'Spatial-only baseline - no training performed'
    }

    # Experiment B: Motion embeddings only (100 steps)
    results_B = run_experiment("B", enable_temporal=False, num_steps=100, seed=42)

    if 'error' in results_B or 'integrity_error' in results_B:
        print("\nERROR: Experiment B failed")
        return 1

    # Experiment C: Motion + temporal (100 steps)
    results_C = run_experiment("C", enable_temporal=True, num_steps=100, seed=42)

    if 'error' in results_C or 'integrity_error' in results_C:
        print("\nERROR: Experiment C failed")
        return 1

    # Save results
    results = {
        'baseline_hashes': baseline_hashes,
        'experiment_A': results_A,
        'experiment_B': results_B,
        'experiment_C': results_C,
        'optimization_config': {
            'thread_count': thread_count,
            'cpu_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
    }

    output_path = Path(__file__).parent / "phase2c_pilot_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
