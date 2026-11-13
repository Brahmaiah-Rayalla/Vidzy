"""
Stage 3.2.9L.7 - Controlled Training: C1 (1%) vs C25 (25%) Gate Initialization

This script runs controlled training comparing 1% vs 25% fusion gate initialization.
"""

import torch
import torch.optim as optim
import json
import numpy as np
from pathlib import Path
import sys
import time
import gc
import psutil
import os

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
        return {'video': video, 'caption': item['caption']}

def get_diffusion_schedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def add_noise(x0, t, alpha_bars):
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t])
    sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1)
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt, noise

def count_parameters(model, trainable_only=True):
    param_info = {'num_tensors': 0, 'num_scalars': 0, 'num_trainable_tensors': 0, 'num_trainable_scalars': 0}
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        param_info['num_tensors'] += 1
        param_info['num_scalars'] += param.numel()
        if param.requires_grad:
            param_info['num_trainable_tensors'] += 1
            param_info['num_trainable_scalars'] += param.numel()
    return param_info

def record_parameter_state(model, trainable_only=False):
    state = {}
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        state[name] = param.data.clone()
    return state

def set_gate_values(model, bottleneck_raw, low_res_raw):
    model.fusion_modules['bottleneck'].gate.data.fill_(bottleneck_raw)
    model.fusion_modules['low_res'].gate.data.fill_(low_res_raw)

def run_experiment(experiment_id, gate_init, num_steps, seed=42):
    device = torch.device('cpu')
    
    stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location=device)
    actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
    )
    spatial_unet.load_state_dict(checkpoint['model_state_dict'])

    text_encoder_standard = TextEncoder(
        vocab_size=actual_vocab_size, d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0
    )
    text_encoder_standard.load_state_dict(checkpoint['text_encoder_state_dict'])

    split_encoder = load_from_standard_encoder(
        text_encoder_standard, motion_vocab_size=40,
        freeze_transformer=True, original_vocab_size=42
    )

    video_unet = VideoUNetFixed(
        spatial_unet=spatial_unet, text_encoder=split_encoder,
        num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True, enable_temporal_processing=True
    )

    # Set gate values
    set_gate_values(video_unet, gate_init['bottleneck'], gate_init['low_res'])

    # Record initial state
    initial_motion_state = record_parameter_state(split_encoder.token_emb.motion_emb, trainable_only=True)
    initial_temporal_state = record_parameter_state(video_unet.temporal_modules, trainable_only=True)
    initial_fusion_state = record_parameter_state(video_unet.fusion_modules, trainable_only=True)

    # Setup optimizer
    motion_params = list(split_encoder.token_emb.motion_emb.parameters())
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.fusion_modules.parameters())
    trainable_params = motion_params + temporal_params + fusion_params

    optimizer = optim.AdamW(trainable_params, lr=0.0002, weight_decay=0.01)

    param_counts = count_parameters(video_unet, trainable_only=True)
    print(f"Experiment {experiment_id}: Trainable parameters: {param_counts['num_trainable_scalars']:,}")

    # Load dataset
    train_metadata = base_dir / "experiments_stage3_2_7/data_synthetic/train_metadata.json"
    dataset = SyntheticVideoDataset(str(train_metadata))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    tokenizer_path = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    betas, alphas, alpha_bars = get_diffusion_schedule()

    torch.manual_seed(seed)
    np.random.seed(seed)

    results = {
        'experiment': experiment_id,
        'gate_init': gate_init,
        'param_counts': param_counts,
        'steps': [],
        'config': {'num_steps': num_steps, 'seed': seed, 'lr': 0.0002, 'wd': 0.01}
    }

    video_unet.train()
    dataloader_iter = iter(dataloader)

    checkpoint_steps = [25, 50, 75, 100]

    for step in range(num_steps):
        step_start = time.time()
        
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        videos = batch['video'].to(device)
        captions = batch['caption']

        t = torch.randint(0, 1000, (videos.shape[0],)).to(device)
        xt, noise = add_noise(videos, t, alpha_bars)

        token_ids_list = []
        mask_list = []
        for caption in captions:
            token_ids, mask = tokenizer.encode(caption, max_length=64)
            token_ids_list.append(token_ids)
            mask_list.append(mask)

        token_ids = torch.tensor(token_ids_list).to(device)
        mask = torch.tensor(mask_list, dtype=torch.float).to(device)

        text_emb, text_mask = video_unet.text_encoder(token_ids, mask)
        predicted_noise = video_unet(xt, t, text_emb, text_mask)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"ERROR: Loss became NaN/inf at step {step + 1}")
            results['error'] = f"Loss became {'NaN' if torch.isnan(loss) else 'infinite'} at step {step + 1}"
            return results

        optimizer.zero_grad()
        loss.backward()

        motion_grad_norm = sum(p.grad.norm().item() for p in motion_params if p.grad is not None)
        temporal_grad_norm = sum(p.grad.norm().item() for p in temporal_params if p.grad is not None)
        fusion_grad_norm = sum(p.grad.norm().item() for p in fusion_params if p.grad is not None)

        optimizer.step()

        gate_values = video_unet.get_gate_values()

        step_time = time.time() - step_start

        results['steps'].append({
            'step': step,
            'loss': loss.item(),
            'motion_grad_norm': motion_grad_norm,
            'temporal_grad_norm': temporal_grad_norm,
            'fusion_grad_norm': fusion_grad_norm,
            'gate_values': gate_values,
            'step_time': step_time
        })

        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1}/{num_steps}: Loss={loss.item():.6f}, Gates={gate_values}")

        # Save checkpoint
        if (step + 1) in checkpoint_steps:
            checkpoint_dir = base_dir / f"experiments_stage3_2_9B/phase2L_training_repair/gate25_training/{experiment_id}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{(step+1):06d}.pt"
            
            torch.save({
                'model_state_dict': video_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': results['config'],
                'experiment': experiment_id,
                'gate_init': gate_init,
                'step': step + 1
            }, checkpoint_path)
            print(f"  Saved checkpoint to: {checkpoint_path}")

    return results

def main():
    print("=" * 80)
    print("STAGE 3.2.9L.7 - CONTROLLED TRAINING: C1 (1%) vs C25 (25%)")
    print("=" * 80)

    # Gate configurations
    configs = {
        'C1': {'bottleneck': -4.595, 'low_res': -4.595, 'label': '1%'},
        'C25': {'bottleneck': -1.099, 'low_res': -1.099, 'label': '25%'}
    }

    # Smoke test (10 steps)
    print("\n--- SMOKE TEST (10 steps) ---")
    smoke_results = {}
    for exp_id, gate_init in configs.items():
        print(f"\nRunning smoke test for {exp_id} ({gate_init['label']})...")
        result = run_experiment(exp_id, gate_init, num_steps=10, seed=42)
        if 'error' in result:
            print(f"SMOKE TEST FAILED for {exp_id}: {result['error']}")
            return 1
        smoke_results[exp_id] = result
        print(f"Smoke test PASSED for {exp_id}")

    # Save smoke test results
    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/STAGE3_2_9L7_SMOKE_TEST_RESULTS.json"
    with open(output_file, 'w') as f:
        json.dump(smoke_results, f, indent=2)
    print(f"\nSmoke test results saved to: {output_file}")

    # 100-step pilot
    print("\n--- 100-STEP PILOT ---")
    pilot_results = {}
    for exp_id, gate_init in configs.items():
        print(f"\nRunning 100-step pilot for {exp_id} ({gate_init['label']})...")
        result = run_experiment(exp_id, gate_init, num_steps=100, seed=42)
        if 'error' in result:
            print(f"PILOT FAILED for {exp_id}: {result['error']}")
            return 1
        pilot_results[exp_id] = result
        print(f"Pilot COMPLETE for {exp_id}")

    # Save pilot results
    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/STAGE3_2_9L7_TRAINING_RESULTS.json"
    with open(output_file, 'w') as f:
        json.dump({'C1': smoke_results['C1'], 'C25': smoke_results['C25'], 'pilot_C1': pilot_results['C1'], 'pilot_C25': pilot_results['C25']}, f, indent=2)
    print(f"\nTraining results saved to: {output_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
