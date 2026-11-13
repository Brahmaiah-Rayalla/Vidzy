"""
Stage 3.2.5: Extended Temporal Training with Comprehensive Validation

Features:
- Explicit spatial parameter preservation verification
- Pre-training, during-training, and post-training spatial validation
- Baseline evaluation at checkpoints
- Comprehensive metrics logging
- CPU-friendly implementation
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.diffusion.process import DiffusionProcess


class SyntheticVideoDataset(Dataset):
    """Dataset for synthetic video data."""

    def __init__(self, data_dir, split='train', num_frames=16, image_size=64):
        self.data_dir = os.path.join(data_dir, split)
        self.num_frames = num_frames
        self.image_size = image_size

        metadata_path = os.path.join(self.data_dir, "all_metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        captions_path = os.path.join(self.data_dir, "captions.txt")
        with open(captions_path, 'r') as f:
            self.captions = [line.strip() for line in f.readlines()]

        self.video_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('video_')
        ])

        print(f"Loaded {len(self.video_dirs)} videos from {split} split")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = os.path.join(self.data_dir, self.video_dirs[idx])

        frame_paths = sorted([
            f for f in os.listdir(video_dir)
            if f.endswith('.png')
        ])

        frames = []
        for frame_path in frame_paths[:self.num_frames]:
            frame = Image.open(os.path.join(video_dir, frame_path))
            frame = frame.resize((self.image_size, self.image_size))
            frame = np.array(frame).astype(np.float32) / 127.5 - 1.0
            frames.append(frame)

        video = np.stack(frames, axis=0)
        video = video.transpose(3, 0, 1, 2)

        caption = self.captions[idx]

        return {
            'video': torch.from_numpy(video).float(),
            'caption': caption,
            'metadata': self.metadata[idx]
        }


def compute_tensor_hash(tensor):
    """Compute a hash of a tensor for integrity checking."""
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def save_spatial_fingerprint(spatial_unet, output_path):
    """Save fingerprints of all spatial parameters."""
    fingerprint = {}
    for name, param in spatial_unet.named_parameters():
        fingerprint[name] = {
            'shape': list(param.shape),
            'mean': float(param.mean().item()),
            'std': float(param.std().item()),
            'min': float(param.min().item()),
            'max': float(param.max().item()),
            'hash': compute_tensor_hash(param),
            'requires_grad': param.requires_grad
        }
    
    with open(output_path, 'w') as f:
        json.dump(fingerprint, f, indent=2)
    
    return fingerprint


def compare_spatial_fingerprints(fingerprint1, fingerprint2):
    """Compare two spatial fingerprints and report differences."""
    differences = []
    max_diff = 0.0
    
    for name in fingerprint1.keys():
        if name not in fingerprint2:
            differences.append(f"Missing in checkpoint: {name}")
            continue
        
        f1 = fingerprint1[name]
        f2 = fingerprint2[name]
        
        if f1['hash'] != f2['hash']:
            diff = abs(f1['mean'] - f2['mean']) + abs(f1['std'] - f2['std'])
            differences.append(f"Changed: {name} (mean diff: {abs(f1['mean'] - f2['mean']):.8f})")
            if diff > max_diff:
                max_diff = diff
    
    return differences, max_diff


def evaluate_baselines(
    video_unet,
    diffusion,
    device,
    num_samples=5,
    num_frames=16,
    image_size=64
):
    """Evaluate all baselines on the same data."""
    video_unet.eval()
    
    with torch.no_grad():
        # Generate consistent test data
        torch.manual_seed(42)
        batch_size = num_samples
        
        clean_video = torch.randn(batch_size, 3, num_frames, image_size, image_size)
        clean_video = (clean_video - clean_video.min()) / (clean_video.max() - clean_video.min()) * 2 - 1
        clean_video = clean_video.to(device)
        
        text_emb = torch.randn(batch_size, 32, 256).to(device)
        text_mask = torch.ones(batch_size, 32).to(device)
        
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        
        # Forward diffusion
        videos_flat = clean_video.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        t_flat = t.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)
        noisy_videos, noise = diffusion.q_sample(videos_flat, t_flat)
        noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)
        
        results = {}
        
        # A. Zero-prediction baseline
        zero_prediction = torch.zeros_like(noisy_videos)
        zero_prediction_flat = zero_prediction.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        zero_mse = nn.functional.mse_loss(zero_prediction_flat, noise)
        results['zero_mse'] = float(zero_mse.item())
        
        # B. Frame-independent mode
        video_unet.use_temporal = False
        predicted_noise_frame_indep = video_unet(noisy_videos, t, text_emb, text_mask)
        predicted_noise_frame_indep_flat = predicted_noise_frame_indep.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        frame_indep_mse = nn.functional.mse_loss(predicted_noise_frame_indep_flat, noise)
        
        correlation_frame_indep = torch.corrcoef(torch.stack([
            noise.flatten(),
            predicted_noise_frame_indep_flat.flatten()
        ]))[0, 1].item()
        
        results['frame_independent_mse'] = float(frame_indep_mse.item())
        results['frame_independent_correlation'] = float(correlation_frame_indep)
        
        # C. Temporal mode
        video_unet.use_temporal = True
        predicted_noise_temporal = video_unet(noisy_videos, t, text_emb, text_mask)
        predicted_noise_temporal_flat = predicted_noise_temporal.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
        temporal_mse = nn.functional.mse_loss(predicted_noise_temporal_flat, noise)
        
        correlation_temporal = torch.corrcoef(torch.stack([
            noise.flatten(),
            predicted_noise_temporal_flat.flatten()
        ]))[0, 1].item()
        
        results['temporal_mse'] = float(temporal_mse.item())
        results['temporal_correlation'] = float(correlation_temporal)
        
        # Gate values
        bottleneck_gate = video_unet.bottleneck_fusion.gate_activation(video_unet.bottleneck_fusion.gate).item()
        low_res_gate = video_unet.low_res_fusion.gate_activation(video_unet.low_res_fusion.gate).item()
        
        results['bottleneck_gate'] = float(bottleneck_gate)
        results['low_res_gate'] = float(low_res_gate)
        
        # Temporal improvement over frame-independent
        improvement = (frame_indep_mse.item() - temporal_mse.item()) / frame_indep_mse.item() * 100
        results['temporal_improvement_pct'] = float(improvement)
    
    return results


def train_video_stage3_2_5(
    data_dir,
    output_dir,
    num_steps=500,
    batch_size=1,
    num_frames=16,
    image_size=64,
    learning_rate=1e-4,
    device='cpu',
    spatial_checkpoint_path=None,
    checkpoint_interval=50,
    seed=42,
    resume_from=None
):
    """Train video diffusion model with comprehensive validation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE 3.2.5: EXTENDED TEMPORAL TRAINING")
    print("=" * 80)

    # Load datasets
    print("\n[1] Loading datasets...")
    train_dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=num_frames, image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Try to load validation dataset
    val_dir = os.path.join(data_dir, 'val')
    if os.path.exists(val_dir) and os.path.exists(os.path.join(val_dir, 'all_metadata.json')):
        val_dataset = SyntheticVideoDataset(data_dir, split='val', num_frames=num_frames, image_size=image_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"Val samples: {len(val_dataset)}")
    else:
        val_dataset = None
        val_loader = None
        print("Val split not found, skipping validation")

    print(f"Train samples: {len(train_dataset)}")

    # Load tokenizer
    print("\n[2] Loading tokenizer...")
    tokenizer_path = os.path.join(data_dir, 'train', '..', '..', 'text_image', 'tokenizer', 'tokenizer.json')

    if os.path.exists(tokenizer_path):
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
    else:
        print("Tokenizer not found, creating simple word-based tokenizer")
        all_words = set()
        for caption in train_dataset.captions:
            all_words.update(caption.lower().split())

        word_to_idx = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        for i, word in enumerate(sorted(all_words), start=3):
            word_to_idx[word] = i

        idx_to_word = {v: k for k, v in word_to_idx.items()}
        vocab_size = len(word_to_idx)

        print(f"Created tokenizer with vocab size: {vocab_size}")

    # Create models
    print("\n[3] Creating models...")
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

    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    )

    # Load spatial checkpoint
    if spatial_checkpoint_path and os.path.exists(spatial_checkpoint_path):
        print(f"\n[4] Loading spatial checkpoint from {spatial_checkpoint_path}")
        checkpoint = torch.load(spatial_checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        spatial_unet.load_state_dict(state_dict, strict=False)

        if 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    else:
        print("No spatial checkpoint provided, training from scratch")

    # Create video U-Net with frozen spatial backbone
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=num_frames,
        image_size=image_size,
        freeze_spatial=True,
        use_temporal=True
    )
    
    # PRE-TRAINING SPATIAL VALIDATION
    print("\n[5] PRE-TRAINING SPATIAL VALIDATION")
    print("-" * 80)
    
    # Count spatial parameters (after VideoUNet creation)
    spatial_params = list(video_unet.spatial_unet.parameters())
    spatial_param_count = len(spatial_params)
    spatial_scalar_count = sum(p.numel() for p in spatial_params)
    
    print(f"Spatial parameter tensors: {spatial_param_count}")
    print(f"Spatial scalar parameters: {spatial_scalar_count:,}")
    
    # Save initial spatial fingerprint
    initial_fingerprint_path = os.path.join(output_dir, 'spatial_fingerprint_initial.json')
    initial_spatial_fingerprint = save_spatial_fingerprint(video_unet.spatial_unet, initial_fingerprint_path)
    print(f"Initial spatial fingerprint saved to {initial_fingerprint_path}")
    
    # Verify requires_grad
    frozen_count = sum(1 for p in spatial_params if not p.requires_grad)
    trainable_count = sum(1 for p in spatial_params if p.requires_grad)
    
    print(f"\nSpatial parameters with requires_grad=False: {frozen_count}/{spatial_param_count}")
    print(f"Spatial parameters with requires_grad=True: {trainable_count}/{spatial_param_count}")
    
    if trainable_count > 0:
        print(f"[ERROR] {trainable_count} spatial parameters are trainable!")
        trainable_names = [name for name, p in video_unet.spatial_unet.named_parameters() if p.requires_grad]
        print(f"Trainable parameters: {trainable_names[:5]}")
        raise RuntimeError("Spatial parameters must be frozen")
    
    print(f"\n[OK] All {spatial_param_count} spatial parameters confirmed frozen")

    video_unet.tokenizer = type('SimpleTokenizer', (), {
        'encode': lambda self, text, max_length: (
            [word_to_idx.get(w, 1) for w in text.lower().split()[:max_length]] + [2],
            [1] * (min(len(text.split()), max_length) + 1)
        ),
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': vocab_size
    })()

    video_unet = video_unet.to(device)

    schedule = GaussianDiffusionSchedule(num_timesteps=1000)
    diffusion = DiffusionProcess(schedule)

    # Create optimizer with explicit parameter validation
    print("\n[6] CREATING OPTIMIZER WITH PARAMETER VALIDATION")
    print("-" * 80)
    
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.bottleneck_fusion.parameters()) + list(video_unet.low_res_fusion.parameters())
    
    # Verify no overlap with spatial parameters
    spatial_param_ids = set(id(p) for p in spatial_params)
    temporal_param_ids = set(id(p) for p in temporal_params)
    fusion_param_ids = set(id(p) for p in fusion_params)
    
    overlap_temporal = spatial_param_ids.intersection(temporal_param_ids)
    overlap_fusion = spatial_param_ids.intersection(fusion_param_ids)
    
    if overlap_temporal:
        raise RuntimeError(f"Temporal parameters include {len(overlap_temporal)} spatial parameters!")
    if overlap_fusion:
        raise RuntimeError(f"Fusion parameters include {len(overlap_fusion)} spatial parameters!")
    
    temporal_scalar_count = sum(p.numel() for p in temporal_params)
    fusion_scalar_count = sum(p.numel() for p in fusion_params)
    
    print(f"Temporal parameter tensors: {len(temporal_params)}")
    print(f"Temporal scalar parameters: {temporal_scalar_count:,}")
    print(f"Fusion parameter tensors: {len(fusion_params)}")
    print(f"Fusion scalar parameters: {fusion_scalar_count:,}")
    print(f"Total trainable scalar parameters: {temporal_scalar_count + fusion_scalar_count:,}")
    
    optimizer = optim.Adam([
        {'params': temporal_params, 'lr': learning_rate, 'weight_decay': 1e-5},
        {'params': fusion_params, 'lr': learning_rate, 'weight_decay': 1e-6}
    ])
    
    print(f"\n[OK] Optimizer created with {len(temporal_params) + len(fusion_params)} parameter groups")
    print("No spatial parameters in optimizer")

    # Initial baseline evaluation
    print("\n[7] INITIAL BASELINE EVALUATION")
    print("-" * 80)
    
    initial_baselines = evaluate_baselines(video_unet, diffusion, device, num_samples=5, num_frames=num_frames, image_size=image_size)
    
    print(f"Zero-prediction MSE: {initial_baselines['zero_mse']:.6f}")
    print(f"Frame-independent MSE: {initial_baselines['frame_independent_mse']:.6f}")
    print(f"Frame-independent correlation: {initial_baselines['frame_independent_correlation']:.6f}")
    print(f"Temporal MSE: {initial_baselines['temporal_mse']:.6f}")
    print(f"Temporal correlation: {initial_baselines['temporal_correlation']:.6f}")
    print(f"Bottleneck gate: {initial_baselines['bottleneck_gate']:.6f}")
    print(f"Low-res gate: {initial_baselines['low_res_gate']:.6f}")
    
    # Save initial baselines
    with open(os.path.join(output_dir, 'initial_baselines.json'), 'w') as f:
        json.dump(initial_baselines, f, indent=2)

    # Training loop
    print("\n[8] TRAINING LOOP")
    print("=" * 80)
    
    global_step = 0
    training_metrics = []
    validation_metrics = []
    
    while global_step < num_steps:
        video_unet.train()
        epoch_loss = 0.0
        steps_this_epoch = 0
        
        pbar = tqdm(train_loader, desc=f"Step {global_step}/{num_steps}")
        for batch in pbar:
            if global_step >= num_steps:
                break
            
            videos = batch['video'].to(device)
            captions = batch['caption']

            batch_size = videos.shape[0]
            text_embs = []
            text_masks = []

            for caption in captions:
                token_ids, mask = video_unet.tokenizer.encode(caption, max_length=32)
                text_embs.append(token_ids)
                text_masks.append(mask)

            max_len = max(len(emb) for emb in text_embs)
            padded_embs = []
            padded_masks = []
            for emb, mask in zip(text_embs, text_masks):
                padded = emb + [0] * (max_len - len(emb))
                padded_mask = mask + [0] * (max_len - len(mask))
                padded_embs.append(padded)
                padded_masks.append(padded_mask)

            text_emb = torch.tensor(padded_embs).to(device)
            text_mask = torch.tensor(padded_masks).to(device)

            text_emb_encoded, _ = video_unet.text_encoder(text_emb, text_mask)
            text_emb = text_emb_encoded

            timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

            videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)
            timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)

            noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)
            noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)

            predicted_noise = video_unet(noisy_videos, timesteps, text_emb, text_mask)
            predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)

            loss = nn.functional.mse_loss(predicted_noise_flat, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect gradient norms
            temporal_grad_norms = []
            for name, param in video_unet.temporal_modules.named_parameters():
                if param.grad is not None:
                    temporal_grad_norms.append(param.grad.norm().item())
            
            fusion_grad_norms = []
            for name, param in video_unet.bottleneck_fusion.named_parameters():
                if param.grad is not None:
                    fusion_grad_norms.append(param.grad.norm().item())
            for name, param in video_unet.low_res_fusion.named_parameters():
                if param.grad is not None:
                    fusion_grad_norms.append(param.grad.norm().item())
            
            avg_temporal_grad = np.mean(temporal_grad_norms) if temporal_grad_norms else 0.0
            avg_fusion_grad = np.mean(fusion_grad_norms) if fusion_grad_norms else 0.0
            
            epoch_loss += loss.item()
            steps_this_epoch += 1
            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Runtime spatial validation every 50 steps
            if global_step % 50 == 0:
                current_fingerprint = save_spatial_fingerprint(video_unet.spatial_unet, 
                                                                os.path.join(output_dir, f'spatial_fingerprint_step_{global_step:06d}.json'))
                differences, max_diff = compare_spatial_fingerprints(initial_spatial_fingerprint, current_fingerprint)
                
                if len(differences) > 0:
                    print(f"\n[WARNING] Step {global_step}: {len(differences)} spatial parameters changed!")
                    print(f"  Max difference: {max_diff:.8f}")
                    print(f"  Sample differences: {differences[:3]}")
                else:
                    print(f"\n[OK] Step {global_step}: All spatial parameters unchanged")

            # Save checkpoint
            if global_step % checkpoint_interval == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{global_step:06d}.pt')
                torch.save({
                    'step': global_step,
                    'model_state_dict': video_unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"\n[CHECKPOINT] Saved to {checkpoint_path}")
                
                # Evaluate baselines at checkpoint
                checkpoint_baselines = evaluate_baselines(video_unet, diffusion, device, num_samples=5, num_frames=num_frames, image_size=image_size)
                
                with open(os.path.join(output_dir, f'baselines_step_{global_step:06d}.json'), 'w') as f:
                    json.dump(checkpoint_baselines, f, indent=2)
                
                print(f"  Frame-independent MSE: {checkpoint_baselines['frame_independent_mse']:.6f}")
                print(f"  Temporal MSE: {checkpoint_baselines['temporal_mse']:.6f}")
                print(f"  Temporal improvement: {checkpoint_baselines['temporal_improvement_pct']:.2f}%")
                print(f"  Bottleneck gate: {checkpoint_baselines['bottleneck_gate']:.6f}")
                print(f"  Low-res gate: {checkpoint_baselines['low_res_gate']:.6f}")
                
                # Record metrics
                training_metrics.append({
                    'step': global_step,
                    'loss': loss.item(),
                    'avg_temporal_grad': avg_temporal_grad,
                    'avg_fusion_grad': avg_fusion_grad,
                    **checkpoint_baselines
                })

        avg_epoch_loss = epoch_loss / steps_this_epoch if steps_this_epoch > 0 else 0.0
        print(f"\nAverage loss for steps {global_step - steps_this_epoch + 1}-{global_step}: {avg_epoch_loss:.4f}")

    # Final checkpoint
    final_checkpoint_path = os.path.join(output_dir, 'checkpoint_final.pt')
    torch.save({
        'step': global_step,
        'model_state_dict': video_unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
    }, final_checkpoint_path)
    print(f"\n[FINAL] Saved to {final_checkpoint_path}")

    # POST-TRAINING SPATIAL VALIDATION
    print("\n[9] POST-TRAINING SPATIAL VALIDATION")
    print("=" * 80)
    
    final_fingerprint_path = os.path.join(output_dir, 'spatial_fingerprint_final.json')
    final_spatial_fingerprint = save_spatial_fingerprint(video_unet.spatial_unet, final_fingerprint_path)
    
    differences, max_diff = compare_spatial_fingerprints(initial_spatial_fingerprint, final_spatial_fingerprint)
    
    print(f"Spatial parameter comparison:")
    print(f"  Total parameter tensors: {spatial_param_count}")
    print(f"  Changed parameter tensors: {len(differences)}")
    print(f"  Unchanged parameter tensors: {spatial_param_count - len(differences)}")
    print(f"  Maximum difference: {max_diff:.8f}")
    
    if len(differences) > 0:
        print(f"\n[ERROR] {len(differences)} spatial parameters changed during training!")
        print(f"Sample differences: {differences[:5]}")
    else:
        print(f"\n[OK] All {spatial_param_count} spatial parameters unchanged")
        print("Stage 2.1 performance preserved")
    
    # Save spatial integrity report
    integrity_report = {
        'spatial_param_count': spatial_param_count,
        'spatial_scalar_count': spatial_scalar_count,
        'changed_param_tensors': len(differences),
        'unchanged_param_tensors': spatial_param_count - len(differences),
        'max_difference': float(max_diff),
        'spatial_preserved': len(differences) == 0,
        'differences': differences[:10]  # First 10 differences
    }
    
    with open(os.path.join(output_dir, 'spatial_integrity_report.json'), 'w') as f:
        json.dump(integrity_report, f, indent=2)
    
    # Save training metrics
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total steps: {global_step}")
    print(f"Spatial parameters preserved: {len(differences) == 0}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 3.2.5: Extended Temporal Training")
    parser.add_argument("--data_dir", type=str, default="../data/synthetic_video", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints_stage3_2_5", help="Output directory")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--spatial_checkpoint", type=str, default="../checkpoints_conditional/checkpoint_final.pt", help="Spatial checkpoint")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Checkpoint interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    train_video_stage3_2_5(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        device=args.device,
        spatial_checkpoint_path=args.spatial_checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()
