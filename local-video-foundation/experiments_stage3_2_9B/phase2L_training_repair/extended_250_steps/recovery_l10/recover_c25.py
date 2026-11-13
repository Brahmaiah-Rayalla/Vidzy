"""
Phase 3 - Recover C25 from step 100 to step 200
"""

import torch
import torch.optim as optim
import json
import numpy as np
from pathlib import Path
import sys
import time

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

def recover_c25():
    device = torch.device('cpu')
    
    # Load L7 C25 step-100 checkpoint
    checkpoint_path = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000100.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Loaded checkpoint from step {checkpoint['step']}")
    print(f"Gate init: {checkpoint['gate_init']}")
    
    # Reconstruct model architecture
    stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    stage2 = torch.load(stage2_checkpoint, map_location=device)
    actual_vocab_size = stage2['text_encoder_state_dict']['token_emb.weight'].shape[0]

    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
    )
    spatial_unet.load_state_dict(stage2['model_state_dict'])

    text_encoder_standard = TextEncoder(
        vocab_size=actual_vocab_size, d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0
    )
    text_encoder_standard.load_state_dict(stage2['text_encoder_state_dict'])

    split_encoder = load_from_standard_encoder(
        text_encoder_standard, motion_vocab_size=40,
        freeze_transformer=True, original_vocab_size=42
    )

    video_unet = VideoUNetFixed(
        spatial_unet=spatial_unet, text_encoder=split_encoder,
        num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
        freeze_spatial=True, use_temporal=True, enable_temporal_processing=True
    )

    # Load model state
    video_unet.load_state_dict(checkpoint['model_state_dict'])
    
    # Setup optimizer and restore state
    motion_params = list(split_encoder.token_emb.motion_emb.parameters())
    temporal_params = list(video_unet.temporal_modules.parameters())
    fusion_params = list(video_unet.fusion_modules.parameters())
    trainable_params = motion_params + temporal_params + fusion_params

    optimizer = optim.AdamW(trainable_params, lr=0.0002, weight_decay=0.01)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Restored optimizer state successfully")

    # Load dataset
    train_metadata = base_dir / "experiments_stage3_2_7/data_synthetic/train_metadata.json"
    dataset = SyntheticVideoDataset(str(train_metadata))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    tokenizer_path = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    betas, alphas, alpha_bars = get_diffusion_schedule()

    torch.manual_seed(42)
    np.random.seed(42)

    results = {
        'experiment': 'C25',
        'start_step': checkpoint['step'],
        'target_step': 200,
        'steps': []
    }

    video_unet.train()
    dataloader_iter = iter(dataloader)

    additional_steps = 200 - checkpoint['step']
    checkpoint_steps = [150, 200]

    for step in range(additional_steps):
        step_start = time.time()
        global_step = checkpoint['step'] + step + 1
        
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
            print(f"ERROR: Loss became NaN/inf at step {global_step}")
            results['error'] = f"Loss became {'NaN' if torch.isnan(loss) else 'infinite'} at step {global_step}"
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
            'step': global_step,
            'loss': loss.item(),
            'motion_grad_norm': motion_grad_norm,
            'temporal_grad_norm': temporal_grad_norm,
            'fusion_grad_norm': fusion_grad_norm,
            'gate_values': gate_values,
            'step_time': step_time
        })

        if global_step % 10 == 0 or global_step == checkpoint['step'] + 1:
            print(f"Step {global_step}/200: Loss={loss.item():.6f}, Gates={gate_values}")

        # Save checkpoint
        if global_step in checkpoint_steps:
            checkpoint_dir = base_dir / f"experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C25"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path_new = checkpoint_dir / f"checkpoint_step_{global_step:06d}.pt"
            
            torch.save({
                'model_state_dict': video_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': global_step,
                'experiment': 'C25',
                'recovery': True,
                'recovery_from': 'L7_step_100'
            }, checkpoint_path_new)
            print(f"  Saved recovery checkpoint to: {checkpoint_path_new}")

    return results

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 3 - RECOVER C25 FROM STEP 100 TO 200")
    print("=" * 80)
    
    result = recover_c25()
    
    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/L10_TRAINING_RECOVERY_PROGRESS.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nRecovery complete. Results saved to: {output_file}")
