"""
Phase 5 - Fixed validation comparison at steps 100, 150, 200, 250
"""

import torch
import json
import numpy as np
from pathlib import Path
import sys

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

def evaluate_checkpoint(checkpoint_path, seed=42):
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model
    stage2 = torch.load(base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt", map_location=device)
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
    video_unet.load_state_dict(checkpoint['model_state_dict'])
    video_unet.eval()

    # Load validation data
    train_metadata = base_dir / "experiments_stage3_2_7/data_synthetic/train_metadata.json"
    with open(train_metadata, 'r') as f:
        metadata = json.load(f)

    tokenizer = SimpleTokenizer.load(base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json")
    betas, alphas, alpha_bars = get_diffusion_schedule()

    torch.manual_seed(seed)
    np.random.seed(seed)

    losses = []
    for i in range(min(8, len(metadata))):
        item = metadata[i]
        video = np.load(item['video_path'])
        video = torch.from_numpy(video).float() / 255.0 * 2 - 1
        video = video.permute(3, 0, 1, 2).unsqueeze(0).to(device)

        t = torch.randint(0, 1000, (1,)).to(device)
        xt, noise = add_noise(video, t, alpha_bars)

        token_ids, mask = tokenizer.encode(item['caption'], max_length=64)
        token_ids = torch.tensor([token_ids]).to(device)
        mask = torch.tensor([mask], dtype=torch.float).to(device)

        with torch.no_grad():
            text_emb, text_mask = video_unet.text_encoder(token_ids, mask)
            predicted_noise = video_unet(xt, t, text_emb, text_mask)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            losses.append(loss.item())

    return {
        'mean': sum(losses) / len(losses),
        'std': (sum((x - sum(losses)/len(losses))**2 for x in losses) / len(losses))**0.5,
        'min': min(losses),
        'max': max(losses),
        'losses': losses
    }

# Evaluate all checkpoints
checkpoints = {
    'C1_100': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C1/checkpoint_step_000100.pt",
    'C1_150': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/C1/checkpoint_step_000150.pt",
    'C1_200': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/C1/checkpoint_step_000200.pt",
    'C1_250': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C1/checkpoint_step_000250.pt",
    'C25_100': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/gate25_training/C25/checkpoint_step_000100.pt",
    'C25_150': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C25/checkpoint_step_000150.pt",
    'C25_200': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C25/checkpoint_step_000200.pt",
    'C25_250': base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C25/checkpoint_step_000250.pt"
}

results = {}
for name, path in checkpoints.items():
    if path.exists():
        print(f"Evaluating {name}...")
        results[name] = evaluate_checkpoint(path)
    else:
        results[name] = {'error': 'checkpoint not found'}

output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/L10_VALIDATION_COMPARISON.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Validation comparison saved to: {output_file}")
