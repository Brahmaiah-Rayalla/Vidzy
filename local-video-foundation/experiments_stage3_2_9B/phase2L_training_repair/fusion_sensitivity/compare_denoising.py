"""
Phase F - Controlled denoising comparison.

Compare denoising performance across gate configurations using fixed validation inputs.
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

def load_model_from_checkpoint(checkpoint_path, enable_temporal):
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
        freeze_spatial=True, use_temporal=True, enable_temporal_processing=enable_temporal
    )

    exp_checkpoint = torch.load(checkpoint_path, map_location=device)
    video_unet.load_state_dict(exp_checkpoint['model_state_dict'])
    video_unet.eval()
    video_unet = video_unet.to(device)

    return video_unet, split_encoder

def set_gate_value(model, gate_name, raw_value):
    if gate_name == 'bottleneck':
        model.fusion_modules['bottleneck'].gate.data.fill_(raw_value)
    elif gate_name == 'low_res':
        model.fusion_modules['low_res'].gate.data.fill_(raw_value)

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

def main():
    print("=" * 80)
    print("PHASE F - CONTROLLED DENOISING COMPARISON")
    print("=" * 80)

    checkpoint_C = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/controlled_training/experiment_C/checkpoints/checkpoint_step_000100.pt"

    # Test configurations (both gates)
    test_configs = [
        {'label': 'A', 'activation': 0.00, 'logit': -10.0},
        {'label': 'B', 'activation': 0.01, 'logit': -4.595},
        {'label': 'F', 'activation': 0.50, 'logit': 0.0}
    ]

    tokenizer_path = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    device = torch.device('cpu')
    betas, alphas, alpha_bars = get_diffusion_schedule()

    # Fixed validation inputs
    num_samples = 8
    videos = [torch.randn(1, 3, 16, 64, 64).to(device) for _ in range(num_samples)]
    captions = ["A red ball moves right"] * num_samples
    timesteps = [100, 500, 900]  # Low, middle, high

    results = []

    for config in test_configs:
        model, _ = load_model_from_checkpoint(checkpoint_C, enable_temporal=True)
        set_gate_value(model, 'bottleneck', config['logit'])
        set_gate_value(model, 'low_res', config['logit'])

        config_results = {
            'config': config,
            'samples': []
        }

        for sample_idx in range(num_samples):
            video = videos[sample_idx]
            caption = captions[sample_idx]

            token_ids_list, mask_list = tokenizer.encode(caption, max_length=64)
            token_ids = torch.tensor([token_ids_list]).to(device)
            mask = torch.tensor([mask_list], dtype=torch.float).to(device)

            sample_metrics = {'sample_idx': sample_idx, 'timesteps': []}

            for t_val in timesteps:
                t = torch.tensor([t_val]).to(device)
                xt, noise = add_noise(video, t, alpha_bars)

                with torch.no_grad():
                    text_emb, text_mask = model.text_encoder(token_ids, mask)
                    predicted_noise = model(xt, t, text_emb, text_mask)

                mse = torch.nn.functional.mse_loss(predicted_noise, noise).item()
                mae = torch.nn.functional.l1_loss(predicted_noise, noise).item()
                rmse = np.sqrt(mse)
                correlation = np.corrcoef(predicted_noise.flatten().cpu().numpy(), noise.flatten().cpu().numpy())[0, 1]

                timestep_metrics = {
                    'timestep': t_val,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'correlation': correlation
                }
                sample_metrics['timesteps'].append(timestep_metrics)

            config_results['samples'].append(sample_metrics)

        results.append(config_results)
        print(f"Config {config['label']} ({config['activation']:.2f}): Complete")

    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/fusion_sensitivity/STAGE3_2_9L5_DENOISING_COMPARISON.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
