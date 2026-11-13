"""
Stage 3.2.7B: Training Script for Experiment A
- Motion vocabulary only
- Temporal modules disabled
- Text encoder trainable
"""

import os
import sys
import yaml
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments_stage3_2_7.video_foundation.models.conditional_unet import ConditionalUNet
from experiments_stage3_2_7.video_foundation.models.text_encoder import TextEncoder
from experiments_stage3_2_7.video_foundation.models.video_unet_feature_fusion import VideoUNetFixed
from experiments_stage3_2_7.training.common_trainer import CommonTrainer
from experiments_stage3_2_7.training.parameter_groups import ParameterGroupManager


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_tokenizer(tokenizer_path: str) -> dict:
    with open(tokenizer_path, 'r') as f:
        return json.load(f)


def main():
    print("=" * 80)
    print("STAGE 3.2.7B: TRAINING EXPERIMENT A - Motion Vocabulary Only")
    print("=" * 80)

    # Load config
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "configs" / "experiment_a.yaml"
    config = load_config(str(config_path))

    # Load tokenizer
    tokenizer_path_str = config['checkpoints']['tokenizer_path']
    tokenizer_path = Path(tokenizer_path_str)
    if not tokenizer_path.is_absolute():
        tokenizer_path = (script_dir / tokenizer_path_str).resolve()
    tokenizer = load_tokenizer(str(tokenizer_path))

    print(f"\nTokenizer: {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer['vocab_size']}")
    print(f"Motion tokens: indices 42-81")

    # Initialize model
    print("\nInitializing model...")
    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=config['training']['num_timesteps'], text_dim=256,
        n_heads=4, use_cross_attn=True
    )

    text_encoder = TextEncoder(
        vocab_size=tokenizer['vocab_size'], d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=config['training']['max_seq_len']
    )

    model = VideoUNetFixed(
        spatial_unet=spatial_unet, text_encoder=text_encoder,
        num_frames=config['model']['num_frames'], image_size=config['model']['image_size'],
        temporal_kernel=config['model']['temporal_kernel'], temporal_layers=config['model']['temporal_layers'],
        freeze_spatial=config['model']['freeze_spatial'], use_temporal=config['model']['use_temporal']
    )

    # Load checkpoints
    print("\nLoading checkpoints...")
    checkpoint_path = script_dir / config['checkpoints']['spatial_checkpoint']
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load from model_state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

            # Load spatial UNet (filter by prefix)
            spatial_state = {k.replace('spatial_unet.', ''): v for k, v in state_dict.items() if k.startswith('spatial_unet.')}
            if spatial_state:
                model.spatial_unet.load_state_dict(spatial_state, strict=False)
                print(f"Loaded spatial UNet from checkpoint ({len(spatial_state)} keys)")

            # Load text encoder (filter by prefix)
            text_state = {k.replace('text_encoder.', ''): v for k, v in state_dict.items() if k.startswith('text_encoder.')}
            if text_state:
                # For expanded vocab, only load compatible layers
                model_state = model.text_encoder.state_dict()
                for key in model_state:
                    if key in text_state and model_state[key].shape == text_state[key].shape:
                        model_state[key] = text_state[key]
                model.text_encoder.load_state_dict(model_state)
                print(f"Loaded text encoder from checkpoint ({len(model_state)} compatible keys)")
        else:
            # Try loading the whole state dict
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded model from checkpoint (partial loading)")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    # Setup trainer
    device = config['device']
    model = model.to(device)

    # Merge config sections
    trainer_config = {
        **config['training'],
        'data_dir': config['data']['data_dir'],
        'checkpoint_dir': config['checkpoints']['checkpoint_dir']
    }

    trainer = CommonTrainer(
        model=model,
        tokenizer=tokenizer,
        experiment_name=config['experiment']['name'],
        config=trainer_config,
        device=device
    )

    # Setup parameters for Experiment A
    print("\nSetting up parameter groups for Experiment A...")
    trainer.setup_parameters('A')

    # Print parameter summary
    param_summary = trainer.param_manager.get_parameter_summary()
    print("\nParameter Summary:")
    for module, counts in param_summary.items():
        print(f"  {module}: total={counts['total']:,}, trainable={counts['trainable']:,}")

    total_params = trainer.param_manager.count_parameters(trainable_only=False)
    trainable_params = trainer.param_manager.count_parameters(trainable_only=True)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    trainer.train(
        num_steps=config['training']['num_steps'],
        save_every=config['training']['save_every'],
        val_every=config['training']['val_every']
    )

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")


if __name__ == "__main__":
    main()
