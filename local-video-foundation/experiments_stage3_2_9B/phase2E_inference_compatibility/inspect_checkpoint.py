"""
Inspect checkpoint structure to understand actual architecture.
"""

import torch
from pathlib import Path

base_dir = Path(__file__).parent.parent.parent

# Load Experiment B checkpoint (Stage 3)
checkpoint_path = base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt"

print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(f"  {key}")

if 'model_state_dict' in checkpoint:
    print("\nModel state dict keys (first 20):")
    for i, key in enumerate(list(checkpoint['model_state_dict'].keys())[:20]):
        print(f"  {key}")
    print(f"  ... total {len(checkpoint['model_state_dict'])} keys")

if 'config' in checkpoint:
    print("\nConfig:")
    print(checkpoint['config'])

# Try to infer model type from keys
print("\nArchitecture inference:")
model_dict = checkpoint.get('video_unet', checkpoint.get('model_state_dict', checkpoint))

has_spatial_unet = any('spatial_unet' in k for k in model_dict)
has_text_encoder = any('text_encoder' in k for k in model_dict)
has_temporal = any('temporal' in k for k in model_dict)
has_gate = any('gate' in k for k in model_dict)

print(f"  Has spatial_unet: {has_spatial_unet}")
print(f"  Has text_encoder: {has_text_encoder}")
print(f"  Has temporal layers: {has_temporal}")
print(f"  Has gate parameters: {has_gate}")

print("\nModel state dict keys (first 30):")
for i, key in enumerate(list(model_dict.keys())[:30]):
    print(f"  {key}")
print(f"  ... total {len(model_dict)} keys")

if has_spatial_unet:
    print("\n  This appears to be a VideoUNetFixed model")
    # Inspect text encoder vocab sizes
    for key in model_dict.keys():
        if 'original_emb.weight' in key:
            print(f"  {key}: {model_dict[key].shape}")
        if 'motion_emb.weight' in key:
            print(f"  {key}: {model_dict[key].shape}")
else:
    print("\n  This appears to be a ConditionalUNet model")
