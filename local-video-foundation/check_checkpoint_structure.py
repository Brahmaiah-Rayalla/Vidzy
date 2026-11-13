"""
Check the structure of Stage 2 checkpoint.
"""

import torch

checkpoint_path = "checkpoints_conditional/checkpoint_final.pt"

print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

for key in checkpoint.keys():
    value = checkpoint[key]
    if isinstance(value, dict):
        print(f"\n{key}:")
        print(f"  Type: dict")
        print(f"  Number of keys: {len(value)}")
        print(f"  Sample keys: {list(value.keys())[:10]}")
    elif isinstance(value, (int, float, str)):
        print(f"\n{key}: {value}")
    else:
        print(f"\n{key}: {type(value)}")
