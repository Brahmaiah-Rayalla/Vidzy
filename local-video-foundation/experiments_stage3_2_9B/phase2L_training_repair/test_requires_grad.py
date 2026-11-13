"""
Test requires_grad status throughout the text encoder.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
stage3_2_7_dir = base_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation" / "models"))

from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder
from models.text_encoder import TextEncoder

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

# Create split encoder
split_encoder = load_from_standard_encoder(
    text_encoder_standard,
    motion_vocab_size=40,
    freeze_transformer=True
)

print("=" * 80)
print("REQUIRES_GRAD STATUS")
print("=" * 80)

# Check motion embedding
print("\nMotion embedding:")
for name, param in split_encoder.token_emb.motion_emb.named_parameters():
    print(f"  {name}: requires_grad = {param.requires_grad}")

# Check original embedding
print("\nOriginal embedding:")
for name, param in split_encoder.token_emb.original_emb.named_parameters():
    print(f"  {name}: requires_grad = {param.requires_grad}")

# Check transformer layers
print("\nTransformer layers:")
for i, layer in enumerate(split_encoder.layers):
    for name, param in layer.named_parameters():
        print(f"  Layer {i} {name}: requires_grad = {param.requires_grad}")

# Test forward pass with requires_grad enabled
print("\n" + "=" * 80)
print("FORWARD PASS TEST")
print("=" * 80)

token_ids = torch.tensor([[5, 10, 42, 43, 15, 44, 20, 45, 25, 50]])
mask = torch.ones(1, 10)

print(f"\nToken IDs: {token_ids}")
print(f"Motion tokens present: {(token_ids >= 42).any().item()}")

# Forward pass
text_emb, text_mask = split_encoder(token_ids, mask)

print(f"\nText embedding shape: {text_emb.shape}")
print(f"Text embedding requires_grad: {text_emb.requires_grad}")

# Try to compute loss and backward
loss = text_emb.sum()
print(f"Loss requires_grad: {loss.requires_grad}")

try:
    loss.backward()
    print("Backward succeeded")

    # Check gradients
    motion_grad = split_encoder.token_emb.motion_emb.weight.grad
    print(f"Motion embedding has gradient: {motion_grad is not None}")
    if motion_grad is not None:
        print(f"Motion embedding gradient norm: {motion_grad.norm().item()}")
except Exception as e:
    print(f"Backward failed: {e}")
