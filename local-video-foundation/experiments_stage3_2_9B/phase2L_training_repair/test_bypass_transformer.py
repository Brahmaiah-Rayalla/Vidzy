"""
Test if we can bypass the transformer for motion embeddings.
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

from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder, SplitEmbedding
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

# Create split encoder with UNFROZEN transformer
print("=" * 80)
print("TEST 1: Unfreeze transformer layers")
print("=" * 80)

split_encoder_unfrozen = load_from_standard_encoder(
    text_encoder_standard,
    motion_vocab_size=40,
    freeze_transformer=False  # UNFROZEN
)

# Unfreeze all transformer layers
for layer in split_encoder_unfrozen.layers:
    for param in layer.parameters():
        param.requires_grad = True

# Keep original embeddings frozen
split_encoder_unfrozen.token_emb.original_emb.weight.requires_grad = False

# Check requires_grad
print("\nMotion embedding requires_grad:", split_encoder_unfrozen.token_emb.motion_emb.weight.requires_grad)
print("Original embedding requires_grad:", split_encoder_unfrozen.token_emb.original_emb.weight.requires_grad)
print("Transformer layer 0 requires_grad:", list(split_encoder_unfrozen.layers[0].parameters())[0].requires_grad)

# Test forward pass
token_ids = torch.tensor([[5, 10, 42, 43, 15, 44, 20, 45, 25, 50]])
mask = torch.ones(1, 10)

text_emb, text_mask = split_encoder_unfrozen(token_ids, mask)
print(f"\nText embedding requires_grad: {text_emb.requires_grad}")

loss = text_emb.sum()
print(f"Loss requires_grad: {loss.requires_grad}")

try:
    loss.backward()
    print("Backward succeeded")

    motion_grad = split_encoder_unfrozen.token_emb.motion_emb.weight.grad
    print(f"Motion embedding has gradient: {motion_grad is not None}")
    if motion_grad is not None:
        print(f"Motion embedding gradient norm: {motion_grad.norm().item()}")
except Exception as e:
    print(f"Backward failed: {e}")

print("\n" + "=" * 80)
print("TEST 2: Use embeddings directly (bypass transformer)")
print("=" * 80)

# Test using embeddings directly without transformer
split_emb = SplitEmbedding(
    original_vocab_size=actual_vocab_size,
    motion_vocab_size=40,
    d_model=256,
    original_embeddings=checkpoint['text_encoder_state_dict']['token_emb.weight'].data.clone()
)

# Ensure motion embedding is trainable
split_emb.motion_emb.weight.requires_grad = True
split_emb.original_emb.weight.requires_grad = False

print(f"Motion embedding requires_grad: {split_emb.motion_emb.weight.requires_grad}")

# Forward pass
embeddings = split_emb(token_ids)
print(f"Embeddings requires_grad: {embeddings.requires_grad}")

loss = embeddings.sum()
print(f"Loss requires_grad: {loss.requires_grad}")

try:
    loss.backward()
    print("Backward succeeded")

    motion_grad = split_emb.motion_emb.weight.grad
    print(f"Motion embedding has gradient: {motion_grad is not None}")
    if motion_grad is not None:
        print(f"Motion embedding gradient norm: {motion_grad.norm().item()}")
except Exception as e:
    print(f"Backward failed: {e}")
