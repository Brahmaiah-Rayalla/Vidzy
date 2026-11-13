"""
Test the fixed SplitEmbedding with torch.where.
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

from models.text_encoder_split import SplitEmbedding
from models.text_encoder import TextEncoder

# Load checkpoint
stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
checkpoint = torch.load(stage2_checkpoint, map_location='cpu')
actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

# Create split embedding with correct vocab sizes
original_vocab_size = 42  # The original vocab size before motion expansion
split_emb = SplitEmbedding(
    original_vocab_size=original_vocab_size,
    motion_vocab_size=40,
    d_model=256,
    original_embeddings=checkpoint['text_encoder_state_dict']['token_emb.weight'][:original_vocab_size].data.clone()
)

# Ensure motion embedding is trainable
split_emb.motion_emb.weight.requires_grad = True
split_emb.original_emb.weight.requires_grad = False

print(f"Motion embedding requires_grad: {split_emb.motion_emb.weight.requires_grad}")
print(f"Original embedding requires_grad: {split_emb.original_emb.weight.requires_grad}")

# Test forward pass
token_ids = torch.tensor([[5, 10, 42, 43, 15, 44, 20, 45, 25, 50]])
print(f"Token IDs: {token_ids}")
print(f"Motion tokens present: {(token_ids >= original_vocab_size).any().item()}")

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
