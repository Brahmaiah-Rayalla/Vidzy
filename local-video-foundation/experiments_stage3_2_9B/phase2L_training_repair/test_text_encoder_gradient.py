"""
Test if text encoder gradients flow when called as standalone vs. as submodule.
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
print("TEST 1: Text encoder as standalone module")
print("=" * 80)

# Test 1: Call text encoder as standalone
token_ids = torch.tensor([[5, 10, 42, 43, 15, 44, 20, 45, 25, 50]])
mask = torch.ones(1, 10)

text_emb_1, text_mask_1 = split_encoder(token_ids, mask)
loss_1 = text_emb_1.sum()
loss_1.backward()

motion_grad_1 = split_encoder.token_emb.motion_emb.weight.grad
print(f"Motion embedding has gradient: {motion_grad_1 is not None}")
if motion_grad_1 is not None:
    print(f"Motion embedding gradient norm: {motion_grad_1.norm().item()}")

# Reset gradients
split_encoder.zero_grad()

print("\n" + "=" * 80)
print("TEST 2: Text encoder as submodule of another module")
print("=" * 80)

# Create a wrapper module
class WrapperModule(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, x, t, text_emb, text_mask):
        # Simple dummy forward
        return text_emb.sum()

wrapper = WrapperModule(split_encoder)

# Call as submodule
text_emb_2, text_mask_2 = wrapper.text_encoder(token_ids, mask)
output_2 = wrapper(None, None, text_emb_2, text_mask_2)
loss_2 = output_2
loss_2.backward()

motion_grad_2 = split_encoder.token_emb.motion_emb.weight.grad
print(f"Motion embedding has gradient: {motion_grad_2 is not None}")
if motion_grad_2 is not None:
    print(f"Motion embedding gradient norm: {motion_grad_2.norm().item()}")

# Reset gradients
split_encoder.zero_grad()

print("\n" + "=" * 80)
print("TEST 3: Check if requires_grad is set correctly")
print("=" * 80)

motion_params = list(split_encoder.token_emb.motion_emb.parameters())
print(f"Motion embedding parameters: {len(motion_params)}")
for i, p in enumerate(motion_params):
    print(f"  Parameter {i}: requires_grad = {p.requires_grad}, shape = {p.shape}")

original_params = list(split_encoder.token_emb.original_emb.parameters())
print(f"Original embedding parameters: {len(original_params)}")
for i, p in enumerate(original_params):
    print(f"  Parameter {i}: requires_grad = {p.requires_grad}, shape = {p.shape}")
