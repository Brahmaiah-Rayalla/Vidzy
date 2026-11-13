"""
Check the actual vocab size.
"""

import torch
from pathlib import Path

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
checkpoint = torch.load(stage2_checkpoint, map_location='cpu')
actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

print(f"Actual vocab size: {actual_vocab_size}")
print(f"Expected original vocab size: 42")
print(f"Expected motion vocab size: 40")
print(f"Total expected: 82")

if actual_vocab_size == 82:
    print("\nVocab size matches expected (42 original + 40 motion)")
    print("Motion tokens are IDs 42-81")
else:
    print(f"\nWARNING: Vocab size {actual_vocab_size} does not match expected 82")
