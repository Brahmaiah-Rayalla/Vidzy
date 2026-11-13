"""
Diagnose the motion embedding gradient bug.
"""

import torch
import torch.nn as nn

class SplitEmbedding(nn.Module):
    """Current (buggy) implementation."""

    def __init__(self, original_vocab_size, motion_vocab_size, d_model):
        super().__init__()
        self.original_vocab_size = original_vocab_size
        self.motion_vocab_size = motion_vocab_size
        self.d_model = d_model

        self.original_emb = nn.Embedding(original_vocab_size, d_model)
        self.original_emb.weight.requires_grad = False

        self.motion_emb = nn.Embedding(motion_vocab_size, d_model)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        original_mask = token_ids < self.original_vocab_size
        motion_mask = ~original_mask

        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        # Get original embeddings
        if original_mask.any():
            original_ids = token_ids.clone()
            original_ids[~original_mask] = 0
            original_embs = self.original_emb(original_ids)
            embeddings[original_mask] = original_embs[original_mask]

        # Get motion embeddings (BUGGY)
        if motion_mask.any():
            motion_ids = token_ids - self.original_vocab_size
            motion_ids[~motion_mask] = 0
            motion_embs = self.motion_emb(motion_ids)
            embeddings[motion_mask] = motion_embs[motion_mask]  # BUG: incorrect indexing

        return embeddings


class SplitEmbeddingFixed(nn.Module):
    """Fixed implementation."""

    def __init__(self, original_vocab_size, motion_vocab_size, d_model):
        super().__init__()
        self.original_vocab_size = original_vocab_size
        self.motion_vocab_size = motion_vocab_size
        self.d_model = d_model

        self.original_emb = nn.Embedding(original_vocab_size, d_model)
        self.original_emb.weight.requires_grad = False

        self.motion_emb = nn.Embedding(motion_vocab_size, d_model)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        original_mask = token_ids < self.original_vocab_size
        motion_mask = ~original_mask

        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        # Get original embeddings
        if original_mask.any():
            original_ids = token_ids.clone()
            original_ids[~original_mask] = 0
            original_embs = self.original_emb(original_ids)
            embeddings[original_mask] = original_embs[original_mask]

        # Get motion embeddings (FIXED)
        if motion_mask.any():
            motion_ids = token_ids - self.original_vocab_size
            # Only look up motion tokens
            motion_embs = self.motion_emb(motion_ids[motion_mask])
            # Assign using the mask
            embeddings[motion_mask] = motion_embs

        return embeddings


def test_gradient_flow():
    print("=" * 80)
    print("TESTING GRADIENT FLOW")
    print("=" * 80)

    # Create test data
    original_vocab_size = 42
    motion_vocab_size = 40
    d_model = 256
    batch_size = 2
    seq_len = 10

    # Create token IDs with some motion tokens (IDs >= 42)
    token_ids = torch.randint(0, original_vocab_size + motion_vocab_size, (batch_size, seq_len))

    print(f"\nToken IDs:\n{token_ids}")
    print(f"Motion tokens present: {(token_ids >= original_vocab_size).any().item()}")

    # Test buggy version
    print("\n" + "=" * 80)
    print("BUGGY VERSION")
    print("=" * 80)

    split_emb_buggy = SplitEmbedding(original_vocab_size, motion_vocab_size, d_model)
    embeddings_buggy = split_emb_buggy(token_ids)
    loss_buggy = embeddings_buggy.sum()
    loss_buggy.backward()

    motion_grad_buggy = split_emb_buggy.motion_emb.weight.grad is not None
    print(f"Motion embedding has gradient: {motion_grad_buggy}")

    if motion_grad_buggy:
        print(f"Motion embedding gradient norm: {split_emb_buggy.motion_emb.weight.grad.norm().item()}")

    # Test fixed version
    print("\n" + "=" * 80)
    print("FIXED VERSION")
    print("=" * 80)

    split_emb_fixed = SplitEmbeddingFixed(original_vocab_size, motion_vocab_size, d_model)
    embeddings_fixed = split_emb_fixed(token_ids)
    loss_fixed = embeddings_fixed.sum()
    loss_fixed.backward()

    motion_grad_fixed = split_emb_fixed.motion_emb.weight.grad is not None
    print(f"Motion embedding has gradient: {motion_grad_fixed}")

    if motion_grad_fixed:
        print(f"Motion embedding gradient norm: {split_emb_fixed.motion_emb.weight.grad.norm().item()}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Buggy version - Motion gradient: {motion_grad_buggy}")
    print(f"Fixed version - Motion gradient: {motion_grad_fixed}")

    if not motion_grad_buggy and motion_grad_fixed:
        print("\n✅ BUG CONFIRMED: Buggy version does not receive gradients, fixed version does.")
    else:
        print("\n❓ UNEXPECTED: Results do not match expected bug pattern.")


if __name__ == "__main__":
    test_gradient_flow()
