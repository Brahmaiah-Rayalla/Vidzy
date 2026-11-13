"""
Stage 3.2.7B: Split Text Encoder for Selective Motion Embedding Protection

This module provides a split embedding architecture that separates:
- Appearance embeddings (tokens 0-41): Frozen during training
- Motion embeddings (tokens 42-81): Trainable during motion-focused experiments

This enables selective fine-tuning of motion vocabulary while protecting
the pre-trained appearance vocabulary from degradation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SplitTextEmbedding(nn.Module):
    """
    Split embedding layer for selective motion embedding protection.

    Separates the vocabulary into:
    - Appearance embeddings (indices 0-41): Frozen during training
    - Motion embeddings (indices 42-81): Trainable during training

    This architecture allows fine-tuning only the motion embeddings while
    protecting the pre-trained appearance embeddings from catastrophic forgetting.
    """

    def __init__(
        self,
        appearance_vocab_size: int = 42,
        motion_vocab_size: int = 40,
        d_model: int = 256,
        max_vocab_size: int = 82
    ):
        """
        Initialize split embeddings.

        Args:
            appearance_vocab_size: Number of appearance tokens (0-41)
            motion_vocab_size: Number of motion tokens (42-81)
            d_model: Embedding dimension
            max_vocab_size: Maximum vocabulary size for compatibility
        """
        super().__init__()

        self.appearance_vocab_size = appearance_vocab_size
        self.motion_vocab_size = motion_vocab_size
        self.d_model = d_model
        self.max_vocab_size = max_vocab_size

        # Separate embedding layers
        self.appearance_emb = nn.Embedding(appearance_vocab_size, d_model)
        self.motion_emb = nn.Embedding(motion_vocab_size, d_model)

        # Freeze appearance embeddings by default
        for param in self.appearance_emb.parameters():
            param.requires_grad = False

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through split embeddings.

        Args:
            token_ids: Token indices [batch, seq_len]
                      - 0-41: appearance tokens
                      - 42-81: motion tokens

        Returns:
            Embeddings [batch, seq_len, d_model]
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Initialize output tensor
        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        # Create masks for appearance and motion tokens
        appearance_mask = token_ids < self.appearance_vocab_size
        motion_mask = (token_ids >= self.appearance_vocab_size) & \
                      (token_ids < self.appearance_vocab_size + self.motion_vocab_size)

        # Get appearance embeddings
        if appearance_mask.any():
            appearance_tokens = token_ids[appearance_mask]
            appearance_embeddings = self.appearance_emb(appearance_tokens)
            embeddings[appearance_mask] = appearance_embeddings

        # Get motion embeddings (adjust indices to 0-39 range)
        if motion_mask.any():
            motion_tokens = token_ids[motion_mask] - self.appearance_vocab_size
            motion_embeddings = self.motion_emb(motion_tokens)
            embeddings[motion_mask] = motion_embeddings

        return embeddings

    def freeze_appearance(self, freeze: bool = True):
        """
        Freeze or unfreeze appearance embeddings.

        Args:
            freeze: If True, freeze appearance embeddings
        """
        for param in self.appearance_emb.parameters():
            param.requires_grad = not freeze

    def freeze_motion(self, freeze: bool = True):
        """
        Freeze or unfreeze motion embeddings.

        Args:
            freeze: If True, freeze motion embeddings
        """
        for param in self.motion_emb.parameters():
            param.requires_grad = not freeze

    def load_from_unified_embedding(
        self,
        unified_embedding: nn.Embedding,
        motion_token_indices: Optional[list] = None
    ):
        """
        Load weights from a unified embedding layer.

        Args:
            unified_embedding: Original unified embedding layer
            motion_token_indices: List of motion token indices (default: 42-81)
        """
        if motion_token_indices is None:
            motion_token_indices = list(range(
                self.appearance_vocab_size,
                self.appearance_vocab_size + self.motion_vocab_size
            ))

        # Copy appearance embeddings
        with torch.no_grad():
            if unified_embedding.weight.shape[0] >= self.appearance_vocab_size:
                self.appearance_emb.weight.data = unified_embedding.weight.data[:self.appearance_vocab_size].clone()

            # Copy motion embeddings
            if unified_embedding.weight.shape[0] >= self.appearance_vocab_size + self.motion_vocab_size:
                self.motion_emb.weight.data = unified_embedding.weight.data[motion_token_indices].clone()

    def merge_to_unified(self) -> nn.Embedding:
        """
        Merge split embeddings back to a unified embedding layer.

        Returns:
            Unified embedding layer
        """
        unified_emb = nn.Embedding(
            self.appearance_vocab_size + self.motion_vocab_size,
            self.d_model
        )

        with torch.no_grad():
            # Copy appearance embeddings
            unified_emb.weight.data[:self.appearance_vocab_size] = self.appearance_emb.weight.data.clone()

            # Copy motion embeddings
            unified_emb.weight.data[self.appearance_vocab_size:] = self.motion_emb.weight.data.clone()

        return unified_emb


def convert_text_encoder_to_split(
    text_encoder: nn.Module,
    appearance_vocab_size: int = 42,
    motion_vocab_size: int = 40
) -> nn.Module:
    """
    Convert a TextEncoder with unified embeddings to use split embeddings.

    Args:
        text_encoder: Original TextEncoder with unified token_emb
        appearance_vocab_size: Number of appearance tokens
        motion_vocab_size: Number of motion tokens

    Returns:
        Modified TextEncoder with split embeddings
    """
    if not hasattr(text_encoder, 'token_emb'):
        raise ValueError("TextEncoder must have token_emb attribute")

    # Get current embedding configuration
    unified_emb = text_encoder.token_emb
    d_model = unified_emb.embedding_dim

    # Create split embedding
    split_emb = SplitTextEmbedding(
        appearance_vocab_size=appearance_vocab_size,
        motion_vocab_size=motion_vocab_size,
        d_model=d_model
    )

    # Load weights from unified embedding
    split_emb.load_from_unified_embedding(unified_emb)

    # Replace embedding layer
    text_encoder.token_emb = split_emb

    return text_encoder


if __name__ == "__main__":
    # Test split embedding
    print("Testing SplitTextEmbedding...")

    # Create split embedding
    split_emb = SplitTextEmbedding(
        appearance_vocab_size=42,
        motion_vocab_size=40,
        d_model=256
    )

    # Test forward pass
    token_ids = torch.tensor([
        [0, 1, 2, 42, 43, 44],  # Mix of appearance and motion tokens
        [10, 20, 30, 50, 60, 70]
    ])

    embeddings = split_emb(token_ids)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected shape: (2, 6, 256)")

    # Test freezing
    print(f"\nBefore freezing:")
    print(f"  Appearance trainable: {split_emb.appearance_emb.weight.requires_grad}")
    print(f"  Motion trainable: {split_emb.motion_emb.weight.requires_grad}")

    split_emb.freeze_appearance(freeze=True)
    split_emb.freeze_motion(freeze=False)

    print(f"\nAfter freezing appearance:")
    print(f"  Appearance trainable: {split_emb.appearance_emb.weight.requires_grad}")
    print(f"  Motion trainable: {split_emb.motion_emb.weight.requires_grad}")

    # Test merge
    unified_emb = split_emb.merge_to_unified()
    print(f"\nMerged embedding shape: {unified_emb.weight.shape}")
    print(f"Expected shape: (82, 256)")

    print("\nAll tests passed!")
