"""
Text encoder with split embedding tables for selective training.

Separates embeddings into:
- Original appearance embeddings (frozen)
- Motion embeddings (trainable)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from .text_encoder import PositionalEncoding, TransformerEncoderLayer


class SplitEmbedding(nn.Module):
    """
    Split embedding layer with separate frozen and trainable tables.

    Args:
        original_vocab_size: Size of original vocabulary (0 to original_vocab_size-1)
        motion_vocab_size: Size of motion vocabulary (original_vocab_size to original_vocab_size+motion_vocab_size-1)
        d_model: Embedding dimension
        original_embeddings: Pre-trained original embeddings (optional)
    """

    def __init__(
        self,
        original_vocab_size: int,
        motion_vocab_size: int,
        d_model: int,
        original_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.original_vocab_size = original_vocab_size
        self.motion_vocab_size = motion_vocab_size
        self.total_vocab_size = original_vocab_size + motion_vocab_size
        self.d_model = d_model

        # Frozen original embeddings
        self.original_emb = nn.Embedding(original_vocab_size, d_model)
        if original_embeddings is not None:
            self.original_emb.weight.data.copy_(original_embeddings)
        self.original_emb.weight.requires_grad = False

        # Trainable motion embeddings
        self.motion_emb = nn.Embedding(motion_vocab_size, d_model)
        # Initialize motion embeddings similarly to original (for stability)
        if original_embeddings is not None:
            # Use mean of original embeddings as initialization
            mean_emb = original_embeddings.mean(dim=0)
            self.motion_emb.weight.data.normal_(mean=0.0, std=0.02)  # Fix: use float, not Tensor
        else:
            self.motion_emb.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through split embeddings.

        Args:
            token_ids: Token indices [batch, seq_len]

        Returns:
            Embeddings [batch, seq_len, d_model]
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Split tokens into original and motion
        original_mask = token_ids < self.original_vocab_size
        motion_mask = ~original_mask

        # Initialize output by looking up all tokens in both tables
        # and then selecting the appropriate one
        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        # Get original embeddings for all positions
        original_ids = token_ids.clone()
        original_ids[~original_mask] = 0  # Pad motion tokens with dummy
        original_embs = self.original_emb(original_ids)

        # Get motion embeddings for all positions
        motion_ids = token_ids - self.original_vocab_size
        motion_ids[~motion_mask] = 0  # Pad original tokens with dummy
        motion_embs = self.motion_emb(motion_ids)

        # Select embeddings based on mask using torch.where
        # This preserves gradient connection to motion_emb
        embeddings = torch.where(original_mask.unsqueeze(-1), original_embs, motion_embs)

        return embeddings

    def get_original_embeddings(self) -> torch.Tensor:
        """Get frozen original embeddings."""
        return self.original_emb.weight.data.clone()

    def get_motion_embeddings(self) -> torch.Tensor:
        """Get trainable motion embeddings."""
        return self.motion_emb.weight.data.clone()

    def verify_original_preserved(self, initial_embeddings: torch.Tensor) -> bool:
        """Verify that original embeddings haven't changed."""
        current = self.get_original_embeddings()
        return torch.allclose(current, initial_embeddings, atol=1e-6)


class TextEncoderSplit(nn.Module):
    """
    Text encoder with split embeddings for selective training.

    Architecture:
    - Split embedding layer (frozen original, trainable motion)
    - Positional encoding
    - Transformer encoder layers (frozen)
    - Output: token-level representations + attention mask
    """

    def __init__(
        self,
        original_vocab_size: int,
        motion_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        original_embeddings: Optional[torch.Tensor] = None,
        freeze_transformer: bool = True
    ):
        super().__init__()

        self.original_vocab_size = original_vocab_size
        self.motion_vocab_size = motion_vocab_size
        self.total_vocab_size = original_vocab_size + motion_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # Split embeddings
        self.token_emb = SplitEmbedding(
            original_vocab_size,
            motion_vocab_size,
            d_model,
            original_embeddings
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)

        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Freeze transformer layers if specified
        if freeze_transformer:
            for layer in self.layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text tokens.

        Args:
            token_ids: Token indices [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (1 for real tokens, 0 for padding)

        Returns:
            Tuple of (encoded representations, attention mask)
            - encoded: [batch, seq_len, d_model]
            - mask: [batch, seq_len]
        """
        # Token embeddings (through split embedding)
        x = self.token_emb(token_ids)  # [batch, seq_len, d_model]

        # Positional encoding
        x = self.pos_enc(x)

        # Dropout
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        return x, attention_mask

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def verify_original_embeddings_preserved(self, initial_embeddings: torch.Tensor) -> bool:
        """Verify that original embeddings haven't changed."""
        return self.token_emb.verify_original_preserved(initial_embeddings)

    def get_initial_original_embeddings(self) -> torch.Tensor:
        """Get the initial original embeddings for comparison."""
        return self.token_emb.get_original_embeddings()


def load_from_standard_encoder(
    standard_encoder,
    motion_vocab_size: int,
    freeze_transformer: bool = True,
    original_vocab_size: Optional[int] = None
) -> TextEncoderSplit:
    """
    Create a split embedding encoder from a standard encoder.

    Args:
        standard_encoder: Standard TextEncoder to convert
        motion_vocab_size: Number of motion tokens to add
        freeze_transformer: Whether to freeze transformer layers
        original_vocab_size: Original vocab size before motion expansion.
                           If None, uses standard_encoder.vocab_size.

    Returns:
        TextEncoderSplit with split embeddings
    """
    if original_vocab_size is None:
        original_vocab_size = standard_encoder.vocab_size

    # Get original embeddings (only the original portion if vocab was expanded)
    if standard_encoder.vocab_size > original_vocab_size:
        # Encoder was already expanded, extract only original embeddings
        original_embeddings = standard_encoder.token_emb.weight.data[:original_vocab_size].clone()
    else:
        # Encoder not expanded, use all embeddings
        original_embeddings = standard_encoder.token_emb.weight.data.clone()

    # Create split encoder
    split_encoder = TextEncoderSplit(
        original_vocab_size=original_vocab_size,
        motion_vocab_size=motion_vocab_size,
        d_model=standard_encoder.d_model,
        n_heads=standard_encoder.n_heads,
        n_layers=standard_encoder.n_layers,
        d_ff=standard_encoder.d_ff,
        max_seq_len=standard_encoder.max_seq_len,
        dropout=0.0,  # Match standard encoder
        original_embeddings=original_embeddings,
        freeze_transformer=freeze_transformer
    )

    # Copy transformer weights from the standard encoder.
    # freeze_transformer only controls requires_grad; weights must ALWAYS be
    # copied or the split encoder runs with randomly initialized transformer
    # layers (L.13/L.14 parity defect).
    for i, layer in enumerate(standard_encoder.layers):
        split_encoder.layers[i].load_state_dict(layer.state_dict())

    return split_encoder
