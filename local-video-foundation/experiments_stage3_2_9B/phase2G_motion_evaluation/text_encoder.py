"""
Text encoder for conditioning diffusion models.

Implements a simple tokenizer, embeddings, and a small Transformer encoder
trained from scratch.
"""

import json
import math
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from collections import Counter


class SimpleTokenizer:
    """
    Simple word-level tokenizer with vocabulary building.

    Features:
    - Lowercase conversion
    - Basic punctuation handling
    - Special tokens: <PAD>, <UNK>, <EOS>
    - Configurable vocabulary size
    """

    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<EOS>': 2,
    }

    def __init__(self, vocab_size: int = 1000, min_freq: int = 2):
        """
        Initialize the tokenizer.

        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a word to be included
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word_to_idx = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.idx_to_word = {k: v for k, v in self.SPECIAL_TOKENS.items()}
        self.word_freq = Counter()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Basic punctuation handling
        for punct in ['.', ',', '!', '?', ';', ':']:
            text = text.replace(punct, f' {punct}')

        # Split on whitespace
        tokens = text.split()

        return tokens

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of training texts
        """
        # Count word frequencies
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)

        # Add most common words to vocabulary
        # Reserve space for special tokens
        max_words = self.vocab_size - len(self.SPECIAL_TOKENS)

        # Filter by minimum frequency
        common_words = self.word_freq.most_common(max_words)
        common_words = [(w, c) for w, c in common_words if c >= self.min_freq]

        # Add to vocabulary
        for word, _ in common_words:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        print(f"Built vocabulary: {len(self.word_to_idx)} tokens")
        print(f"  Special tokens: {len(self.SPECIAL_TOKENS)}")
        print(f"  Regular tokens: {len(self.word_to_idx) - len(self.SPECIAL_TOKENS)}")

    def encode(self, text: str, max_length: Optional[int] = None) -> Tuple[List[int], List[int]]:
        """
        Encode text to token indices.

        Args:
            text: Input text
            max_length: Maximum sequence length (pad if shorter, truncate if longer)

        Returns:
            Tuple of (token indices, attention mask)
        """
        tokens = self.tokenize(text)

        # Convert to indices
        indices = []
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                indices.append(self.SPECIAL_TOKENS['<UNK>'])

        # Add EOS token
        indices.append(self.SPECIAL_TOKENS['<EOS>'])

        # Pad or truncate
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length-1] + [self.SPECIAL_TOKENS['<EOS>']]
            else:
                # Pad
                pad_length = max_length - len(indices)
                indices += [self.SPECIAL_TOKENS['<PAD>']] * pad_length

        # Create attention mask (1 for real tokens, 0 for padding)
        mask = [1 if idx != self.SPECIAL_TOKENS['<PAD>'] else 0 for idx in indices]

        return indices, mask

    def decode(self, indices: List[int]) -> str:
        """
        Decode token indices to text.

        Args:
            indices: Token indices

        Returns:
            Decoded text
        """
        tokens = []
        for idx in indices:
            if idx in self.idx_to_word:
                token = self.idx_to_word[idx]
                if token == '<EOS>':
                    break
                if token not in ['<PAD>', '<UNK>']:
                    tokens.append(token)

        return ' '.join(tokens)

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        data = {
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': {str(k): v for k, v in self.idx_to_word.items()},
            'word_freq': dict(self.word_freq),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved tokenizer to {path}")

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data['vocab_size'], min_freq=data['min_freq'])
        tokenizer.word_to_idx = {k: int(v) if isinstance(v, str) else v for k, v in data['word_to_idx'].items()}
        tokenizer.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
        tokenizer.word_freq = Counter(data['word_freq'])

        print(f"Loaded tokenizer from {path}")
        return tokenizer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query: Query tensor [batch, query_len, d_model]
            key: Key tensor [batch, key_len, d_model]
            value: Value tensor [batch, value_len, d_model]
            mask: Attention mask [batch, key_len] (for cross-attention)

        Returns:
            Output tensor [batch, query_len, d_model]
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        # Project to Q, K, V
        Q = self.q_proj(query)  # [batch, query_len, d_model]
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head
        Q = Q.view(batch_size, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, n_heads, seq_len, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Shape: [batch, n_heads, query_len, key_len]

        # Apply mask if provided
        if mask is not None:
            # mask: [batch, key_len] -> [batch, 1, 1, key_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # Shape: [batch, n_heads, query_len, head_dim]

        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, query_len, self.d_model)

        # Output projection
        output = self.out_proj(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0
    ):
        """
        Initialize encoder layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class TextEncoder(nn.Module):
    """
    Text encoder for conditioning diffusion models.

    Architecture:
    - Token embeddings
    - Positional encoding
    - Multiple Transformer encoder layers
    - Output: token-level representations + attention mask
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize text encoder.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)

        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

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
        # Token embeddings
        x = self.token_emb(token_ids)  # [batch, seq_len, d_model]

        # Positional encoding
        x = self.pos_enc(x)

        # Dropout
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        return x, attention_mask

    def save(self, path: str, tokenizer: SimpleTokenizer) -> None:
        """Save encoder and tokenizer."""
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': len(self.layers),
            'd_ff': self.layers[0].ffn[0].out_features,
            'max_seq_len': self.max_seq_len,
        }, path)

        # Save tokenizer
        tokenizer_path = path.replace('.pt', '_tokenizer.json')
        tokenizer.save(tokenizer_path)

        print(f"Saved text encoder to {path}")
        print(f"Saved tokenizer to {tokenizer_path}")

    @classmethod
    def load(cls, path: str, tokenizer_path: str) -> Tuple['TextEncoder', SimpleTokenizer]:
        """Load encoder and tokenizer."""
        checkpoint = torch.load(path, map_location='cpu')
        tokenizer = SimpleTokenizer.load(tokenizer_path)

        encoder = cls(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            n_heads=checkpoint['n_heads'],
            n_layers=checkpoint['n_layers'],
            d_ff=checkpoint['d_ff'],
            max_seq_len=checkpoint['max_seq_len'],
        )
        encoder.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded text encoder from {path}")
        return encoder, tokenizer


if __name__ == "__main__":
    # Test the tokenizer
    texts = [
        "A small red circle on a blue background.",
        "A large green triangle near the center.",
        "A yellow square on a white background.",
    ]

    tokenizer = SimpleTokenizer(vocab_size=100, min_freq=1)
    tokenizer.build_vocab(texts)

    for text in texts:
        indices, mask = tokenizer.encode(text, max_length=20)
        decoded = tokenizer.decode(indices)
        print(f"Original: {text}")
        print(f"Indices: {indices}")
        print(f"Mask: {mask}")
        print(f"Decoded: {decoded}")
        print()

    # Test the encoder
    encoder = TextEncoder(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=20
    )

    token_ids = torch.tensor([indices])
    attention_mask = torch.tensor([mask], dtype=torch.float)

    encoded, mask = encoder(token_ids, attention_mask)
    print(f"Encoded shape: {encoded.shape}")
    print(f"Mask shape: {mask.shape}")
