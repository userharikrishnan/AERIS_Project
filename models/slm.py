"""
AERIS SLM — Small Language Model
Upgraded to Transformer decoder architecture.
Custom-built: no third-party LLM dependencies.

Architecture:
- Token Embedding + Positional Encoding
- N Transformer Decoder layers (multi-head self-attention + FFN)
- Layer normalization throughout
- Output projection to vocabulary

The legacy GRU model class is preserved as AerisGRUSLM for backward compat
with existing checkpoints. AerisSLM now points to the Transformer version.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embeddings import EmbeddingLayer


# =========================================================
# Positional Encoding (Sinusoidal)
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =========================================================
# Transformer Decoder Layer (custom)
# =========================================================
class AerisTransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        # Causal self-attention (look-ahead mask for autoregressive generation)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask, is_causal=(causal_mask is not None))
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# =========================================================
# AerisSLM — Transformer Decoder (NEW PRIMARY MODEL)
# =========================================================
class AerisSLM(nn.Module):
    """
    AERIS Small Language Model — Transformer Decoder Architecture
    Custom-built, no third-party LLM dependencies.

    Args:
        vocab_size: Total vocabulary size
        embed_dim: Token embedding dimension (default: 256)
        hidden_dim: Alias for embed_dim (backward compat)
        num_layers: Number of Transformer decoder layers (default: 4)
        num_heads: Number of attention heads (default: 4)
        ffn_dim: Feed-forward inner dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
        max_len: Maximum sequence length (default: 256)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,   # kept for backward compat
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_len: int = 256
    ):
        super().__init__()
        # Use max(embed_dim, hidden_dim) so old callers with hidden_dim=128 still work
        d_model = max(embed_dim, hidden_dim)
        # Ensure d_model is divisible by num_heads
        if d_model % num_heads != 0:
            d_model = num_heads * (d_model // num_heads + 1)

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            AerisTransformerLayer(d_model, num_heads, ffn_dim * 2, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

        # Tie embedding and output weights (reduces parameters, improves quality)
        if d_model == self.embedding.embedding.embedding_dim:
            self.fc.weight = self.embedding.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        mask = mask.masked_fill(mask, float("-inf"))
        return mask

    def forward(self, token_ids: torch.Tensor):
        """
        Args:
            token_ids: (B, T) integer token IDs
        Returns:
            logits: (B, T, V)
            attn_weights: None (for interface compat with old SLM)
        """
        B, T = token_ids.shape
        device = token_ids.device

        x = self.embedding(token_ids)         # (B, T, d_model)
        x = self.pos_encoding(x)              # (B, T, d_model) + positional

        causal_mask = self._make_causal_mask(T, device)

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.norm(x)
        logits = self.fc(x)                   # (B, T, V)

        return logits, None  # None keeps interface compat with old SLM

    def predict_next(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Predict the next token ID."""
        logits, _ = self.forward(token_ids)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        return torch.argmax(probs, dim=-1)


# =========================================================
# Legacy GRU SLM — kept for backward compat with old checkpoints
# =========================================================
class AerisGRUSLM(nn.Module):
    """Original GRU-based SLM. Use for loading old aeris_slm.pt checkpoints."""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        from models.attention import SelfAttention
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        rnn_out, _ = self.rnn(x)
        rnn_out = self.norm(rnn_out)
        rnn_out = self.dropout(rnn_out)
        attended, attn_weights = self.attention(rnn_out)
        logits = self.fc(attended)
        return logits, attn_weights

    def predict_next(self, token_ids):
        logits, _ = self.forward(token_ids)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        return torch.argmax(probs, dim=-1)