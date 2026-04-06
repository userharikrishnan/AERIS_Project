"""
AERIS SLM — Small Language Model v2.0
Upgraded Transformer Decoder for 15k+ training pairs.

Architecture upgrades:
- embed_dim:  256 → 384
- num_layers: 4   → 6
- num_heads:  4   → 6
- ffn_dim:    512 → 1536  (4× embed_dim ratio maintained)
- max_len:    256 → 512
- dropout:    0.1 → 0.15
- Adds n-gram repetition penalty in generate()
- Adds top-k nucleus sampling directly on model
- ~12M parameters — runs fully on CPU with 16GB RAM
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embeddings import EmbeddingLayer


# =========================================================
# Positional Encoding — Sinusoidal (no learned params, OOD-safe)
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
        if embed_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)   # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =========================================================
# Pre-Norm Transformer Decoder Layer
# (Pre-LayerNorm is more stable during long training runs)
# =========================================================
class AerisTransformerLayer(nn.Module):
    """
    Pre-LN Transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual.
    Empirically more stable than Post-LN, especially on CPU with small batch sizes.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm1    = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2    = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout  = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN → attention → residual
        nx = self.norm1(x)
        attn_out, _ = self.self_attn(nx, nx, nx, attn_mask=causal_mask,
                                     is_causal=(causal_mask is not None))
        x = x + self.dropout(attn_out)

        # Pre-LN → FFN → residual
        x = x + self.ffn(self.norm2(x))
        return x


# =========================================================
# AerisSLM — Upgraded Transformer Decoder (PRIMARY MODEL)
# =========================================================
class AerisSLM(nn.Module):
    """
    AERIS Small Language Model — Transformer Decoder v2.0
    Production-grade. No third-party LLM dependencies.

    Defaults tuned for 15k training pairs on 16GB RAM / i5 CPU:
        vocab_size: dynamic (from tokenizer)
        embed_dim:  384
        num_layers: 6
        num_heads:  6
        ffn_dim:    1536   (4× embed_dim)
        dropout:    0.15
        max_len:    512
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int = 384,
        hidden_dim: int = 384,   # alias kept for backward compat
        num_layers: int = 6,
        num_heads:  int = 6,
        ffn_dim:    int = 1536,
        dropout:    float = 0.15,
        max_len:    int = 512,
    ):
        super().__init__()

        # Use max of embed_dim / hidden_dim for backward compat with old callers
        d_model = max(embed_dim, hidden_dim)

        # Ensure divisibility by num_heads
        if d_model % num_heads != 0:
            d_model = num_heads * ((d_model // num_heads) + 1)

        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding    = EmbeddingLayer(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            AerisTransformerLayer(d_model, num_heads, ffn_dim if ffn_dim else d_model * 4, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.fc   = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: output projection shares embedding weights
        # Reduces params & improves token prediction quality
        if d_model == self.embedding.embedding.embedding_dim:
            self.fc.weight = self.embedding.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Xavier / scaled init for stable convergence on CPU."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular float mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask.bool(), float('-inf'))
        return mask

    def forward(self, token_ids: torch.Tensor):
        """
        Args:
            token_ids: (B, T) integer token IDs
        Returns:
            logits:       (B, T, V)
            attn_weights: None  (compat with legacy interface callers)
        """
        B, T      = token_ids.shape
        device    = token_ids.device
        causal    = self._make_causal_mask(T, device)

        x = self.embedding(token_ids)    # (B, T, d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, causal)

        x      = self.norm(x)
        logits = self.fc(x)              # (B, T, V)

        return logits, None  # None keeps interface compat

    def predict_next(self, token_ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Greedy next-token prediction. Used internally by generate()."""
        logits, _ = self.forward(token_ids)
        scaled    = logits[:, -1, :] / max(temperature, 1e-7)
        return torch.argmax(torch.softmax(scaled, dim=-1), dim=-1)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids:      torch.Tensor,
        max_new_tokens:  int   = 80,
        temperature:     float = 0.75,
        top_k:           int   = 50,
        top_p:           float = 0.92,
        eos_id:          int   = 7,     # <EOS>
        pad_id:          int   = 0,
        repetition_penalty: float = 1.3,
        ngram_block_size: int  = 3,
    ) -> torch.Tensor:
        """
        Nucleus (top-p) sampling with n-gram repetition blocking.

        Args:
            prompt_ids:         (1, T) seed token IDs
            max_new_tokens:     max tokens to generate beyond prompt
            temperature:        sampling temperature (lower = more focused)
            top_k:              keep top-k logits before nucleus filter
            top_p:              nucleus probability threshold
            eos_id:             stop token ID
            pad_id:             suppress pad token from sampling
            repetition_penalty: penalise recently generated tokens (>1 = less likely)
            ngram_block_size:   block repeating n-grams of this size (0 = off)

        Returns:
            (1, T + new_tokens) integer tensor
        """
        self.eval()
        ids      = prompt_ids.clone()
        new_toks: List[int] = []

        for _ in range(max_new_tokens):
            logits, _ = self.forward(ids)
            logits    = logits[:, -1, :] / max(temperature, 1e-7)  # (1, V)

            # Suppress pad / unk absolutely
            logits[0, pad_id] = -1e9
            logits[0, 1]      = -1e9   # <UNK>

            # Token-level repetition penalty (applied over entire new_toks)
            if repetition_penalty != 1.0 and new_toks:
                for tid in set(new_toks[-20:]):
                    logits[0, tid] = logits[0, tid] / repetition_penalty if logits[0, tid] > 0 \
                                     else logits[0, tid] * repetition_penalty

            # N-gram blocking: suppress any token that would create a repeated n-gram
            if ngram_block_size > 0 and len(new_toks) >= ngram_block_size - 1:
                suffix = tuple(new_toks[-(ngram_block_size - 1):])
                n      = ngram_block_size - 1
                for i in range(len(new_toks) - n):
                    window = tuple(new_toks[i: i + n])
                    if window == suffix:
                        banned = new_toks[i + n]
                        logits[0, banned] = -1e9

            # Top-k filter
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_topk     = topk_vals[:, -1].unsqueeze(-1)
                logits       = logits.masked_fill(logits < min_topk, -1e9)

            # Nucleus (top-p) filter
            probs       = torch.softmax(logits, dim=-1)
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cumsum      = torch.cumsum(sorted_p, dim=-1)
            remove      = (cumsum - sorted_p) >= top_p
            sorted_p[remove] = 0.0
            sorted_p   /= sorted_p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            next_id_rel = torch.multinomial(sorted_p, 1)
            next_token  = sorted_idx.gather(-1, next_id_rel)   # (1, 1)

            token_val = next_token.item()
            ids       = torch.cat([ids, next_token], dim=1)
            new_toks.append(token_val)

            if token_val == eos_id:
                break

        return ids


# =========================================================
# Legacy GRU SLM — kept for backward compat with old checkpoints
# =========================================================
class AerisGRUSLM(nn.Module):
    """
    Original GRU-based SLM.
    Only use this for loading old aeris_slm.pt checkpoints that were
    trained before the Transformer upgrade.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        from models.attention import SelfAttention
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.rnn       = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.norm      = nn.LayerNorm(hidden_dim)
        self.dropout   = nn.Dropout(0.1)
        self.attention = SelfAttention(hidden_dim)
        self.fc        = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids):
        x          = self.embedding(token_ids)
        rnn_out, _ = self.rnn(x)
        rnn_out    = self.norm(rnn_out)
        rnn_out    = self.dropout(rnn_out)
        attended, attn_weights = self.attention(rnn_out)
        logits     = self.fc(attended)
        return logits, attn_weights

    def predict_next(self, token_ids, temperature: float = 1.0):
        logits, _ = self.forward(token_ids)
        scaled    = logits[:, -1, :] / max(temperature, 1e-7)
        probs     = F.softmax(scaled, dim=-1)
        return torch.argmax(probs, dim=-1)