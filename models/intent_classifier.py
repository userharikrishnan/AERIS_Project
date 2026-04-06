import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embeddings import EmbeddingLayer

# =========================================================
# Intent Classification Constants
# SYNCHRONIZED WITH TRAINING DATA — do not reorder indices
# =========================================================

INTENT_TO_IDX = {
    # Application Control
    "OPEN_APP":         0,
    "CLOSE_APP":        1,

    # Web Operations
    "WEB_SEARCH":       2,
    "WEB_NAVIGATE":     3,
    "WEB_SCRAPE":       27,

    # File System
    "FILE_READ":        4,
    "FILE_WRITE":       5,
    "FILE_DELETE":      6,
    "FILE_LIST":        7,

    # Report Generation
    "GENERATE_REPORT":  28,

    # Memory Operations
    "MEMORY_STORE":     8,
    "MEMORY_RECALL":    9,
    "MEMORY_FORGET":    10,

    # Goal Management
    "GOAL_CREATE":      11,
    "GOAL_LIST":        12,
    "GOAL_PAUSE":       13,
    "GOAL_RESUME":      14,
    "GOAL_COMPLETE":    15,

    # Vision / Screen
    "VISION_QUERY":     16,
    "READ_SCREEN":      17,
    "ACTIVE_WINDOW":    18,
    "LIST_WINDOWS":     19,
    "SCREENSHOT":       30,

    # System Control
    "ROLLBACK":         20,
    "CONFIRM":          21,
    "CANCEL":           22,

    # System Information
    "SYSTEM_INFO":      29,

    # Cognitive
    "REASONING":        23,
    "IDENTITY_QUERY":   24,

    # Fallback
    "CHAT":             25,
    "UNKNOWN":          26,
}

NUM_INTENT_CLASSES = len(INTENT_TO_IDX)
IDX_TO_INTENT      = {v: k for k, v in INTENT_TO_IDX.items()}


# =========================================================
# Intent Classifier — v2.0
# Architecture: Embedding → BiGRU (512-dim) → Multi-Head
# Attention (4 heads) → Residual Classifier → Output
#
# Upgrades from v1:
#   - embed_dim:  128 → 256
#   - hidden_dim: 256 → 512  (BiGRU: 512 * 2 = 1024 context)
#   - Fixed double-forward pass bug in predict_with_confidence
#   - Confidence threshold: 0.60 → 0.55
#   - Temperature calibration: 1.4 → 1.2
#   - Clean single-pass inference via _single_forward()
# =========================================================

class IntentClassifier(nn.Module):
    """
    Production Neural Intent Classifier for AERIS v2.0

    Architecture:
        Embedding (256) → embed_norm → embed_dropout
        → BiGRU (512 × 2 layers, bidirectional) → gru_norm
        → Multi-head attention (4 heads, each learns different semantic focus)
        → Head fusion (learned attention weighting across heads)
        → Residual MLP classifier
        → Output logits (NUM_INTENT_CLASSES)
        → [Uncertainty head] (sigmoid aleatoric uncertainty estimate)

    Designed for 15k+ training pairs on CPU.
    """

    def __init__(
        self,
        vocab_size:     int,
        embed_dim:      int = 256,        # upgraded from 128
        hidden_dim:     int = 512,        # upgraded from 256
        num_classes:    int = NUM_INTENT_CLASSES,
        num_heads:      int = 4,
        dropout:        float = 0.35,     # slightly relaxed for larger model
        num_gru_layers: int = 2,
    ):
        super().__init__()

        self.vocab_size   = vocab_size
        self.hidden_dim   = hidden_dim
        self.num_classes  = num_classes
        self.num_heads    = num_heads
        self.dropout_rate = dropout

        # -- Input embedding --
        self.embedding    = EmbeddingLayer(vocab_size, embed_dim)
        self.embed_norm   = nn.LayerNorm(embed_dim)
        self.embed_drop   = nn.Dropout(dropout / 2)

        # -- Bidirectional GRU encoder --
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        gru_out_dim = hidden_dim * 2   # bidirectional
        self.gru_norm = nn.LayerNorm(gru_out_dim)

        # -- Multi-head attention pooling --
        # Each head learns to attend to different intent-relevant tokens
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gru_out_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1),
            )
            for _ in range(num_heads)
        ])

        # Learned head fusion
        self.head_fusion = nn.Sequential(
            nn.Linear(num_heads * gru_out_dim, gru_out_dim),
            nn.LayerNorm(gru_out_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )

        # -- Context projection (Residual) --
        self.context_proj = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim),
            nn.LayerNorm(gru_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # -- Deep residual classifier --
        clf_dim   = gru_out_dim
        half_dim  = hidden_dim

        self.clf_block1 = nn.Sequential(
            nn.Linear(clf_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.clf_block2 = nn.Sequential(
            nn.Linear(hidden_dim, half_dim),
            nn.LayerNorm(half_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.skip_proj   = nn.Linear(clf_dim, half_dim)
        self.output_proj = nn.Linear(half_dim, num_classes)

        # -- Aleatoric uncertainty head --
        self.uncertainty_head = nn.Sequential(
            nn.Linear(half_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    # ------------------------------------------------------------------
    # Internal single-pass encoder (avoids double-forward bug)
    # ------------------------------------------------------------------

    def _encode(self, token_ids: torch.Tensor):
        """
        Full encode pass returning (logits, features, head_weights).
        Called by forward(), predict_with_confidence(), and get_attention_visualization().
        SINGLE PASS — no redundant recomputation.
        """
        # Embed + normalize
        embedded = self.embed_drop(self.embed_norm(self.embedding(token_ids)))  # (B, T, E)

        # BiGRU encode
        gru_out, _ = self.gru(embedded)                  # (B, T, H*2)
        gru_out    = self.gru_norm(gru_out)

        # Multi-head attention pooling
        head_contexts = []
        head_weights  = []
        for head in self.attention_heads:
            w = head(gru_out)                            # (B, T, 1)
            c = torch.sum(gru_out * w, dim=1)           # (B, H*2)
            head_contexts.append(c)
            head_weights.append(w)

        # Fuse heads: concat → linear projection
        fused   = torch.cat(head_contexts, dim=-1)      # (B, num_heads * H*2)
        context = self.head_fusion(fused)               # (B, H*2)

        # Context residual
        context = self.context_proj(context) + context  # (B, H*2)

        # Classifier
        x1 = self.clf_block1(context)                   # (B, H)
        x2 = self.clf_block2(x1)                        # (B, H/2)
        x  = x2 + self.skip_proj(context)               # residual

        logits      = self.output_proj(x)               # (B, num_classes)
        uncertainty = self.uncertainty_head(x)          # (B, 1)

        return logits, uncertainty, context, head_weights

    # ------------------------------------------------------------------
    # Forward (for training)
    # ------------------------------------------------------------------

    def forward(self, token_ids: torch.Tensor, return_attention: bool = False):
        """
        Standard forward pass for training.

        Args:
            token_ids:       (B, T) token indices
            return_attention: return attention weights for visualization

        Returns:
            logits: (B, num_classes)  — or (logits, head_weights) if return_attention=True
        """
        logits, _, _, head_weights = self._encode(token_ids)
        if return_attention:
            return logits, head_weights
        return logits

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict(self, token_ids: torch.Tensor, confidence_threshold: float = 0.55) -> str:
        """
        Fast production prediction — single forward pass.

        Uses margin-based uncertainty for robust CHAT/action disambiguation.
        Confidence threshold: 0.55 (lower than v1's 0.60 — large-data model is more reliable).
        """
        self.eval()
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        with torch.no_grad():
            logits, _, _, _ = self._encode(token_ids)
            probs            = torch.softmax(logits / 1.2, dim=1)   # T=1.2 calibrated

            top2_p, top2_i  = torch.topk(probs, 2, dim=1)
            pred_idx         = top2_i[0][0].item()
            confidence       = top2_p[0][0].item()
            second_conf      = top2_p[0][1].item()
            margin           = confidence - second_conf

            intent       = IDX_TO_INTENT.get(pred_idx, 'UNKNOWN')
            second_intent = IDX_TO_INTENT.get(top2_i[0][1].item(), 'UNKNOWN')

            # Weak margin → favour CHAT if it's second
            if intent != 'CHAT' and second_intent == 'CHAT' and margin < 0.20:
                return 'CHAT'

            # Below threshold → CHAT fallback
            if confidence < confidence_threshold:
                return 'CHAT'

            # UNKNOWN is always mapped to CHAT
            if intent == 'UNKNOWN':
                return 'CHAT'

        return intent

    def predict_with_confidence(self, token_ids: torch.Tensor) -> dict:
        """
        Rich prediction for AERIS decision engine — SINGLE forward pass (bug fixed).

        Returns:
            intent, confidence, uncertainty, margin, alternatives (top-3), is_uncertain
        """
        self.eval()
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        with torch.no_grad():
            logits, uncertainty_t, _, _ = self._encode(token_ids)
            probs                        = torch.softmax(logits / 1.2, dim=1)[0]

            top3_p, top3_i = torch.topk(probs, 3)
            margin          = (top3_p[0] - top3_p[1]).item()
            uncertainty_score = uncertainty_t[0].item()

            is_uncertain = (
                margin < 0.12
                or top3_p[0].item() < 0.55
                or uncertainty_score > 0.55
            )

            return {
                'intent':     IDX_TO_INTENT.get(top3_i[0].item(), 'UNKNOWN'),
                'confidence': round(top3_p[0].item(), 4),
                'uncertainty': round(1 - top3_p[0].item(), 4),
                'margin':     round(margin, 4),
                'uncertainty_score': round(uncertainty_score, 4),
                'alternatives': [
                    {
                        'intent':      IDX_TO_INTENT.get(idx.item(), 'UNKNOWN'),
                        'probability': round(p.item(), 4),
                    }
                    for idx, p in zip(top3_i, top3_p)
                ],
                'is_uncertain': is_uncertain,
            }

    # ------------------------------------------------------------------
    # Attention visualization
    # ------------------------------------------------------------------

    def get_attention_visualization(self, token_ids: torch.Tensor, tokenizer) -> list:
        """Return per-head attention weights mapped to surface tokens."""
        self.eval()
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        tokens = [tokenizer.id2word.get(idx.item(), '<UNK>') for idx in token_ids[0]]

        with torch.no_grad():
            _, _, _, head_weights = self._encode(token_ids)
            return [
                {
                    'head':    i,
                    'tokens':  tokens,
                    'weights': weights[0, :, 0].cpu().numpy().tolist(),
                }
                for i, weights in enumerate(head_weights)
            ]

    # ------------------------------------------------------------------
    # Dynamic dropout (used during curriculum training)
    # ------------------------------------------------------------------

    def set_dropout(self, dropout_rate: float):
        """Adjust dropout rate on all Dropout layers — useful for fine-tuning phases."""
        self.dropout_rate = dropout_rate
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate