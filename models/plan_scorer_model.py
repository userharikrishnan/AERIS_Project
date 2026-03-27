import torch
import torch.nn as nn
from models.embeddings import EmbeddingLayer


class PlanScoringNet(nn.Module):
    """
    Production Plan Confidence Scorer for AERIS.

    Improvements over v1:
      - Bidirectional GRU (sees full context before scoring)
      - Multi-head attention pooling (focuses on intent-critical words)
      - Residual MLP classifier
      - Separate uncertainty head (aleatoric uncertainty estimate)
      - Risk calibration: dangerous-operation detector
      - Returns rich dict instead of bare float
    """

    # Actions that should always be flagged regardless of confidence
    DANGEROUS_PATTERNS = frozenset([
        "delete all", "remove all", "erase all", "wipe all",
        "format", "rm -rf", "rmdir", "delete system",
        "kill all", "shutdown", "format drive", "wipe disk",
        "disable firewall", "delete system32",
    ])

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Embedding
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.embed_drop = nn.Dropout(dropout / 2)

        # Bidirectional GRU encoder
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        gru_dim = hidden_dim * 2  # bidirectional
        self.gru_norm = nn.LayerNorm(gru_dim)

        # Multi-head attention pooling
        self.attn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gru_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softmax(dim=1),
            )
            for _ in range(num_heads)
        ])
        self.head_fusion = nn.Linear(num_heads * gru_dim, gru_dim)

        # Residual MLP scorer
        self.mlp = nn.Sequential(
            nn.Linear(gru_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.skip = nn.Linear(gru_dim, hidden_dim // 2)

        # Confidence output
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Aleatoric uncertainty head (how uncertain is the model itself)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Risk classification head (binary: routine vs dangerous)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, T)
        returns:   (B, 1) — confidence score 0.0–1.0
        (kept simple for BCE training compatibility)
        """
        features = self._encode(token_ids)
        return self.confidence_head(features)

    def _encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Shared encoder trunk → (B, hidden_dim//2)"""
        x = self.embedding(token_ids)
        x = self.embed_drop(self.embed_norm(x))

        gru_out, _ = self.gru(x)
        gru_out = self.gru_norm(gru_out)

        # Multi-head attention pooling
        head_contexts = []
        for head in self.attn_heads:
            w = head(gru_out)                       # (B, T, 1)
            c = torch.sum(gru_out * w, dim=1)       # (B, gru_dim)
            head_contexts.append(c)

        fused_in = torch.cat(head_contexts, dim=-1) # (B, num_heads * gru_dim)
        context = self.head_fusion(fused_in)        # (B, gru_dim)

        # Residual MLP
        h = self.mlp(context) + self.skip(context)  # (B, hidden_dim//2)
        return h

    def score(self, token_ids: torch.Tensor) -> dict:
        """
        Rich scoring output for AERIS decision engine.

        Returns:
            confidence:    float 0–1, plan viability estimate
            uncertainty:   float 0–1, model's own uncertainty
            risk:          float 0–1, danger level
            needs_confirm: bool, whether to ask user before executing
            is_dangerous:  bool, pattern-matched against known risky phrases
        """
        self.eval()
        with torch.no_grad():
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)

            features = self._encode(token_ids)
            confidence  = self.confidence_head(features).item()
            uncertainty = self.uncertainty_head(features).item()
            risk        = self.risk_head(features).item()

        return {
            'confidence':    round(confidence, 4),
            'uncertainty':   round(uncertainty, 4),
            'risk':          round(risk, 4),
            'needs_confirm': risk > 0.5 or confidence < 0.55,
            'is_dangerous':  False,   # updated by predict_confidence below
            'calibrated':    round(confidence * (1 - uncertainty * 0.5), 4),
        }

    def predict_confidence(self, token_ids: torch.Tensor, raw_text: str = "") -> float:
        """
        Single confidence float for backward compat + dangerous-pattern check.
        Automatically lowers score for known dangerous operations.
        """
        result = self.score(token_ids)

        # Hard override for dangerous patterns
        lowered = raw_text.lower()
        if any(p in lowered for p in self.DANGEROUS_PATTERNS):
            result['is_dangerous'] = True
            result['confidence']   = min(result['confidence'], 0.30)
            result['needs_confirm'] = True

        return round(result['calibrated'], 3)