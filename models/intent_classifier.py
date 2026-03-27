import torch
import torch.nn as nn
from models.embeddings import EmbeddingLayer

# =========================================================
# Intent Classification Constants - SYNCHRONIZED WITH TRAINING DATA
# =========================================================

INTENT_TO_IDX = {
    # Application Control
    "OPEN_APP": 0,
    "CLOSE_APP": 1,
    
    # Web Operations
    "WEB_SEARCH": 2,
    "WEB_NAVIGATE": 3,
    
    # File System
    "FILE_READ": 4,
    "FILE_WRITE": 5,
    "FILE_DELETE": 6,
    "FILE_LIST": 7,
    
    # Memory Operations
    "MEMORY_STORE": 8,
    "MEMORY_RECALL": 9,
    "MEMORY_FORGET": 10,
    
    # Goal Management
    "GOAL_CREATE": 11,
    "GOAL_LIST": 12,
    "GOAL_PAUSE": 13,
    "GOAL_RESUME": 14,
    "GOAL_COMPLETE": 15,
    
    # Vision/Screen Operations
    "VISION_QUERY": 16,
    "READ_SCREEN": 17,
    "ACTIVE_WINDOW": 18,
    "LIST_WINDOWS": 19,
    
    # System Control
    "ROLLBACK": 20,
    "CONFIRM": 21,
    "CANCEL": 22,
    
    # Cognitive
    "REASONING": 23,
    "IDENTITY_QUERY": 24,
    
    # Fallback
    "CHAT": 25,
    "UNKNOWN": 26
}

NUM_INTENT_CLASSES = len(INTENT_TO_IDX)
IDX_TO_INTENT = {v: k for k, v in INTENT_TO_IDX.items()}


class IntentClassifier(nn.Module):
    """
    Production-grade Neural Intent Classifier for AERIS
    Architecture: Embedding → BiGRU → Multi-Head Attention → Residual Classifier
    
    Features:
    - LayerNorm for training stability
    - Multi-head attention (4 heads) for diverse feature extraction
    - Residual connections for gradient flow
    - Uncertainty estimation for production confidence calibration
    - Attention visualization capability
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 128, 
        hidden_dim: int = 256,
        num_classes: int = NUM_INTENT_CLASSES,
        num_heads: int = 4,
        dropout: float = 0.4,
        num_gru_layers: int = 2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        # Input embedding with normalization
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.embed_dropout = nn.Dropout(dropout / 2)
        
        # Bidirectional GRU encoder
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=True
        )
        
        # GRU output normalization
        gru_output_dim = hidden_dim * 2  # bidirectional
        self.gru_norm = nn.LayerNorm(gru_output_dim)
        
        # Multi-head self-attention for intent-relevant word focus
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gru_output_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            ) for _ in range(num_heads)
        ])
        
        # Attention fusion - learns to weight different heads
        self.attention_fusion = nn.Sequential(
            nn.Linear(num_heads, num_heads),
            nn.ReLU(),
            nn.Linear(num_heads, 1)
        )
        
        # Context projection with residual
        self.context_proj = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim),
            nn.LayerNorm(gru_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Deep classifier with residual connections
        classifier_dim = gru_output_dim
        
        # First classifier block
        self.classifier_block1 = nn.Sequential(
            nn.Linear(classifier_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Second classifier block
        self.classifier_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim // 2, num_classes)
        
        # Skip connection projection (if dims don't match)
        self.skip_proj = nn.Linear(classifier_dim, hidden_dim // 2) if classifier_dim != hidden_dim // 2 else nn.Identity()
        
        # Uncertainty estimation head for production confidence calibration
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for stable training"""
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
        
    def forward(self, token_ids: torch.Tensor, return_attention: bool = False):
        """
        Forward pass with optional attention visualization
        
        Args:
            token_ids: (batch_size, seq_len) token indices
            return_attention: If True, return attention weights for visualization
            
        Returns:
            logits: (batch_size, num_classes) class logits
            attention_weights: (optional) List of attention weight tensors
        """
        # Embed and normalize: (B, T, E)
        embedded = self.embedding(token_ids)
        embedded = self.embed_norm(embedded)
        embedded = self.embed_dropout(embedded)
        
        # Encode sequence: (B, T, H*2)
        gru_out, _ = self.gru(embedded)
        gru_out = self.gru_norm(gru_out)
        
        # Multi-head attention: each head focuses on different aspects
        head_contexts = []
        head_weights = []
        
        for head in self.attention_heads:
            weights = head(gru_out)  # (B, T, 1)
            context = torch.sum(gru_out * weights, dim=1)  # (B, H*2)
            head_contexts.append(context)
            head_weights.append(weights)
        
        # Stack heads: (B, H*2, num_heads)
        stacked = torch.stack(head_contexts, dim=-1)
        
        # Fuse heads with learned weights
        fusion_weights = torch.softmax(self.attention_fusion(stacked), dim=-1)
        context = torch.sum(stacked * fusion_weights, dim=-1)  # (B, H*2)
        
        # Residual projection
        context = self.context_proj(context) + context  # Residual
        
        # Classifier with residual
        x = self.classifier_block1(context)
        x_res = self.classifier_block2(x)
        x_skip = self.skip_proj(context)
        x = x_res + x_skip  # Residual connection
        
        # Output logits
        logits = self.output_proj(x)  # (B, num_classes)
        
        if return_attention:
            return logits, head_weights
        return logits
    
    def predict(self, token_ids: torch.Tensor, confidence_threshold: float = 0.6) -> str:
        """
        Production prediction with calibrated confidence thresholding
        
        Uses margin-based uncertainty (difference between top 2 predictions)
        for more robust decision making.
        """
        self.eval()
        
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(token_ids)
            # FIX 1: Temperature scaling for calibrated probabilities
            temperature = 1.4
            probs = torch.softmax(logits / temperature, dim=1)
            
            # Get top-2 for uncertainty analysis
            top2_probs, top2_indices = torch.topk(probs, 2, dim=1)
            predicted_idx = top2_indices[0][0].item()
            confidence = top2_probs[0][0].item()
            second_confidence = top2_probs[0][1].item()
            
            # Margin-based uncertainty (difference between top 2)
            margin = confidence - second_confidence
            
            # FIX 2: Smart chat fallback with second intent check
            intent = IDX_TO_INTENT.get(predicted_idx, "UNKNOWN")
            
            # Smart fallback logic - check if second best is CHAT and close
            if intent != "CHAT":
                second_intent = IDX_TO_INTENT.get(top2_indices[0][1].item(), "UNKNOWN")
                if second_intent == "CHAT" and margin < 0.25:
                    return "CHAT"
            
            # Low confidence fallback
            if confidence < confidence_threshold:
                return "CHAT"
            
            # Check for UNKNOWN class - also fallback to CHAT
            if intent == "UNKNOWN":
                return "CHAT"
                
        return intent
    
    def predict_with_confidence(self, token_ids: torch.Tensor) -> dict:
        """
        Rich prediction output for AERIS decision engine
        
        Returns comprehensive information for downstream decision making:
        - intent: Predicted intent label
        - confidence: Probability of top prediction
        - uncertainty: 1 - confidence (for risk assessment)
        - margin: Gap between top 2 predictions (decisiveness)
        - alternatives: Top 3 predictions with probabilities
        - uncertainty_score: Learned uncertainty from uncertainty head
        """
        self.eval()
        
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(token_ids)
            # FIX 1: Temperature scaling for calibrated probabilities
            temperature = 1.4
            probs = torch.softmax(logits / temperature, dim=1)[0]
            
            # Get top-3 predictions
            top3_probs, top3_indices = torch.topk(probs, 3)
            
            # Calculate margin
            margin = (top3_probs[0] - top3_probs[1]).item()
            
            # Get learned uncertainty
            # Need to run forward partially to get hidden representation
            embedded = self.embedding(token_ids)
            embedded = self.embed_norm(embedded)
            gru_out, _ = self.gru(embedded)
            gru_out = self.gru_norm(gru_out)
            
            # Quick single-head attention for uncertainty
            attn_weights = self.attention_heads[0](gru_out)
            context = torch.sum(gru_out * attn_weights, dim=1)
            context = self.context_proj(context) + context
            x = self.classifier_block1(context)
            x_res = self.classifier_block2(x)
            x_skip = self.skip_proj(context)
            x = x_res + x_skip
            uncertainty_score = self.uncertainty_head(x).item()
            
            # FIX 3: Use uncertainty head in decision logic
            is_uncertain = (
                margin < 0.15
                or top3_probs[0].item() < 0.6
                or uncertainty_score > 0.6
            )
            
            results = {
                'intent': IDX_TO_INTENT.get(top3_indices[0].item(), "UNKNOWN"),
                'confidence': round(top3_probs[0].item(), 4),
                'uncertainty': round(1 - top3_probs[0].item(), 4),
                'margin': round(margin, 4),
                'uncertainty_score': round(uncertainty_score, 4),
                'alternatives': [
                    {
                        'intent': IDX_TO_INTENT.get(idx.item(), "UNKNOWN"),
                        'probability': round(prob.item(), 4)
                    }
                    for idx, prob in zip(top3_indices, top3_probs)
                ],
                'is_uncertain': is_uncertain
            }
            
        return results
    
    def get_attention_visualization(self, token_ids: torch.Tensor, tokenizer):
        """
        Get attention weights for visualization/debugging
        
        Returns attention weights mapped to tokens for interpretability.
        """
        self.eval()
        
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        tokens = [tokenizer.id2word.get(idx.item(), '<UNK>') for idx in token_ids[0]]
        
        with torch.no_grad():
            logits, head_weights = self.forward(token_ids, return_attention=True)
            
            # Convert to numpy for visualization
            attention_data = []
            for i, weights in enumerate(head_weights):
                attn = weights[0, :, 0].cpu().numpy()  # (seq_len,)
                attention_data.append({
                    'head': i,
                    'tokens': tokens,
                    'weights': attn.tolist()
                })
            
            return attention_data
    
    def set_dropout(self, dropout_rate: float):
        """Dynamic dropout adjustment for training phases"""
        self.dropout_rate = dropout_rate
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate