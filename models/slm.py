import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embeddings import EmbeddingLayer
from models.attention import SelfAttention

class AerisSLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids):
        x = self.embedding(token_ids)              # (B, T, E)
        rnn_out, _ = self.rnn(x)                   # (B, T, H)

        attended, attn_weights = self.attention(rnn_out)
        logits = self.fc(attended)                 # (B, T, V)

        return logits, attn_weights

    def predict_next(self, token_ids):
        logits, _ = self.forward(token_ids)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        return torch.argmax(probs, dim=-1)
