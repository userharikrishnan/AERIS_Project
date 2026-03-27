import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum
        attended = torch.matmul(weights, V)
        return attended, weights
