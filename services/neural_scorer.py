import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanScoringNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class NeuralScorer:
    def __init__(self):
        self.model = PlanScoringNet()
        self.model.eval()

    def _encode_plan(self, plan):
        """
        Very simple encoding for now.
        Later: embeddings / transformers.
        """
        length = len(plan)
        action_hash = sum(hash(step["action"]) % 100 for step in plan)
        return torch.tensor([[length, action_hash] + [0]*8], dtype=torch.float32)

    def score(self, plan):
        if not plan:
            return 0.0

        encoded = self._encode_plan(plan)
        with torch.no_grad():
            confidence = self.model(encoded).item()

        return round(confidence, 3)
