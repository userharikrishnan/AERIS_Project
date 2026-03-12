import torch
import torch.nn as nn
import torch.optim as optim

class SLMTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()
