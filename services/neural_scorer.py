import torch
import torch.nn as nn
import json
import os
from models.tokenizer import Tokenizer
from models.plan_scorer_model import PlanScoringNet

CHECKPOINT_SCORER    = "checkpoints/plan_scorer.pt"
CHECKPOINT_TOKENIZER = "checkpoints/tokenizer.json"


class NeuralScorer:
    """
    ML-based plan confidence scorer.
    Encodes the action string through the shared tokenizer + trained GRU
    and predicts a confidence score 0.0 - 1.0.

    Bootstrapped on synthetic data during train.py.
    Continuously improves via experience replay as AERIS executes real actions.
    """

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model     = None
        self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(CHECKPOINT_TOKENIZER):
            with open(CHECKPOINT_TOKENIZER, "r") as f:
                data = json.load(f)

                # Handle structured tokenizer
                if "word2id" in data:
                    word2id = data["word2id"]
                else:
                    word2id = data
                    
            clean_word2id = {}

            for k, v in word2id.items():
                if isinstance(v, int):
                    clean_word2id[k] = v
                elif isinstance(v, str):
                    try:
                        clean_word2id[k] = int(v)
                    except:
                        continue
                elif isinstance(v, dict) and "id" in v:
                    clean_word2id[k] = int(v["id"])

            self.tokenizer.word2id = clean_word2id
            self.tokenizer.id2word = {v: k for k, v in clean_word2id.items()}
            vocab_size = len(word2id)
        else:
            print("[NeuralScorer] WARNING: No tokenizer checkpoint")
            vocab_size = 500

        if os.path.exists(CHECKPOINT_SCORER):
            self.model = PlanScoringNet(vocab_size=vocab_size, embed_dim=128, hidden_dim=128)
            self.model.load_state_dict(
                torch.load(CHECKPOINT_SCORER, map_location="cpu")
            )
            self.model.eval()
            print(f"[NeuralScorer] Plan scorer loaded from {CHECKPOINT_SCORER}")
        else:
            print("[NeuralScorer] WARNING: No scorer checkpoint — confidence will default to 0.75 until trained")

    def score(self, plan) -> float:
        """
        Scores a plan by encoding each action through the ML model.
        Returns average confidence across all steps.
        """
        if not plan:
            return 0.0

        if self.model is None:
            # No checkpoint yet — return safe default so pipeline runs
            return 0.75

        scores = []
        for step in plan:
            action = step.get("action", "")
            if not action:
                continue

            # Convert action to natural language for tokenizer
            # e.g. "open_app" → "open app"
            action_text = action.replace("_", " ")
            token_ids   = self.tokenizer.encode(action_text)

            if not token_ids:
                scores.append(0.75)
                continue

            input_tensor = torch.tensor([token_ids])
            confidence   = self.model.predict_confidence(input_tensor)
            scores.append(confidence)

        if not scores:
            return 0.75

        return round(sum(scores) / len(scores), 3)

    def update_from_experience(self, action: str, success: bool):
        """
        Online learning hook — called after each real execution.
        Updates the scorer based on actual outcome.
        Plugs into the experience replay system.
        """
        if self.model is None:
            return

        action_text = action.replace("_", " ")
        token_ids   = self.tokenizer.encode(action_text)
        if not token_ids:
            return

        target     = torch.tensor([[1.0 if success else 0.0]])
        input_t    = torch.tensor([token_ids])

        optimizer  = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        loss_fn    = nn.BCELoss()

        self.model.train()
        optimizer.zero_grad()
        pred  = self.model(input_t)
        loss  = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        self.model.eval()

        # Persist updated weights immediately
        torch.save(self.model.state_dict(), CHECKPOINT_SCORER)