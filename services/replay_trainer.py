import torch
from models.slm import AerisSLM
from models.tokenizer import Tokenizer
from models.trainer import SLMTrainer
from services.experience_buffer import ExperienceBuffer

class ReplayTrainer:
    def __init__(self, model: AerisSLM, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = SLMTrainer(model)
        self.buffer = ExperienceBuffer()

    def replay(self, batch_size=16):
        experiences = self.buffer.fetch_all(limit=batch_size)
        if not experiences:
            return {"status": "no data"}

        losses = []

        for input_text, response in experiences:
            self.tokenizer.train([input_text, response])

            input_ids = self.tokenizer.encode(input_text)
            target_ids = self.tokenizer.encode(response)

            if not input_ids or not target_ids:
                continue

            inputs = torch.tensor([input_ids])
            targets = torch.tensor([target_ids])

            loss = self.trainer.train_step(inputs, targets)
            losses.append(loss)

        return {
            "trained_on": len(losses),
            "avg_loss": sum(losses) / len(losses) if losses else None
        }

    def conditional_replay(self, reflection_decision):
        if reflection_decision.retrain:
            return self.replay()
        return {"status": "no retraining needed"}