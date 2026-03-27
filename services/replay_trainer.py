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

        # FIX 6: Add adaptive training - prevent overfitting on tiny samples
        if len(experiences) < 5:
            return {"status": "insufficient data"}

        # FIX 5: Batch training - collect all data first
        inputs_batch = []
        targets_batch = []

        for input_text, response in experiences:
            # FIX 4: REMOVE tokenizer retraining - tokenizer remains stable
            input_ids = self.tokenizer.encode(input_text)
            target_ids = self.tokenizer.encode(response)

            if input_ids and target_ids:
                inputs_batch.append(input_ids)
                targets_batch.append(target_ids)

        if not inputs_batch:
            return {"status": "no valid data"}

        # Pad sequences for batch processing
        inputs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in inputs_batch],
            batch_first=True
        )

        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in targets_batch],
            batch_first=True
        )

        # Train on entire batch
        loss = self.trainer.train_step(inputs, targets)

        # FIX 7: Save model checkpoint after training
        torch.save(self.model.state_dict(), "checkpoints/aeris_slm_replay.pt")

        return {
            "trained_on": len(inputs_batch),
            "avg_loss": loss
        }

    def conditional_replay(self, reflection_decision):
        if reflection_decision.retrain:
            return self.replay()
        return {"status": "no retraining needed"}