import torch
from models.slm import AerisSLM
from models.tokenizer import Tokenizer

class LanguageEngine:
    def __init__(self, model: AerisSLM, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_from_plan(self, plan, confidence=None, verified=None):
        """
        Generate language strictly grounded in reasoning plan
        """

        if not plan:
            return "I am not confident enough to proceed."

        # Convert plan to a structured prompt (NOT user text)
        plan_text = self._plan_to_text(plan, confidence, verified)

        token_ids = self.tokenizer.encode(plan_text)
        if not token_ids:
            return "I could not form a response."

        input_tensor = torch.tensor([token_ids])

        next_token = self.model.predict_next(input_tensor)
        output_text = self.tokenizer.decode(next_token.tolist())

        return output_text

    def _plan_to_text(self, plan, confidence, verified):
        steps = []
        for step in plan:
            action = step.get("action")
            params = step.get("params", {})
            steps.append(f"{action} {params}")

        meta = []
        if confidence is not None:
            meta.append(f"confidence {confidence}")
        if verified is not None:
            meta.append(f"verified {verified}")

        return " ; ".join(steps + meta)
