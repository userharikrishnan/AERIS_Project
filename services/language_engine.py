import torch
import json
import os
from models.slm import AerisSLM
from models.tokenizer import Tokenizer

CHECKPOINT_MODEL      = "checkpoints/aeris_slm.pt"
CHECKPOINT_MODEL_BEST = "checkpoints/aeris_slm_best.pt"
CHECKPOINT_TOKENIZER  = "checkpoints/tokenizer.json"


class LanguageEngine:
    def __init__(self, model: AerisSLM, tokenizer: Tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self._load_checkpoint()

    def _load_checkpoint(self):
        # -------------------------
        # Load tokenizer vocab
        # -------------------------
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
            print(f"[LanguageEngine] Tokenizer loaded: {len(word2id)} words")
        else:
            print("[LanguageEngine] WARNING: No tokenizer checkpoint found")

        # -------------------------
        # Load best model if available, else fall back to final
        # -------------------------
        checkpoint_to_load = None
        if os.path.exists(CHECKPOINT_MODEL_BEST):
            checkpoint_to_load = CHECKPOINT_MODEL_BEST
        elif os.path.exists(CHECKPOINT_MODEL):
            checkpoint_to_load = CHECKPOINT_MODEL

        if checkpoint_to_load:
            vocab_size = len(self.tokenizer.word2id)
            self.model = AerisSLM(
                vocab_size=vocab_size,
                embed_dim=128,
                hidden_dim=256
            )
            self.model.load_state_dict(
                torch.load(checkpoint_to_load, map_location="cpu")
            )
            self.model.eval()
            print(f"[LanguageEngine] Model loaded from {checkpoint_to_load}")
        else:
            print("[LanguageEngine] WARNING: No model checkpoint found, using random weights")

    def _sample_token(self, logits, top_k=20, top_p=0.9):
        probs = torch.softmax(logits, dim=-1)

        # Top-k
        topk_probs, topk_indices = torch.topk(probs, top_k)

        # Sort for nucleus
        sorted_probs, sorted_indices = torch.sort(topk_probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # Nucleus filter
        mask = cumulative <= top_p
        mask[..., 0] = True  # ensure at least 1 token

        filtered_probs = sorted_probs[mask]
        filtered_indices = sorted_indices[mask]

        filtered_probs = filtered_probs / filtered_probs.sum()

        return filtered_indices[torch.multinomial(filtered_probs, 1)]

    def _template_response(self, intent_type):
        """
        Fallback templates for low-confidence scenarios.
        """
        templates = {
            "OPEN_APP": "Opening application.",
            "WEB_SEARCH": "Searching the web.",
            "WEB_NAVIGATE": "Navigating to website.",
            "FILE_READ": "Reading file.",
            "FILE_WRITE": "Writing file.",
        }
        return templates.get(intent_type, "Processing request.")

    def generate_from_text(
        self,
        original_text: str,
        intent_type: str = "",
        confidence: float = 1.0,
        max_tokens: int = 12
    ):
        """
        Primary generation method.
        Seeds the SLM with structured prompt for better context anchoring.
        Includes top-k sampling, EOS stopping, repetition penalty, and sanity filtering.
        """
        if not original_text:
            return "Ready."

        # Fallback to template for low confidence
        if confidence < 0.4 and intent_type:
            return self._template_response(intent_type)

        # Add structured prompt with intent context
        prompt = f"[INTENT:{intent_type}] {original_text.lower()}" if intent_type else original_text.lower()
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            return "Ready."

        generated   = list(token_ids)
        seen_tokens = []

        # Stop token ids — never generate these
        stop_ids = {
            self.tokenizer.word2id.get("<PAD>", 0),
            self.tokenizer.word2id.get("<UNK>", 1),
        }
        
        # Get EOS token ID for stopping
        eos_id = self.tokenizer.word2id.get("<EOS>")

        self.model.eval()

        with torch.no_grad():
            for _ in range(max_tokens):
                input_tensor = torch.tensor([generated])
                logits, _    = self.model(input_tensor)

                # Temperature scales with confidence
                # High confidence → lower temp → more decisive output
                temperature = max(0.5, 1.0 - confidence * 0.5)
                last_logits = logits[0, -1, :] / temperature

                # Suppress stop tokens
                for sid in stop_ids:
                    last_logits[sid] = -1e9

                # Stronger repetition penalty — subtract from last 6 tokens
                for prev in set(seen_tokens[-6:]):
                    last_logits[prev] -= 2.0

                # Top-k + nucleus sampling
                next_token = self._sample_token(last_logits).item()

                # EOS stopping
                if eos_id is not None and next_token == eos_id:
                    break

                # Hard stop if token appears 2+ times already (loop detection)
                if seen_tokens.count(next_token) >= 2:
                    break

                generated.append(next_token)
                seen_tokens.append(next_token)

        new_tokens  = generated[len(token_ids):]
        output_text = self.tokenizer.decode(new_tokens)
        
        # Sanity filter: remove garbage repetition
        output_text = output_text.strip()
        if output_text:
            words = output_text.split()
            if len(set(words)) < len(words) * 0.5:
                return f"Executing {intent_type.lower()} action." if intent_type else "Ready."
        
        return output_text if output_text else "Ready."

    def generate_from_plan(self, plan, confidence=None, verified=None, max_tokens=12):
        """
        Legacy compatibility method.
        Converts plan action to natural language seed and delegates to generate_from_text.
        Keeps backward compatibility with any callers that still use the old interface.
        """
        if not plan:
            return "I am not confident enough to proceed."

        # Extract action and convert underscore format to natural language
        action = ""
        if isinstance(plan, list) and plan:
            action = plan[0].get("action", "").replace("_", " ")

        seed_text = action if action else "ready"
        return self.generate_from_text(
            original_text=seed_text,
            intent_type=action.replace(" ", "_").upper() if action else "",
            confidence=confidence or 0.7,
            max_tokens=max_tokens
        )