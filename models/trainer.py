import torch
import torch.nn as nn
import torch.optim as optim
import math


class SLMTrainer:
    """
    Production SLM Trainer for AERIS.
    Improvements over v1:
      - Gradient clipping (prevents exploding gradients in GRU)
      - Label smoothing (reduces overconfidence, better generalisation)
      - Perplexity tracking (more meaningful than raw CE loss)
      - Ignore padding tokens in loss (idx 0)
      - Warmup scheduler support
      - Train/eval mode management
    """

    def __init__(
        self,
        model,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 200,
    ):
        self.model = model
        self.max_grad_norm = max_grad_norm
        self._step = 0
        self._warmup_steps = warmup_steps

        # CrossEntropyLoss with label smoothing and padding ignore
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=0,          # ignore PAD token
            label_smoothing=label_smoothing,
        )

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Cosine annealing; caller can also attach an external scheduler
        # and call scheduler.step() themselves — this is a fallback.
        self._base_lr = lr

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Single optimisation step.

        Args:
            inputs:  (B, T) token IDs — context window
            targets: (B, T) token IDs — next-token labels

        Returns:
            dict with 'loss' (float) and 'perplexity' (float)
        """
        self.model.train()
        self._step += 1

        # Warmup LR override (linear ramp)
        if self._step <= self._warmup_steps:
            warmup_lr = self._base_lr * (self._step / self._warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = warmup_lr

        self.optimizer.zero_grad()

        # Forward — model returns (logits, attn_weights)
        logits, _ = self.model(inputs)          # (B, T, V)

        # Flatten for loss
        V = logits.size(-1)
        loss = self.loss_fn(
            logits.view(-1, V),     # (B*T, V)
            targets.view(-1),       # (B*T,)
        )

        loss.backward()

        # Gradient clipping — critical for GRU stability
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

        self.optimizer.step()

        loss_val = loss.item()
        perplexity = math.exp(min(loss_val, 20))  # cap to avoid inf display

        return {
            'loss': loss_val,
            'perplexity': perplexity,
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            'lr': self.optimizer.param_groups[0]['lr'],
            'step': self._step,
        }

    # ------------------------------------------------------------------
    # Evaluation (no grad, no dropout)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """Evaluate on a batch without updating weights."""
        self.model.eval()
        logits, _ = self.model(inputs)
        V = logits.size(-1)
        loss = self.loss_fn(logits.view(-1, V), targets.view(-1))
        loss_val = loss.item()
        return {
            'loss': loss_val,
            'perplexity': math.exp(min(loss_val, 20)),
        }

    # ------------------------------------------------------------------
    # Generation helper (greedy / nucleus)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """
        Nucleus (top-p) sampling from the SLM.

        Args:
            prompt_ids:     (1, T) seed token IDs
            max_new_tokens: maximum tokens to generate
            temperature:    sampling temperature (lower = more deterministic)
            top_p:          nucleus probability threshold
            eos_id:         end-of-sequence token to stop generation

        Returns:
            (1, T+new) token tensor
        """
        self.model.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            logits, _ = self.model(ids)
            next_logits = logits[:, -1, :] / temperature  # (1, V)

            # Nucleus sampling
            probs = torch.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens above threshold (keep those where cumsum - prob < top_p)
            remove_mask = (cumulative - sorted_probs) >= top_p
            sorted_probs[remove_mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            # Sample
            chosen = torch.multinomial(sorted_probs, 1)  # (1, 1)
            next_token = sorted_idx.gather(-1, chosen)   # (1, 1)

            ids = torch.cat([ids, next_token], dim=1)

            if next_token.item() == eos_id:
                break

        return ids

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, extra: dict = None):
        """Save model + optimizer state for resumable training."""
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'step': self._step,
            'base_lr': self._base_lr,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self._step = state.get('step', 0)
        self._base_lr = state.get('base_lr', self._base_lr)
        return state