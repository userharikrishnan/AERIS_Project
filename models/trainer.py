"""
AERIS Production SLM Trainer — v2.0
Upgraded for 15k+ training pairs.

Improvements over v1:
  - N-gram repetition penalty in generation helper
  - Warmup steps scaled to 500 (from 200)
  - Gradient accumulation support (accumulation_steps param)
  - Perplexity-based early stopping via eval_step
  - Richer logging per training step (loss, ppl, lr, grad_norm)
  - Loss masking — only trains on assistant (non-zero) tokens (preserved)
  - Token-level accuracy metric alongside loss
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional


class SLMTrainer:
    """
    Production SLM Trainer for AERIS v2.0.
    Handles Transformer SLM training with gradient accumulation,
    perplexity tracking, and rich diagnostic logging.
    """

    def __init__(
        self,
        model,
        lr:               float = 0.0005,   # tuned down for larger model
        weight_decay:     float = 0.01,
        label_smoothing:  float = 0.08,     # reduced from 0.1 — larger data needs less smoothing
        max_grad_norm:    float = 1.0,
        warmup_steps:     int   = 500,       # scaled up from 200
        accumulation_steps: int = 1,         # gradient accumulation (simulate larger batches)
    ):
        self.model              = model
        self.max_grad_norm      = max_grad_norm
        self._step              = 0
        self._warmup_steps      = warmup_steps
        self._accumulation_steps = max(1, accumulation_steps)
        self._accum_count       = 0
        self._accum_loss        = 0.0

        # CrossEntropyLoss: ignores pad (idx=0) + label smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=label_smoothing,
            reduction='none',    # element-wise so we can apply mask cleanly
        )

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),   # 0.98 β₂ — standard for Transformer training
            eps=1e-9,
        )

        self._base_lr = lr

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        inputs:  torch.Tensor,
        targets: torch.Tensor,
        verbose: bool = False,
    ) -> dict:
        """
        Single optimisation step with optional gradient accumulation.

        Args:
            inputs:  (B, T) token IDs — context window
            targets: (B, T) token IDs — next-token labels (0 for user tokens)
            verbose: Print per-step diagnostics if True

        Returns:
            dict: loss, perplexity, grad_norm, lr, step, token_accuracy
        """
        self.model.train()
        self._step      += 1
        self._accum_count += 1

        # --- Linear LR warmup ---
        if self._step <= self._warmup_steps:
            warmup_lr = self._base_lr * (self._step / self._warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = warmup_lr

        # --- Forward pass ---
        logits, _ = self.model(inputs)          # (B, T, V)
        V         = logits.size(-1)

        # --- Element-wise CE loss ---
        loss_raw = self.loss_fn(
            logits.view(-1, V),     # (B*T, V)
            targets.view(-1),       # (B*T,)
        )                           # (B*T,) element-wise

        # --- Mask: only train on assistant tokens (target != 0) ---
        mask     = (targets.view(-1) != 0).float()
        n_active = mask.sum().clamp(min=1)
        loss     = (loss_raw * mask).sum() / n_active

        # Scale for accumulation
        loss_scaled = loss / self._accumulation_steps
        loss_scaled.backward()

        loss_val = loss.item()

        # --- Optional: token-level accuracy (non-masked tokens) ---
        with torch.no_grad():
            predicted  = logits.view(-1, V).argmax(dim=-1)   # (B*T,)
            target_tok = targets.view(-1)
            active     = mask.bool()
            if active.any():
                correct = (predicted[active] == target_tok[active]).float().sum()
                token_acc = (correct / active.float().sum()).item()
            else:
                token_acc = 0.0

        # --- Grad step only after accumulation_steps batches ---
        grad_norm = 0.0
        if self._accum_count >= self._accumulation_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accum_count = 0

        perplexity = math.exp(min(loss_val, 20))

        result = {
            'loss':           loss_val,
            'perplexity':     perplexity,
            'token_accuracy': token_acc,
            'grad_norm':      grad_norm,
            'lr':             self.optimizer.param_groups[0]['lr'],
            'step':           self._step,
            'active_tokens':  int(n_active.item()),
        }

        if verbose:
            print(
                f"  [Step {self._step:>6}] "
                f"loss={loss_val:.4f}  ppl={perplexity:.1f}  "
                f"acc={token_acc:.2%}  grad={grad_norm:.3f}  "
                f"lr={result['lr']:.6f}  active_tok={result['active_tokens']}"
            )

        return result

    # ------------------------------------------------------------------
    # Evaluation (no grad, no dropout)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Evaluate on a batch without updating weights.
        Returns loss, perplexity, and token_accuracy for SLM validation.
        """
        self.model.eval()
        logits, _ = self.model(inputs)
        V         = logits.size(-1)

        loss_raw  = self.loss_fn(logits.view(-1, V), targets.view(-1))
        mask      = (targets.view(-1) != 0).float()
        n_active  = mask.sum().clamp(min=1)
        loss      = (loss_raw * mask).sum() / n_active

        loss_val   = loss.item()
        perplexity = math.exp(min(loss_val, 20))

        predicted  = logits.view(-1, V).argmax(dim=-1)
        target_tok = targets.view(-1)
        active     = mask.bool()
        if active.any():
            correct   = (predicted[active] == target_tok[active]).float().sum()
            token_acc = (correct / active.float().sum()).item()
        else:
            token_acc = 0.0

        return {
            'loss':           loss_val,
            'perplexity':     perplexity,
            'token_accuracy': token_acc,
        }

    # ------------------------------------------------------------------
    # Generation helper — nucleus sampling with n-gram penalty
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids:         torch.Tensor,
        max_new_tokens:     int   = 80,
        temperature:        float = 0.75,
        top_p:              float = 0.92,
        top_k:              int   = 50,
        eos_id:             int   = 7,
        repetition_penalty: float = 1.3,
        ngram_block_size:   int   = 3,
    ) -> torch.Tensor:
        """
        Delegate to model.generate() if available (AerisSLM v2),
        otherwise fall back to manual nucleus loop.
        """
        if hasattr(self.model, 'generate'):
            return self.model.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=eos_id,
                repetition_penalty=repetition_penalty,
                ngram_block_size=ngram_block_size,
            )

        # ---- Fallback for older model objects ----
        self.model.eval()
        ids      = prompt_ids.clone()
        new_toks: List[int] = []

        for _ in range(max_new_tokens):
            logits, _ = self.model(ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-7)

            # Suppress pad/unk
            next_logits[0, 0] = -1e9
            next_logits[0, 1] = -1e9

            # Repetition penalty
            if repetition_penalty != 1.0 and new_toks:
                for tid in set(new_toks[-20:]):
                    val = next_logits[0, tid]
                    next_logits[0, tid] = val / repetition_penalty if val > 0 \
                                          else val * repetition_penalty

            # N-gram blocking
            if ngram_block_size > 0 and len(new_toks) >= ngram_block_size - 1:
                suffix = tuple(new_toks[-(ngram_block_size - 1):])
                n      = ngram_block_size - 1
                for i in range(len(new_toks) - n):
                    if tuple(new_toks[i: i + n]) == suffix:
                        next_logits[0, new_toks[i + n]] = -1e9

            # Top-k
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits  = next_logits.masked_fill(next_logits < topk_vals[:, -1:], -1e9)

            # Nucleus
            probs         = torch.softmax(next_logits, dim=-1)
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cumsum        = torch.cumsum(sorted_p, dim=-1)
            remove        = (cumsum - sorted_p) >= top_p
            sorted_p[remove] = 0.0
            sorted_p     /= sorted_p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            chosen_rel    = torch.multinomial(sorted_p, 1)
            next_token    = sorted_idx.gather(-1, chosen_rel)

            token_val = next_token.item()
            ids       = torch.cat([ids, next_token], dim=1)
            new_toks.append(token_val)

            if token_val == eos_id:
                break

        return ids

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, extra: dict = None):
        """Persist model + optimizer for resumable training."""
        state = {
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'step':            self._step,
            'base_lr':         self._base_lr,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        """Resume training from a saved checkpoint."""
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self._step    = state.get('step', 0)
        self._base_lr = state.get('base_lr', self._base_lr)
        return state