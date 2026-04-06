"""
AERIS Language Engine — v2.0
Upgraded for Transformer SLM trained on 15k+ pairs.

Key fixes from v1:
  1. Transformer loaded FIRST (not GRU) — old priority was backwards
  2. Template fallback threshold lowered: 0.35 → 0.25
  3. top_k: 20 → 50, top_p: 0.9 → 0.92 for richer sampling
  4. N-gram repetition blocker (2-gram and 3-gram) applied at inference
  5. Minimum response length enforcement (retry with lower temp if < 4 tokens)
  6. Context-seeded prompt now includes session history (up to 2 turns)
  7. Post-processor: strip leading punct, fix capitalization, remove tokens
  8. SLM model dimensions match new AerisSLM defaults (embed=384, layers=6)
"""

import re
import torch
import json
import os
from typing import Optional, List

from models.slm import AerisSLM, AerisGRUSLM
from models.tokenizer import Tokenizer

CHECKPOINT_MODEL      = 'checkpoints/aeris_slm.pt'
CHECKPOINT_MODEL_BEST = 'checkpoints/aeris_slm_best.pt'
CHECKPOINT_TOKENIZER  = 'checkpoints/tokenizer.json'

# =========================================================
# Jarvis-style template responses
# Used ONLY when model is unavailable or confidence < 0.25
# =========================================================
INTENT_RESPONSES = {
    'OPEN_APP':        'Opening {app} for you now.',
    'CLOSE_APP':       'Closing {app}.',
    'WEB_SEARCH':      "Searching the web for '{query}'.",
    'WEB_NAVIGATE':    'Navigating to {url}.',
    'WEB_SCRAPE':      'Extracting content from {url}. Stand by.',
    'FILE_READ':       'Reading {filename}.',
    'FILE_WRITE':      'Writing to {filename}.',
    'FILE_DELETE':     'Deleting {filename}. Confirmed — proceeding.',
    'FILE_LIST':       'Listing directory contents.',
    'MEMORY_STORE':    "Understood. I've stored that in memory.",
    'MEMORY_RECALL':   'Let me check my memory for that.',
    'MEMORY_FORGET':   "Done. I've cleared that from memory.",
    'GOAL_CREATE':     "Goal created. I'll track that for you.",
    'GOAL_LIST':       'Here are your active goals.',
    'GOAL_PAUSE':      'Goal paused. I will resume when you say so.',
    'GOAL_RESUME':     'Resuming goal. On it.',
    'GOAL_COMPLETE':   'Marking that goal as complete. Well done.',
    'GENERATE_REPORT': 'Generating your report now. One moment.',
    'SCREENSHOT':      'Capturing your screen now.',
    'SYSTEM_INFO':     'Pulling up your system diagnostics.',
    'LIST_WINDOWS':    'Listing all open windows.',
    'ACTIVE_WINDOW':   'Checking the active window.',
    'READ_SCREEN':     'Reading screen content.',
    'VISION_QUERY':    'Analyzing the current display.',
    'REASONING':       'Let me think through that for you.',
    'IDENTITY_QUERY': (
        'I am AERIS — Autonomous Execution and Reasoning Intelligence System. '
        'I can open applications, search and navigate the web, scrape sites, '
        'generate reports, take screenshots, manage your files, remember information, '
        'track your goals, and execute complex multi-step tasks — all on your command. '
        'How may I assist you?'
    ),
    'ROLLBACK':  'Rolling back the last action.',
    'CONFIRM':   'Confirmed. Proceeding.',
    'CANCEL':    'Understood. Action cancelled.',
    'CHAT':      'Of course. How can I help you today?',
}

# Tokens that are never valid response starters
_BAD_STARTERS = re.compile(r'^[\.\,\!\?\;\:\)\]\}\"\']+')


# =========================================================
# Language Engine
# =========================================================

class LanguageEngine:
    """
    Primary response generation layer for AERIS.

    Loads the best available Transformer SLM checkpoint and uses
    nucleus sampling for natural, non-repetitive, non-robotic responses.
    Falls back to Jarvis-style templates only when confidence is very low.
    """

    # SLM architecture defaults — must match train.py
    _SLM_DEFAULTS = dict(
        embed_dim  = 384,
        num_layers = 6,
        num_heads  = 6,
        ffn_dim    = 1536,
        dropout    = 0.0,       # no dropout during inference
        max_len    = 512,
    )

    def __init__(self, model: AerisSLM, tokenizer: Tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self._model_type = 'unloaded'
        self._load_checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self):
        """
        Load tokenizer then model.
        Priority: Transformer (new) → GRU (legacy) → template-only.
        """
        # 1. Tokenizer
        if os.path.exists(CHECKPOINT_TOKENIZER):
            try:
                self.tokenizer.load(CHECKPOINT_TOKENIZER)
                print(f'[LanguageEngine] ✓ Tokenizer: {self.tokenizer.vocab_size:,} tokens')
            except Exception as e:
                print(f'[LanguageEngine] ⚠ Tokenizer load failed: {e}')
        else:
            print('[LanguageEngine] ⚠ No tokenizer checkpoint found')

        # 2. Model — try best first, then final
        checkpoint_path = None
        if os.path.exists(CHECKPOINT_MODEL_BEST):
            checkpoint_path = CHECKPOINT_MODEL_BEST
        elif os.path.exists(CHECKPOINT_MODEL):
            checkpoint_path = CHECKPOINT_MODEL

        if not checkpoint_path:
            print('[LanguageEngine] ⚠ No model checkpoint — template-only mode')
            self._model_type = 'template_only'
            return

        vocab_size = self.tokenizer.vocab_size or 1000

        # --- Try Transformer first (new format) ---
        try:
            model = AerisSLM(vocab_size=vocab_size, **self._SLM_DEFAULTS)
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            self.model       = model
            self._model_type = 'transformer'
            print(f'[LanguageEngine] ✓ Transformer SLM loaded: {checkpoint_path}')
            return
        except Exception as t_err:
            pass   # fall through to GRU attempt

        # --- Try legacy GRU (older checkpoint format) ---
        try:
            model = AerisGRUSLM(vocab_size=vocab_size, embed_dim=128, hidden_dim=256)
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            self.model       = model
            self._model_type = 'gru'
            print(f'[LanguageEngine] ✓ Legacy GRU loaded: {checkpoint_path}')
            return
        except Exception as g_err:
            pass

        print('[LanguageEngine] ⚠ Could not load any checkpoint — template-only mode')
        self._model_type = 'template_only'

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_token(
        self,
        logits:  torch.Tensor,
        top_k:   int   = 50,
        top_p:   float = 0.92,
    ) -> int:
        """Top-k + nucleus (top-p) sampling from a logits vector (1D or 1×V)."""
        logits = logits.squeeze()
        probs  = torch.softmax(logits, dim=-1)

        # Top-k
        topk_p, topk_i = torch.topk(probs, min(top_k, probs.size(-1)))

        # Nucleus filter
        sorted_p, sort_order = torch.sort(topk_p, descending=True)
        cumsum = torch.cumsum(sorted_p, dim=-1)
        keep   = (cumsum - sorted_p) < top_p
        keep[0] = True
        filtered_p = sorted_p[keep]
        filtered_i = topk_i[sort_order][keep]

        filtered_p = filtered_p / filtered_p.sum().clamp(min=1e-9)
        chosen     = torch.multinomial(filtered_p, 1)
        return filtered_i[chosen].item()

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    # Known training data fragments and hallucinations that should NEVER appear in responses
    _BLOCKED_FRAGMENTS = [
        # Original training data leakage
        'compare risk', 'effort and expected', 'before deciding',
        'generation and expected', 'outcome before deciding',
        'risk effort and expected', 'can you tell me more specifically',
        'expected outcome before', 'and expected outcome',
        'compare risk effort',
        # Observed SLM hallucinations from low-confidence generation
        'choose the safer option', 'safer option first', 'optimize if needed',
        'searching the failure', 'searching for the failure',
        'choose the safer', 'then optimize', 'be the safest',
        'the safest option', 'best option first', 'safer option then',
        'failure now', 'look into the failure', 'searching the',
        'cannot determine', 'i am not sure what', 'i am unable to',
        'i cannot perform', 'could not determine', 'i will look into',
        'not confident performing', 'not fully confident',
    ]

    def _postprocess(self, text: str, intent_type: str = '') -> str:
        """
        Clean up raw model output:
          - Remove leading punctuation
          - Capitalise first letter
          - Block training data fragments
          - Limit to 3 sentences max
        """
        text = text.strip()
        if not text:
            return ''

        # Block known training data fragments — fall back to template
        text_lower = text.lower()
        for fragment in self._BLOCKED_FRAGMENTS:
            if fragment in text_lower:
                return ''

        # Remove leading punctuation
        text = _BAD_STARTERS.sub('', text).strip()

        # Capitalise first character
        if text:
            text = text[0].upper() + text[1:]

        # Hard limit: 3 sentences
        sents = re.split(r'(?<=[.!?])\s+', text)
        text  = ' '.join(sents[:3])

        return text.strip()


    # ------------------------------------------------------------------
    # N-gram checker (inline — avoids model rebuild roundtrip)
    # ------------------------------------------------------------------

    @staticmethod
    def _ngram_blocked(seen_tokens: List[int], next_id: int, ngram_size: int = 3) -> bool:
        """Return True if adding next_id would create a repeated n-gram."""
        if ngram_size <= 1 or len(seen_tokens) < ngram_size - 1:
            return False
        suffix = tuple(seen_tokens[-(ngram_size - 1):])
        n = ngram_size - 1
        for i in range(len(seen_tokens) - n):
            if tuple(seen_tokens[i: i + n]) == suffix:
                if i + n < len(seen_tokens) and seen_tokens[i + n] == next_id:
                    return True
        return False

    # ------------------------------------------------------------------
    # Template fallback
    # ------------------------------------------------------------------

    def _template_response(self, intent_type: str, entities: dict = None) -> str:
        """Jarvis-style template with entity interpolation."""
        template = INTENT_RESPONSES.get(intent_type, 'I am ready. How can I help?')
        if entities:
            try:
                fill = {k: v for k, v in entities.items() if not k.startswith('_')}
                return template.format(**fill)
            except (KeyError, IndexError):
                pass
        return template

    # ------------------------------------------------------------------
    # Primary generation
    # ------------------------------------------------------------------

    def generate_from_text(
        self,
        original_text: str,
        intent_type:   str   = '',
        confidence:    float = 1.0,
        max_tokens:    int   = 80,
        entities:      dict  = None,
        history:       list  = None,   # list of (user_str, aeris_str) tuples
    ) -> str:
        """
        Primary response generation.

        Seeds the Transformer SLM with a structured prompt and uses
        nucleus sampling with n-gram repetition blocking.

        Falls back to templates only if:
          - model is unavailable (template_only mode), OR
          - confidence < 0.25 AND intent_type given (very low confidence)
        """
        if not original_text:
            return 'Ready.'

        # Template-only mode (SLM not loaded)
        if self._model_type == 'template_only':
            return self._template_response(intent_type, entities)

        # Low confidence → SLM output is unreliable, use template
        # Threshold kept at 0.50 so rule-based hits (conf=0.97) always use SLM.
        # Only truly uncertain commands fall back to templates.
        if confidence < 0.50 and intent_type:
            return self._template_response(intent_type, entities)

        # --- Build an entity-aware, action-contextual prompt ---
        # Give the SLM enough signal to produce a Jarvis-style acknowledgement
        # (e.g. "Opening Chrome right away." / "Searching for Python tutorials.")
        prompt_parts = []

        # Inject last 2 conversation turns for continuity
        if history:
            for user_turn, aeris_turn in history[-2:]:
                prompt_parts.append(
                    f'<USER> {user_turn.lower()} <ASSISTANT> {aeris_turn.lower()} <EOS>'
                )

        # Build an action-enriched seed: embed entity values directly so the
        # SLM can weave them naturally into its response.
        action_seed = original_text.lower()

        if intent_type and entities:
            # Strip internal metadata keys
            clean_ents = {
                k: v for k, v in entities.items()
                if not k.startswith('_') and k != 'text'
                and isinstance(v, (str, int, float))
            }
            if clean_ents:
                ent_str = ', '.join(f'{k}={v}' for k, v in clean_ents.items())
                action_seed = f'[INTENT:{intent_type}] [{ent_str}] {original_text.lower()}'
            else:
                action_seed = f'[INTENT:{intent_type}] {original_text.lower()}'
        elif intent_type:
            action_seed = f'[INTENT:{intent_type}] {original_text.lower()}'

        prompt_parts.append(f'<USER> {action_seed} <ASSISTANT>')
        prompt = ' '.join(prompt_parts)
        token_ids = self.tokenizer.encode(prompt)

        if not token_ids:
            return self._template_response(intent_type, entities)

        # --- Token IDs we never want in output ---
        pad_id   = self.tokenizer.word2id.get('<PAD>', 0)
        unk_id   = self.tokenizer.word2id.get('<UNK>', 1)
        eos_id   = self.tokenizer.word2id.get('<EOS>', 7)
        stop_ids = {pad_id, unk_id}

        # Temperature strategy:
        #   - High confidence ACTION commands  → 0.55 (focused, decisive)
        #   - CHAT / REASONING                 → 0.75 (more expressive)
        #   - Medium confidence                → interpolated
        CHAT_INTENTS = {'CHAT', 'REASONING', 'IDENTITY_QUERY', ''}
        if intent_type in CHAT_INTENTS:
            temperature = 0.75
        else:
            # Clamp: 0.55 at conf=1.0, 0.70 at conf=0.50
            temperature = max(0.55, min(0.70, 1.05 - confidence * 0.50))

        generated   = list(token_ids)
        seen_tokens: List[int] = []

        self.model.eval()

        with torch.no_grad():
            for step in range(max_tokens):
                input_tensor = torch.tensor([generated])
                logits, _    = self.model(input_tensor)
                last_logits  = logits[0, -1, :].clone() / temperature

                # Suppress stop tokens absolutely
                for sid in stop_ids:
                    last_logits[sid] = -1e9

                # Token-level repetition penalty (last 8 tokens)
                for prev in set(seen_tokens[-8:]):
                    last_logits[prev] -= 2.5

                # N-gram blocking — suppress token if it would repeat a 3-gram
                if len(seen_tokens) >= 2:
                    for cand_id in range(last_logits.size(0)):
                        if self._ngram_blocked(seen_tokens, cand_id, ngram_size=3):
                            last_logits[cand_id] = -1e9

                # Sample
                next_token = self._sample_token(last_logits, top_k=50, top_p=0.92)

                # EOS → stop
                if eos_id is not None and next_token == eos_id:
                    break

                # Hard loop detection: token appears 3+ times → stop
                if seen_tokens.count(next_token) >= 3:
                    break

                generated.append(next_token)
                seen_tokens.append(next_token)

        new_tokens  = generated[len(token_ids):]
        output_text = self.tokenizer.decode(new_tokens)
        output_text = self._postprocess(output_text, intent_type)

        # Sanity gate: if output is mostly repeated tokens → fallback
        if output_text:
            words = output_text.split()
            if len(words) >= 4 and len(set(words)) < len(words) * 0.45:
                return self._template_response(intent_type, entities)

        # Minimum meaningful response (retry with lower temp if too short)
        if len(output_text.split()) < 3:
            if confidence >= 0.5:
                # Second attempt: greedy at very low temperature
                return self._retry_greedy(token_ids, intent_type, entities, max_tokens)
            return self._template_response(intent_type, entities)

        return output_text if output_text else self._template_response(intent_type, entities)

    def _retry_greedy(
        self,
        token_ids:   list,
        intent_type: str,
        entities:    dict,
        max_tokens:  int,
    ) -> str:
        """
        Second-chance greedy generation at temperature=0.3.
        Called when nucleus sampling produced a too-short response.
        """
        generated   = list(token_ids)
        seen_tokens: List[int] = []
        eos_id  = self.tokenizer.word2id.get('<EOS>', 7)
        pad_id  = self.tokenizer.word2id.get('<PAD>', 0)
        unk_id  = self.tokenizer.word2id.get('<UNK>', 1)

        with torch.no_grad():
            for _ in range(max_tokens):
                input_tensor = torch.tensor([generated])
                logits, _    = self.model(input_tensor)
                last_logits  = logits[0, -1, :].clone() / 0.3

                last_logits[pad_id] = -1e9
                last_logits[unk_id] = -1e9

                for prev in set(seen_tokens[-5:]):
                    last_logits[prev] -= 3.0

                next_token = torch.argmax(last_logits).item()

                if next_token == eos_id or seen_tokens.count(next_token) >= 2:
                    break

                generated.append(next_token)
                seen_tokens.append(next_token)

        new_tokens  = generated[len(token_ids):]
        output_text = self.tokenizer.decode(new_tokens)
        output_text = self._postprocess(output_text, intent_type)

        if len(output_text.split()) < 3:
            return self._template_response(intent_type, entities)

        return output_text

    # ------------------------------------------------------------------
    # Legacy compat
    # ------------------------------------------------------------------

    def generate_from_plan(self, plan, confidence=None, verified=None, max_tokens=80):
        """
        Legacy compatibility bridge.
        Converts plan action → natural language seed → delegates to generate_from_text.
        """
        if not plan:
            return 'I am not confident enough to proceed.'

        action = ''
        if isinstance(plan, list) and plan:
            action = plan[0].get('action', '').replace('_', ' ')

        seed_text = action if action else 'ready'
        return self.generate_from_text(
            original_text=seed_text,
            intent_type=action.replace(' ', '_').upper() if action else '',
            confidence=confidence or 0.7,
            max_tokens=max_tokens,
        )