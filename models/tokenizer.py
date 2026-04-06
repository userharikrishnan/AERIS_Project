"""
AERIS Production Tokenizer — v2.0
Upgraded for 15k+ training pairs with 32k vocabulary.

Features:
- Word-level tokenization with regex preprocessing
- Special token handling (<PAD>, <UNK>, <START>, <END>, <USER>, <ASSISTANT>, <EOS>, <MASK>, <SEP>)
- 32,000 vocabulary ceiling to cover 15k+ diverse training pairs
- Frequency-based vocabulary pruning with min_freq support
- Batch encoding with padding/truncation and attention masks
- Serialization / deserialization (JSON)
- __contains__ for fast OOV checking
- encode_with_offsets() for attention visualization
- pad_to_length() and truncate() helpers
- n-gram decode with proper punctuation attachment
"""

import re
import json
from collections import defaultdict
from typing import List, Dict, Optional, Union, Tuple


class Tokenizer:
    """
    Production-grade tokenizer for AERIS SLM — supports 32k vocabulary.
    Designed to handle 15k+ training pairs without OOV explosion.
    """

    SPECIAL_TOKENS = {
        '<PAD>':       0,
        '<UNK>':       1,
        '<START>':     2,
        '<END>':       3,
        '<MASK>':      4,
        '<USER>':      5,
        '<ASSISTANT>': 6,
        '<EOS>':       7,
        '<SEP>':       8,   # separator between multi-turn context
        '<SYS>':       9,   # system prompt token (reserved for future use)
    }

    # Punctuation characters that should attach to the preceding word
    _NO_SPACE_BEFORE = frozenset(".,!?;:')]}\"")

    def __init__(self, max_vocab_size: int = 32000):
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.word_freq: Dict[str, int] = defaultdict(int)
        self.max_vocab_size = max_vocab_size

        # Seed special tokens
        for token, idx in self.SPECIAL_TOKENS.items():
            self.word2id[token] = idx
            self.id2word[idx] = token

        # Regex: special tokens first (angle-bracket form), then word chars, then punctuation
        self._compiled_regex = re.compile(r'<[^>]+>|\w+|[^\w\s]')

    # ------------------------------------------------------------------
    # Vocabulary building
    # ------------------------------------------------------------------

    def train(self, texts: List[str], min_freq: int = 1):
        """
        Build vocabulary from corpus with frequency filtering.

        Args:
            texts:    List of text strings (or individual tokens) to count.
            min_freq: Minimum word frequency to include in the vocabulary.
        """
        for text in texts:
            if not isinstance(text, str):
                continue
            for word in self._tokenize_text(text):
                self.word_freq[word] += 1

        # Sort by frequency descending, then alphabetically for ties (deterministic)
        sorted_words = sorted(
            ((freq, word) for word, freq in self.word_freq.items()),
            key=lambda x: (-x[0], x[1])
        )

        for freq, word in sorted_words:
            if len(self.word2id) >= self.max_vocab_size:
                break
            if freq < min_freq:
                continue
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

        print(
            f"[Tokenizer] Vocabulary built: {len(self.word2id):,} tokens "
            f"(trained on {len(texts):,} texts, min_freq={min_freq})"
        )

    # ------------------------------------------------------------------
    # Core tokenisation
    # ------------------------------------------------------------------

    def _tokenize_text(self, text: str) -> List[str]:
        """Internal regex tokenizer. Preserves special <TOKEN> forms unchanged."""
        text = text.strip()
        tokens = self._compiled_regex.findall(text)
        return [t for t in tokens if t.strip()]

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text:               String or list of strings.
            add_special_tokens: Prepend <START> and append <END>.
            max_length:         Truncate result to this length.

        Returns:
            List of integer token IDs.
        """
        if isinstance(text, list):
            text = ' '.join(str(t) for t in text)

        tokens = self._tokenize_text(text)

        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.SPECIAL_TOKENS['<START>'])

        unk_id = self.SPECIAL_TOKENS['<UNK>']
        for token in tokens:
            ids.append(self.word2id.get(token, unk_id))

        if add_special_tokens:
            ids.append(self.SPECIAL_TOKENS['<END>'])

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[str]]:
        """
        Encode text and return both IDs and the corresponding surface tokens.
        Useful for attention visualization overlaid on the original tokens.

        Returns:
            (ids, tokens) — parallel lists.
        """
        tokens = self._tokenize_text(text)
        unk_id = self.SPECIAL_TOKENS['<UNK>']
        ids = [self.word2id.get(t, unk_id) for t in tokens]
        return ids, tokens

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, List]:
        """
        Batch encode with optional padding and truncation.

        Returns dict with 'input_ids' and 'attention_mask'.
        """
        batch_ids: List[List[int]] = []

        for text in texts:
            ids = self.encode(text)
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]
            batch_ids.append(ids)

        if padding:
            if max_length:
                target_len = max_length
            elif batch_ids:
                target_len = max(len(ids) for ids in batch_ids)
            else:
                target_len = 0

            pad_id = self.SPECIAL_TOKENS['<PAD>']
            batch_ids = [
                ids + [pad_id] * (target_len - len(ids))
                for ids in batch_ids
            ]

        pad_id = self.SPECIAL_TOKENS['<PAD>']
        attention_masks = [
            [1 if tok != pad_id else 0 for tok in ids]
            for ids in batch_ids
        ]

        return {
            'input_ids': batch_ids,
            'attention_mask': attention_masks,
        }

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text with clean punctuation attachment.

        Args:
            ids:                  List of integer token IDs.
            skip_special_tokens:  Strip special tokens from output.

        Returns:
            Reconstructed text string.
        """
        tokens: List[str] = []
        special_set = set(self.SPECIAL_TOKENS.keys())

        for idx in ids:
            token = self.id2word.get(idx)
            if token is None:
                continue
            if skip_special_tokens and token in special_set:
                continue
            tokens.append(token)

        if not tokens:
            return ''

        # Smart join: no space before punctuation
        result_parts: List[str] = [tokens[0]]
        for token in tokens[1:]:
            if token and token[0] in self._NO_SPACE_BEFORE:
                result_parts.append(token)           # attach directly
            else:
                result_parts.append(' ' + token)

        return ''.join(result_parts).strip()

    # ------------------------------------------------------------------
    # Padding / Truncation helpers
    # ------------------------------------------------------------------

    def pad_to_length(self, ids: List[int], length: int) -> List[int]:
        """Right-pad ids with <PAD> to reach `length`. Truncates if longer."""
        pad_id = self.SPECIAL_TOKENS['<PAD>']
        if len(ids) >= length:
            return ids[:length]
        return ids + [pad_id] * (length - len(ids))

    def truncate(self, ids: List[int], max_length: int) -> List[int]:
        """Truncate ids to at most `max_length` tokens."""
        return ids[:max_length]

    # ------------------------------------------------------------------
    # Membership test
    # ------------------------------------------------------------------

    def __contains__(self, token: str) -> bool:
        """Check if a token is in the vocabulary (not OOV)."""
        return token in self.word2id

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Persist vocabulary to JSON for checkpoint restoration."""
        data = {
            'version':        '2.0',
            'max_vocab_size': self.max_vocab_size,
            'word2id':        self.word2id,
            'word_freq':      dict(self.word_freq),
            'special_tokens': self.SPECIAL_TOKENS,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Tokenizer] Saved: {path} ({len(self.word2id):,} tokens)")

    def load(self, path: str):
        """Restore vocabulary from a JSON checkpoint."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        raw_word2id = data.get('word2id', {})

        # Coerce values — older checkpoints may store strings or dicts
        clean_word2id: Dict[str, int] = {}
        for k, v in raw_word2id.items():
            if isinstance(v, int):
                clean_word2id[k] = v
            elif isinstance(v, str):
                try:
                    clean_word2id[k] = int(v)
                except ValueError:
                    continue
            elif isinstance(v, dict) and 'id' in v:
                clean_word2id[k] = int(v['id'])

        self.word2id        = clean_word2id
        self.id2word        = {v: k for k, v in clean_word2id.items()}
        self.word_freq      = defaultdict(int, data.get('word_freq', {}))
        self.max_vocab_size = data.get('max_vocab_size', 32000)

        print(f"[Tokenizer] Loaded: {path} ({len(self.word2id):,} tokens)")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    def __len__(self) -> int:
        return len(self.word2id)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size:,}, "
            f"max_vocab_size={self.max_vocab_size:,})"
        )