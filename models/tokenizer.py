import re
import json
from collections import defaultdict
from typing import List, Dict, Optional, Union


class Tokenizer:
    """
    Production-grade tokenizer for AERIS SLM
    
    Features:
    - Word-level tokenization with regex preprocessing
    - Special token handling (<PAD>, <UNK>, <START>, <END>, <USER>, <ASSISTANT>, <EOS>)
    - Vocabulary management with frequency tracking
    - Serialization support for checkpointing
    - Case normalization and punctuation handling
    """
    
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3,
        '<MASK>': 4,
        '<USER>': 5,
        '<ASSISTANT>': 6,
        '<EOS>': 7
    }
    
    def __init__(self, max_vocab_size: int = 10000):
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.word_freq: Dict[str, int] = defaultdict(int)
        self.max_vocab_size = max_vocab_size
        
        # Initialize special tokens
        for token, idx in self.SPECIAL_TOKENS.items():
            self.word2id[token] = idx
            self.id2word[idx] = token
        
        self._compiled_regex = re.compile(r"<[^>]+>|\w+|[^\w\s]")
        
    def train(self, texts: List[str], min_freq: int = 1):
        """
        Build vocabulary from corpus with frequency filtering
        
        Args:
            texts: List of text strings to build vocabulary from
            min_freq: Minimum word frequency to include in vocabulary
        """
        # Count frequencies
        for text in texts:
            if not isinstance(text, str):
                continue
            words = self._tokenize_text(text)
            for word in words:
                self.word_freq[word] += 1
        
        # Sort by frequency (descending) and filter
        sorted_words = sorted(
            self.word_freq.items(),
            key=lambda x: (-x[1], x[0])
        )
        
        # Add to vocabulary (respecting max_vocab_size and min_freq)
        for word, freq in sorted_words:
            if len(self.word2id) >= self.max_vocab_size:
                break
            if freq < min_freq:
                continue
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word
        
        print(f"[Tokenizer] Vocabulary size: {len(self.word2id)} (trained on {len(texts)} texts)")
        
    def _tokenize_text(self, text: str) -> List[str]:
        """Internal tokenization with preprocessing"""
        # Normalize: strip only (preserve special tokens case)
        text = text.strip()
        
        # Tokenize with regex that preserves special tokens
        tokens = self._compiled_regex.findall(text)
        
        # Filter empty and normalize
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: String or list of strings to encode
            add_special_tokens: Whether to add <START> and <END> tokens
            
        Returns:
            List of token IDs
        """
        if isinstance(text, list):
            # Encode list as single sequence
            text = ' '.join(str(t) for t in text)
        
        tokens = self._tokenize_text(text)
        
        ids = []
        if add_special_tokens:
            ids.append(self.SPECIAL_TOKENS['<START>'])
        
        for token in tokens:
            ids.append(self.word2id.get(token, self.SPECIAL_TOKENS['<UNK>']))
        
        if add_special_tokens:
            ids.append(self.SPECIAL_TOKENS['<END>'])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text string
        """
        tokens = []
        for idx in ids:
            if idx in self.id2word:
                token = self.id2word[idx]
                if skip_special_tokens and token in self.SPECIAL_TOKENS:
                    continue
                tokens.append(token)
        
        # Join with spaces, but handle punctuation properly
        text = ''
        for i, token in enumerate(tokens):
            if i > 0 and token and not token[0].isalnum() and token[0] != '<':
                # Punctuation - no space before
                text += token
            else:
                if i > 0:
                    text += ' '
                text += token
        
        return text
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None, 
                     padding: bool = True, truncation: bool = True) -> Dict[str, List]:
        """
        Batch encoding with padding and truncation
        
        Returns dict with 'input_ids' and 'attention_mask'
        """
        batch_ids = []
        for text in texts:
            ids = self.encode(text)
            
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]
            
            batch_ids.append(ids)
        
        if padding and max_length:
            # Pad to max_length
            for ids in batch_ids:
                while len(ids) < max_length:
                    ids.append(self.SPECIAL_TOKENS['<PAD>'])
        
        # Create attention masks
        attention_masks = []
        for ids in batch_ids:
            mask = [1 if id != self.SPECIAL_TOKENS['<PAD>'] else 0 for id in ids]
            attention_masks.append(mask)
        
        return {
            'input_ids': batch_ids,
            'attention_mask': attention_masks
        }
    
    def save(self, path: str):
        """Save tokenizer to JSON"""
        data = {
            'word2id': self.word2id,
            'word_freq': dict(self.word_freq),
            'max_vocab_size': self.max_vocab_size,
            'special_tokens': self.SPECIAL_TOKENS
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        word2id = data.get("word2id", {})

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

        self.word2id = clean_word2id
        self.id2word = {v: k for k, v in clean_word2id.items()}
        self.word_freq = defaultdict(int, data.get('word_freq', {}))
        self.max_vocab_size = data.get('max_vocab_size', 10000)

        print(f"[Tokenizer] Loaded vocabulary: {len(self.word2id)} words")
    
    @property
    def vocab_size(self) -> int:
        return len(self.word2id)
    
    def __len__(self) -> int:
        return len(self.word2id)