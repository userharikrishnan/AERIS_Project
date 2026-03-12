import re
from collections import defaultdict

class Tokenizer:
    def __init__(self):
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2word = {0: "<PAD>", 1: "<UNK>"}

    def train(self, texts):
        for text in texts:
            for word in re.findall(r"\w+", text.lower()):
                if word not in self.word2id:
                    idx = len(self.word2id)
                    self.word2id[word] = idx
                    self.id2word[idx] = word

    def encode(self, text):
        return [
            self.word2id.get(w, self.word2id["<UNK>"])
            for w in re.findall(r"\w+", text.lower())
        ]

    def decode(self, ids):
        return " ".join(self.id2word.get(i, "<UNK>") for i in ids)
