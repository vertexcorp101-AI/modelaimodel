"""Text embeddings: exact (mean-pooled LM hidden states) or lightweight TF-IDF fallback.

The fallback keeps memory + knowledge retrieval working *before* any training.
torch is imported lazily so pure-GGUF setups never need it installed.
"""
import math

import numpy as np


def _trigrams(text):
    text = " ".join(text.lower().split())[:2000]
    if len(text) < 3:
        return []
    return [text[i:i + 3] for i in range(len(text) - 2)]


class HashedTFIDF:
    """Hashed character-trigram TF-IDF embeddings, pure NumPy."""

    def __init__(self, dim=256):
        self.dim = dim
        self.df = {}
        self.n_docs = 0

    def build(self, docs):
        for doc in docs:
            for gram in set(_trigrams(doc)):
                self.df[gram] = self.df.get(gram, 0) + 1
            self.n_docs += 1
        return self

    def embed_batch(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts = {}
            for gram in _trigrams(text):
                counts[gram] = counts.get(gram, 0) + 1
            vec = np.zeros(self.dim, dtype=np.float32)
            for gram, c in counts.items():
                idf = math.log((1 + self.n_docs) / (1 + self.df.get(gram, 0))) + 1.0
                h = hash(gram) % (self.dim * 2)
                sign = 1.0 if h < self.dim else -1.0
                vec[h % self.dim] += sign * math.sqrt(c) * idf
            n = np.linalg.norm(vec)
            if n > 0:
                vec /= n
            out[i] = vec
        return out

    def embed(self, text):
        return self.embed_batch([text])[0]


class Embedder:
    """Wrap either the model or the TF-IDF fallback behind one interface."""

    def __init__(self, tokenizer, model=None, device="cpu"):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.fallback = HashedTFIDF()

    @property
    def using_model(self):
        return self.model is not None

    def warm_fallback(self, docs):
        if not self.using_model and docs:
            self.fallback.build(docs)
        return self

    @property
    def using_model(self):
        return self.model is not None

    def embed_batch(self, texts):
        import torch
        if self.model is None:
            return self.fallback.embed_batch(texts)
        block = self.model.cfg.block_size
        vecs = []
        for text in texts:
            ids = self.tokenizer.encode(text)[:block]
            if len(ids) == 0:
                vecs.append(np.zeros(self.model.cfg.n_embd, dtype=np.float32))
                continue
            n = len(ids)
            pad = block - n
            x = torch.tensor(ids + [0] * pad, dtype=torch.long,
                             device=self.device).unsqueeze(0)
            self.model.eval()
            h = self.model.hidden(x)[0, :n]  # (n, d)
            vec = h.mean(0).cpu().numpy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vecs.append(vec.astype(np.float32))
        return np.stack(vecs)

    def embed(self, text):
        return self.embed_batch([text])[0]