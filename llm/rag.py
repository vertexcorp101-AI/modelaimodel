"""Retrieval-augmented generation: local vector store + knowledge folder index."""
import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, dimension=256):
        self.dimension = dimension
        self.texts = []
        self.vectors = np.zeros((0, dimension), dtype=np.float32)
        self.sources = []

    def add(self, text, vector, source=""):
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        self.texts.append(text)
        self.sources.append(source)
        self.vectors = np.concatenate([self.vectors, v], axis=0)

    def add_batch(self, texts, vectors, sources=None):
        for i, t in enumerate(texts):
            src = sources[i] if sources else ""
            self.add(t, vectors[i], src)

    def search(self, query_vector, k=3):
        if len(self.texts) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm
        scores = self.vectors @ q
        idx = np.argsort(scores)[::-1][:k]
        return [(self.texts[i], float(scores[i]), self.sources[i]) for i in idx]

    def __len__(self):
        return len(self.texts)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path.with_suffix(".npy")), self.vectors)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "sources": self.sources,
                       "dimension": self.dimension}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        path = Path(path)
        npy = np.load(str(path.with_suffix(".npy")))
        with open(path.with_suffix(".json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        store = cls(data["dimension"])
        store.vectors = npy
        store.texts = data["texts"]
        store.sources = data["sources"]
        return store

    def texts_for_fallback(self):
        return self.texts


MIN_CHUNK = 180
MAX_CHUNK = 480


def chunk_text(text, min_chunk=MIN_CHUNK, max_chunk=MAX_CHUNK):
    """Split a document into self-contained, roughly similar-size chunks."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) < min_chunk:
            buf += (" " if buf else "") + p
            continue
        if buf:
            chunks.append(buf)
            buf = p
        else:
            chunks.extend(p[i:i + max_chunk] for i in range(0, len(p), max_chunk))
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) > 0]


class KnowledgeBase:
    """Indexed knowledge loaded from files under ./knowledge."""

    def __init__(self, store, source_dir="knowledge"):
        self.store = store
        self.source_dir = Path(source_dir)

    def load(self, embedder, rebuild=False):
        index_path = self.source_dir / "vector_store"
        if not rebuild and self.source_dir.exists() and index_path.with_suffix(".npy").exists():
            try:
                self.store = VectorStore.load(index_path)
                return len(self.store)
            except Exception:
                pass
        if not self.source_dir.exists():
            return 0
        docs = []
        files = sorted(self.source_dir.rglob("*.txt")) + \
            sorted(self.source_dir.rglob("*.md"))
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            for chunk in chunk_text(text):
                docs.append((chunk, str(f)))
        if docs:
            self.store = VectorStore(embedder.embed(docs[0][0]).shape[0])
            for i in range(0, len(docs), 32):
                batch = docs[i:i + 32]
                vecs = embedder.embed_batch([d[0] for d in batch])
                self.store.add_batch([d[0] for d in batch], vecs,
                                     [d[1] for d in batch])
            self.store.save(index_path)
        return len(self.store)