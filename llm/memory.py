"""Persistent long-term memory: transcript log + capped vector store of facts."""
import json
import time
from pathlib import Path

from .rag import VectorStore

MAX_ENTRIES = 1500

# Degenerate replies (prompt echoes, empty) should not be remembered.
GARBAGE_MARKERS = ["you are lightbrain, a small", "you are tiny-vertex", "▁sys", "<sys>", "</assistant>"]


class Memory:
    """Conversation transcript plus a retrievable long-term vector memory."""

    def __init__(self, store_dir="runs/memory", embedder=None):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.transcript_path = self.store_dir / "transcript.jsonl"
        dim = self.embedder.embed("").shape[0] if self.embedder else 256
        self.store = VectorStore(dimension=dim)
        self._load(dim)

    def _load(self, fallback_dim=256):
        store_file = self.store_dir / "long_term"
        if store_file.with_suffix(".npy").exists():
            try:
                self.store = VectorStore.load(store_file)
                return
            except Exception:
                pass
        self.store = VectorStore(dimension=fallback_dim)

    def _flush(self):
        self.store.save(self.store_dir / "long_term")

    def remember(self, text, source="",
                 max_entries=MAX_ENTRIES):
        if not text.strip():
            return False
        vec = self.embedder.embed(text)
        self.store.add(text[:600], vec, source)
        if len(self.store) > max_entries:
            recent = self.store.texts[max_entries // 2:]
            self.store = VectorStore(dimension=vec.shape[0])
            for i in range(0, len(recent), 32):
                batch = recent[i:i + 32]
                vecs = self.embedder.embed_batch(batch)
                self.store.add_batch(batch, vecs)
        self._flush()
        return True

    def add_exchange(self, user_text, assistant_text):
        if not user_text.strip() or not assistant_text:
            return False
        if len(assistant_text) > 300 or any(
                m in assistant_text for m in GARBAGE_MARKERS):
            return False
        stamp = time.strftime("%Y-%m-%d %H:%M")
        self._append_transcript(user_text, assistant_text, stamp)
        chunk = f"[{stamp}] User asked: {user_text[:300]} Assistant answered: {assistant_text[:400]}"
        return self.remember(chunk, source="conversation")

    def _append_transcript(self, user, assistant, stamp):
        for role, text in (("user", user), ("assistant", assistant)):
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"role": role, "text": text,
                                    "ts": stamp}) + "\n")

    def recall(self, query, k=3):
        return self.store.search(self.embedder.embed(query), k=k)

    def recent_turns(self, n=6):
        if not self.transcript_path.exists():
            return []
        lines = self.transcript_path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines[-n:]]

    def stats(self):
        return {"long_term_entries": len(self.store),
                "transcript_lines": len(
                    self.transcript_path.read_text(encoding="utf-8").strip().splitlines())
                if self.transcript_path.exists() else 0}