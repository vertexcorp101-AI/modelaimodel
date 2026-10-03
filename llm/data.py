"""Corpus loading and random-window batching for CPU training."""
import random
from pathlib import Path

import torch


def load_texts(paths):
    """Read every .txt/.md/.py/.csv/.json file under the given paths."""
    paths = [p for p in paths if p]
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = sorted(p.rglob("*.txt")) + sorted(p.rglob("*.md")) + \
                sorted(p.rglob("*.csv")) + sorted(p.rglob("*.json"))
        else:
            continue
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(text.strip()) >= 40:
                out.append(text)
    return out


class WindowDataset:
    """Keeps the whole corpus tokenized once, then samples block_size windows."""

    def __init__(self, texts, tokenizer, block_size, device="cpu"):
        ids = []
        for t in texts:
            ids.extend(tokenizer.encode(t))
            ids.append(tokenizer.token_to_id.get("<s>", 0))
        tensor = torch.tensor(ids, dtype=torch.long) if ids else torch.zeros(1, dtype=torch.long)
        if len(tensor) < block_size + 1:
            reps = (block_size + 1) // max(1, len(tensor)) + 1
            tensor = tensor.repeat(reps)
        self.ids = tensor
        self.block_size = block_size
        self.device = device

    def __len__(self):
        return max(1, len(self.ids) - self.block_size)

    def batch(self, batch_size, rng=None):
        rng = rng or random
        hi = len(self.ids) - self.block_size
        starts = [rng.randrange(hi) for _ in range(batch_size)]
        x = torch.stack([self.ids[s:s + self.block_size] for s in starts]).to(self.device)
        y = torch.stack([self.ids[s + 1:s + self.block_size + 1] for s in starts]).to(self.device)
        return x, y