"""Model + training configuration."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Decoder-only transformer layout. Defaults are sized for single-CPU training.

    Raise n_layer/n_embd/n_head/block_size/vocab_size to scale toward 1B params
    (that scale requires a GPU).
    """
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    block_size: int = 256
    dropout: float = 0.1
    tie_weights: bool = True

    def param_count(self):
        return _estimate_params(self.vocab_size, self.n_layer, self.n_head,
                                self.n_embd, self.tie_weights)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


def _estimate_params(vocab, layers, heads, embd, tied):
    attn = 4 * embd * embd
    mlp = 8 * embd * embd
    params = layers * (attn + mlp)
    if tied:
        params += 0
    else:
        params += vocab * embd
    return params


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    min_lr: float = 1e-5
    warmup_steps: int = 200
    batch_size: int = 4
    block_size: int = 128
    grad_accum: int = 4
    epochs: int = 3
    max_steps: int = 0
    log_every: int = 20
    save_every: int = 200
    seed: int = 1337
    sample_every: int = 100
    sample_prompt: str = "Once upon a time"
    sample_tokens: int = 80