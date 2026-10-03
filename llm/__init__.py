"""LightBrain package. All heavy imports (torch, numpy) are lazy so that the
OpenAI-compatible provider (provider.py, low-memory servers) never loads them.
Submodules are imported on first attribute access via PEP 562 __getattr__.
"""
import importlib

_LAZY = {
    "agents": ("llm.agents", "Agent"),
    "config": ("llm.config", "ModelConfig"),
    "embeddings": ("llm.embeddings", "Embedder"),
    "memory": ("llm.memory", "Memory"),
    "model": ("llm.model", "GPT"),
    "rag": ("llm.rag", "KnowledgeBase"),
    "VectorStore": ("llm.rag", "VectorStore"),
    "tokenizer": ("llm.tokenizer", "Tokenizer"),
    "tools": ("llm.tools", "build_default_registry"),
    "data": ("llm.data", None),
    "train_cmd": ("llm.train_cmd", None),
    "chat_cmd": ("llm.chat_cmd", None),
    "gguf_backend": ("llm.gguf_backend", None),
    "runtime": ("llm.runtime", None),
    "numpy": ("numpy", None),
}

__all__ = ["Agent", "Embedder", "GPT", "KnowledgeBase", "Memory", "ModelConfig",
           "Tokenizer", "VectorStore", "build_default_registry", "numpy"]


def __getattr__(name):
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module 'llm' has no attribute {name!r}")
    mod = importlib.import_module(entry[0])
    return getattr(mod, entry[1]) if entry[1] else mod


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(__all__))