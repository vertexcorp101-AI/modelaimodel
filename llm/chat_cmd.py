"""Interactive chat: `python -m llm.chat_cmd --model runs/checkpoints/model`."""
import argparse
import time
from pathlib import Path

from .config import ModelConfig
from .embeddings import Embedder
from .memory import Memory
from .rag import KnowledgeBase, VectorStore
from .tokenizer import Tokenizer
from .tools import build_default_registry

WELCOME = """
LightBrain - local English assistant with memory, RAG and agents.
Commands: /help  /remember <text>  /stats  /tools  /quit
Type your message to chat.
"""


def build_parser():
    p = argparse.ArgumentParser(description="Chat with LightBrain")
    p.add_argument("--model", default="runs/checkpoints/model",
                   help="checkpoint prefix (model.pt + model.json + tokenizer.json)")
    p.add_argument("--knowledge", default="knowledge", help="folder of documents")
    p.add_argument("--memory", default="runs/memory", help="memory folder")
    p.add_argument("--device", default="auto")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max_tokens", type=int, default=30)
    p.add_argument("--no-rag", action="store_true", help="skip knowledge index")
    return p


def pick_device(name):
    if name == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_backend(model="runs/checkpoints/model", knowledge_dir="knowledge",
                 memory_dir="runs/memory", device="auto", max_tokens=80,
                 no_rag=False):
    """Load tokenizer + model + RAG + memory + tools. Returns an AgentBundle."""
    device = pick_device(device)
    prefix = Path(model)

    from .gguf_backend import load_backend as _gguf
    gguf = _gguf(model, knowledge_dir, memory_dir, device, max_tokens, no_rag)
    if gguf is not None:
        return gguf

    if not prefix.with_name("model.pt").exists():
        raise SystemExit(
            "No trained model found at " + str(prefix) +
            ".\nTrain one first:  python -m llm.train_cmd --data samples\n"
            "(several minutes on CPU with the default tiny model).")

    import torch
    from .agents import Agent
    from .model import GPT

    tokenizer = Tokenizer.load(prefix.with_name("tokenizer.json"))
    mcfg = ModelConfig.load(prefix.with_name("model.json"))
    model = GPT(mcfg).to(device)
    model.load_state_dict(torch.load(prefix.with_name("model.pt"),
                                     map_location=device)["model"])
    model.eval()
    print(f"loaded model: {mcfg.param_count()/1e6:.1f}M params, vocab={mcfg.vocab_size}, "
          f"device={device}")

    embedder = Embedder(tokenizer, model if mcfg.param_count() < 100_000_000 else None,
                        device=device)
    knowledge = None
    if not no_rag:
        kb = KnowledgeBase(VectorStore(), source_dir=knowledge_dir)
        n = kb.load(embedder)
        if n:
            print(f"knowledge index: {n} chunks")
        knowledge = kb
    memory = Memory(store_dir=memory_dir, embedder=embedder)
    registry = build_default_registry()
    agent = Agent(model, tokenizer, embedder, registry, knowledge, memory,
                  device=device, max_tokens=max_tokens)
    return {"agent": agent, "memory": memory, "registry": registry,
            "knowledge": knowledge, "embedder": embedder, "tokenizer": tokenizer,
            "model": model, "device": device}


def serve_commands(user, memory, registry):
    """Handle built-in chat commands; returns a reply string or None."""
    if user in ("/help", "/help"):
        return WELCOME
    if user == "/stats":
        return str(memory.stats())
    if user == "/tools":
        return registry.specs_text()
    if user.startswith("/remember "):
        ok = memory.remember(user[len("/remember "):], source="user")
        return "saved." if ok else "failed."
    return None


def main():
    args = build_parser().parse_args()
    backend = load_backend(args.model, args.knowledge, args.memory,
                           args.device, args.max_tokens, args.no_rag)
    agent, memory, registry = backend["agent"], backend["memory"], backend["registry"]

    print(WELCOME)
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        cmds = serve_commands(user, memory, registry)
        if cmds is not None:
            print(cmds)
            continue

        t0 = time.time()
        reply, tools = agent.respond(user)
        dt = time.time() - t0
        print(f"\nlightbrain> {reply}")
        if tools or dt > 2:
            print(f"  [tools: {', '.join(t for t, _ in tools)}] [{dt:.1f}s]")
        memory.add_exchange(user, reply)


if __name__ == "__main__":
    main()