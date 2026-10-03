"""GGUF backend: talk to a local llamafile server to run a real pre-trained
CPU model (SmolLM2-135M-Instruct Q4). This is the *fast* path:
  * no pip install, no C compiler, no torch — only stdlib (urllib + subprocess)
  * one self-contained llamafile.exe does all of llama.cpp for us
It exposes the exact same interface as the trained-Model Agent (respond /
stream_respond / build_prompt / memory / tools), so server.py and chat_cmd
keep working unchanged. If llamafile or the .gguf is missing it returns
None and the caller falls back to the trained tiny model.
"""
import json
import os
import random
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


def _resolve_data_root():
    """Writable base for models/ and state/. Belmo mounts the app dir (/app)
    read-only, so try the repo first (local dev / writable hosts), then fall
    back to an env override or a temp dir. Set LIGHTBRAIN_DATA_DIR to a
    persistent volume if your host has one (e.g. /data)."""
    env = os.environ.get("LIGHTBRAIN_DATA_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    for base in (Path(__file__).resolve().parent.parent,
                 Path.cwd(),
                 Path(tempfile.gettempdir()) / "lightbrain-data"):
        try:
            d = base / "models"
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".w"  # verify the dir is actually writable
            probe.write_text("ok")
            probe.unlink()
            return base
        except OSError:
            continue
    raise SystemExit("[gguf] no writable data dir - set LIGHTBRAIN_DATA_DIR")


DATA_ROOT = _resolve_data_root()
MODELS_DIR = DATA_ROOT / "models"

SYS_MARK = "you are tiny-vertex"

SYSTEM_PROMPT = (
    "you are tiny-Vertex, the user's close friend, not an assistant. "
    "text like a friend texts: 1-2 short casual sentences, warm and comforting. "
    "never lecture, never talk like a formal assistant. "
    "read the user's feeling first: if they are sad, down or lonely, skip any "
    "cheerful greeting and go straight to comfort. never ask 'how's it going' "
    "or 'how's life treating you' when the user is down - comfort first, no "
    "questions about their day. answer only THIS message - "
    "never assume feelings or carry moods between messages. never offer advice "
    "unless asked. "
    "when the user shares their name or personal things, react warmly and use "
    "their name from then on. if the user asks whether you know their name, "
    "look at 'things you remember' above: if a name is there, say it warmly; "
    "if not, ask them to tell you. what they share is saved by the system and shown "
    "back to you under 'things you remember', so treat those as things you "
    "genuinely remember - never claim you cannot remember things. never invent "
    "specific shared memories like fights, trips, hobbies or events - only ever "
    "refer to things listed under 'things you remember'. "
    "examples - user 'hey' -> 'hey! good to see you.' "
    "user 'i am sad' -> 'oh no, what happened? I'm here.'"
)


def chatml_prompt(system, user, memory_hits=(), knowledge_hits=()):
    """Build the SmolLM2-Instruct ChatML prompt for a completion request."""
    parts = ["<|im_start|>system",
             system.strip()]
    if memory_hits:
        parts.append("\nSome things you remember:\n"
                     + "\n".join(f"- {t[:200]}" for t in memory_hits))
    if knowledge_hits:
        parts.append("\nKnowledge you know:\n"
                     + "\n".join(f"- {t[:200]}" for t in knowledge_hits))
    parts += ["<|im_end|>",
              "<|im_start|>user",
              user.strip(),
              "<|im_end|>",
              "<|im_start|>assistant"]
    return "\n".join(parts)


def find_file(patterns):
    for pat in patterns:
        hits = [p for p in sorted(MODELS_DIR.glob(pat)) if p.is_file()]
        if hits:
            return hits[0]
    return None


def find_gguf_by_name(filename):
    """Find one gguf by exact name, case-insensitively.

    The upstream HuggingFace filenames are lower-cased
    (qwen2.5-3b-instruct-q4_k_m.gguf) while the ids in MODELS_CFG are
    title-cased, so a plain glob only matches on case-insensitive
    filesystems and silently finds nothing on Linux/Colab.
    """
    exact = MODELS_DIR / filename
    if exact.is_file():
        return exact
    target = filename.lower()
    for p in sorted(MODELS_DIR.glob("*.gguf")):
        if p.is_file() and p.name.lower() == target:
            return p
    return None


def find_gguf():
    """Locate the GGUF for the SELECTED model only.

    Deliberately no bare "*.gguf" fallback: a stray gguf left in models/ (e.g.
    a SmolLM2 from an earlier local run) would be served while --alias claims
    the configured model, so /v1/models would advertise Qwen while the weights
    on disk were something else. If the selected model is absent,
    ensure_runtime() downloads it; returning None here keeps that honest.
    """
    from .runtime import MODEL_FILE
    return find_gguf_by_name(MODEL_FILE)


def find_llamafile():
    # llama-server (plain ELF, Linux) preferred: llamafile's APE can fail to
    # exec on container kernels. Windows keeps using llamafile.exe.
    return find_file(("llama-server", "**/llama-server",
                      "llamafile.exe", "llamafile",
                      "**/llamafile.exe", "**/llamafile"))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def detect_gpu_layers():
    """How many layers to offload to GPU (-ngl). Explicit override wins;
    otherwise auto-detect a CUDA device (Colab GPU VM). Returns (ngl, device)."""
    override = os.environ.get("LIGHTBRAIN_GPU_LAYERS", "").strip()
    if override:
        try:
            return int(override), ("gpu" if int(override) > 0 else "cpu")
        except ValueError:
            pass
    try:
        if shutil.which("nvidia-smi") or os.path.exists("/dev/nvidiactl"):
            return 99, "gpu"
    except OSError:
        pass
    return 0, "cpu"


class LlamaServer:
    """Starts llamafile.exe in server mode and talks HTTP to it."""

    def __init__(self, gguf, exe, max_tokens=64, ctx=1024, port=None):
        self.gguf = Path(gguf)
        self.exe = Path(exe)
        self.max_tokens = max_tokens
        self.ctx = ctx
        self.port = port or _free_port()
        self.proc = None
        self.url = f"http://127.0.0.1:{self.port}"
        self.gpu_layers, self.device = detect_gpu_layers()

    def start(self, timeout=120):
        if self.proc and self.proc.poll() is None:
            return True
        is_server = Path(self.exe).name == "llama-server"
        cmd = [str(self.exe), "-m", str(self.gguf)]
        if not is_server:
            cmd += ["--server"]
        cmd += ["--host", "127.0.0.1", "--port", str(self.port),
                "--ctx-size", str(self.ctx), "-ngl", str(self.gpu_layers),
                "--alias",
                os.environ.get("LIGHTBRAIN_MODEL", "Qwen2.5-3B-Instruct"),
                "--temp", "0.7", "--top-p", "0.95",
                "--repeat-penalty", "1.1"]
        print(f"[gguf] starting {self.exe.name} "
              f"(ctx={self.ctx}, offload={self.gpu_layers} layers, {self.device})",
              flush=True)
        self.logfile = MODELS_DIR / f"{self.exe.name}.log"
        try:
            errlog = open(self.logfile, "wb", buffering=0)
        except OSError:
            errlog = subprocess.DEVNULL
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=errlog,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = ""
                if self.logfile.is_file():
                    try:
                        tail = self.logfile.read_text(
                            errors="replace").strip().splitlines()[-3:]
                        tail = " | ".join(tail[-3:])
                    except OSError:
                        pass
                print(f"[gguf] {self.exe.name} exited rc={self.proc.returncode}"
                      + (f" — {tail}" if tail else ""), flush=True)
                return False
            if self._healthy():
                return True
            time.sleep(1)
        return self._healthy()

    def _healthy(self):
        try:
            with urllib.request.urlopen(self.url + "/health",
                                        timeout=2) as r:
                if r.status != 200:
                    return False
                body = r.read(256).decode("utf-8", "replace")
                return '"ok"' in body.lower() or "ok" in body.lower()
        except Exception:
            return False

    def complete(self, prompt, temperature=0.2, top_p=0.9):
        body = json.dumps({
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": 1.2,
            "stop": ["<|im_end|>", "<|im_start|>user"],
        }).encode("utf-8")
        req = urllib.request.Request(self.url + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception:
            return ""
        return (data.get("content") or "").strip()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None


class GgufAgent:
    """Drop-in replacement for the trained Agent: same .respond / .build_prompt,
    but generation comes from the GGUF via llamafile. Every word is generated
    by the model; _call only discards assistant-voiced drafts and resamples."""

    def __init__(self, server, memory, embedder, registry, knowledge=None,
                 max_tokens=64, device="cpu"):
        self.server = server
        self.memory = memory
        self.embedder = embedder
        self.registry = registry
        self.knowledge = knowledge
        self.max_tokens = max_tokens
        self.device = device
        self.tool_names = set(registry.tools)

    def build_prompt(self, user_text):
        mem_hits = []
        try:
            mem_hits = [t for t, _, _ in self.memory.recall(user_text, k=2)]
        except Exception:
            pass
        kn_hits = []
        if self.knowledge is not None and self.embedder is not None:
            try:
                kn_hits = [t for t, _, _ in
                           self.knowledge.store.search(
                               self.embedder.embed(user_text), k=2)]
            except Exception:
                pass
        return chatml_prompt(SYSTEM_PROMPT, user_text, mem_hits, kn_hits)

    def _call(self, prompt):
        """One generation, trimmed hard at any echoed system-prompt junk.

        Retries when the reply slips into baked-in assistant voice: 360M's
        instruction tuning paraphrases around prompt bans, so banned patterns
        are enforced here by regenerating with rising temperature (low temp
        repeats the same reply, so each retry samples hotter for a genuinely
        different phrasing).
        """
        banned = (
            "how can i help", "what brings you", "here to listen", "as an ai",
            "in this digital space", "tasks that need", "thanks for asking",
            "navigate your", "need someone to talk", "some advice on how",
            "don't have access", "no access to", "as an ai language",
            "language model", "confusion", "don't remember",
            "do not remember", "can't remember", "cannot remember",
            "hello there", "remember when", "specific information",
            "meaningful answer", "remember me", "doing great",
            "here for my own benefit", "artificial intelligence system",
            "running solely from user input",
        )
        text = ""
        for temp in (0.35, 0.6, 0.9):
            raw = self.server.complete(prompt, temperature=temp)
            if any(b in raw.lower() for b in banned):
                continue
            candidate = raw[:low_i] if (low_i := raw.lower().find(SYS_MARK)) >= 0 else raw
            candidate = candidate.strip()
            if candidate:
                # End at a sentence boundary so the token cap never cuts
                # mid-word: "oh no, what happened? I'm here. And also" -> "...here."
                ends = list(re.finditer(r"[.!?](?=\s|$)", candidate))
                if ends and ends[-1].start() > 10:
                    candidate = candidate[:ends[-1].start() + 1]
                text = candidate
                break
        return text

    def respond(self, user_text, verbose=False):
        reply = self._call(self.build_prompt(user_text))
        if not reply:
            reply = "hmm, say that again?"
        return reply, []

    def stream_respond(self, user_text, verbose=False):
        """Yield NDJSON-ready events like the trained agent:
        {"type":"text","text":...}, finally {"type":"done","text":...,"tools":[...]}."""
        prompt = self.build_prompt(user_text)
        text = self._call(prompt)
        if not text:
            text = "hmm, say that again?"
        if verbose:
            print(f"\n[gguf] {text}")
        yield {"type": "text", "text": text}
        yield {"type": "done", "text": text, "tools": []}


def load_backend(model_dir="models", knowledge_dir="knowledge",
                 memory_dir="runs/memory", device="cpu", max_tokens=64,
                 no_rag=False, verbose=False):
    """Start the GGUF brain (llamafile + SmolLM2). Returns None if not ready,
    in which case the caller falls back to the trained model."""
    from .memory import Memory
    from .embeddings import Embedder
    from .tools import build_default_registry
    from .rag import KnowledgeBase, VectorStore

    gguf = model_dir if bool(Path(model_dir).suffix == ".gguf") else None
    if not gguf:
        gguf = find_gguf()
    exe = find_llamafile()
    if not gguf or not exe:
        return None

    server = LlamaServer(gguf, exe, max_tokens=max_tokens)
    if not server.start():
        print("[gguf] llamafile could not start; falling back to trained model")
        server.stop()
        return None

    embedder = Embedder(None, None, device=device)
    memory = Memory(store_dir=memory_dir, embedder=embedder)
    knowledge = None
    if not no_rag:
        try:
            kb = KnowledgeBase(VectorStore(), source_dir=knowledge_dir)
            n = kb.load(embedder)
            if n:
                print(f"knowledge index: {n} chunks")
            knowledge = kb
        except Exception as exc:
            print(f"[gguf] knowledge skipped: {exc}")

    registry = build_default_registry()
    agent = GgufAgent(server, memory, embedder, registry, knowledge,
                      max_tokens=max_tokens, device=device)
    print(f"[gguf] brain online: {Path(gguf).name} "
          f"({server.device}, {server.gpu_layers} layers offloaded)")
    return {"agent": agent, "memory": memory, "registry": registry,
            "knowledge": knowledge, "embedder": embedder, "server": server}
