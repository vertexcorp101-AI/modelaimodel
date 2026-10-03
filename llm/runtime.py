"""Auto-download the model + llamafile runtime when they are missing.

Used on first launch (and on server deploys like Belmo, where the
hundred-MB binaries cannot be committed to git). Downloading here at
startup keeps the repo small and the app self-healing:
  * models/*.gguf           (the brain)
  * models/llamafile[.exe]  (the runtime that executes it)
"""
import os
import platform
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from .gguf_backend import DATA_ROOT, MODELS_DIR, find_llamafile

LLAMAFILE_VER = "0.10.6"
BASE = "https://github.com/mozilla-ai/llamafile/releases/download"

# Swappable model. Qwen2.5-3B-Instruct Q4_K_M (2007 MB) is the default: it is
# the smallest tier that reliably follows the ~9k-token VERTEX master prompt,
# and it still fits a Colab T4 (1.96 GB weights + 1.21 GB KV cache at ctx 32768,
# whose 32768 is the model's native max_position_embeddings).
# Qwen2.5-0.5B stays available for the CPU / tiny-host fallback; SmolLM2 for
# genuinely cramped hosts. Set LIGHTBRAIN_MODEL to pick.
MODELS_CFG = {
    "Qwen2.5-3B-Instruct": (
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/"
        "resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"),
    "Qwen2.5-0.5B-Instruct": (
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/"
        "resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "SmolLM2-360M-Instruct": (
        "SmolLM2-360M-Instruct-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/SmolLM2-360M-Instruct-GGUF/"
        "resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"),
    "SmolLM2-135M-Instruct": (
        "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/"
        "resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
}

MODEL_NAME = os.environ.get("LIGHTBRAIN_MODEL", "Qwen2.5-3B-Instruct")
if MODEL_NAME not in MODELS_CFG:
    raise SystemExit(
        f"[runtime] unknown LIGHTBRAIN_MODEL={MODEL_NAME!r}; "
        f"choose one of {sorted(MODELS_CFG)}")
MODEL_FILE, GGUF_URL = MODELS_CFG[MODEL_NAME]

# Plain-ELF llama.cpp server used on Linux servers (Belmo etc.). llamafile's
# APE single-file binaries can fail with ENOEXEC on container kernels that
# lack the APE/binfmt_misc trampoline; the llama.cpp release ships a normal
# ELF `llama-server` that runs anywhere. CLI flags are the same llama.cpp.
LLAMA_CPP_VER = "b11179"
LLAMA_CPP_URL = ("https://github.com/ggml-org/llama.cpp/releases/download/"
                 f"{LLAMA_CPP_VER}/llama-{LLAMA_CPP_VER}-bin-ubuntu-{{arch}}.tar.gz")


def _download(url, dest):
    """Stream url -> dest with progress. Resumes an existing *.part file via
    the HTTP Range header, prints MB/percent/speed so the operator (and the
    Belmo server log) can watch it."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".part")
    name = url.rsplit("/", 1)[-1]

    start = tmp.stat().st_size if tmp.is_file() else 0
    headers = {"User-Agent": "lightbrain"}
    mode = "ab"
    if start:
        headers["Range"] = f"bytes={start}-"
        print(f"[runtime] resuming {name} at {start/1e6:.1f} MB ...", flush=True)
    else:
        print(f"[runtime] downloading {name} ...", flush=True)

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=1800) as r:
        total = None
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            total = int(cr.rsplit("/", 1)[1])
        elif r.status == 200 and start == 0:
            total = int(r.headers.get("Content-Length") or 0)
        if r.status == 200 and start:
            mode = "wb"
            start = 0

        t0 = time.time()
        done = start
        with open(tmp, mode) as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                f.flush()
                done += len(chunk)
                now = time.time()
                if now - t0 >= 2:
                    mb = done / 1e6
                    rate = len(chunk) / 1e6 / (now - t0)
                    if total:
                        pct = 100.0 * done / total
                        print(f"[runtime] {name}: {mb:.1f}/{total/1e6:.1f} MB "
                              f"({pct:.1f}%, {rate:.2f} MB/s)", flush=True)
                    else:
                        print(f"[runtime] {name}: {mb:.1f} MB "
                              f"({rate:.2f} MB/s)", flush=True)
                    t0 = now

    tmp.replace(dest)
    print(f"[runtime] saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB)",
          flush=True)


def _head_length(url):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "lightbrain"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _ensure_gguf():
    want = MODELS_DIR / MODEL_FILE
    # kill leftovers from interrupted downloads (BELMO: restarts wipe /tmp,
    # leaving cut-off files that then eat disk + break the next boot)
    for leftover in list(MODELS_DIR.glob("*.part")):
        try:
            leftover.unlink()
        except OSError:
            pass
    expected = _head_length(GGUF_URL)
    if want.is_file():
        if not expected or want.stat().st_size in (expected, 0):
            if want.stat().st_size:
                return True
        try:
            want.unlink()
        except OSError:
            pass
    _download(GGUF_URL, want)
    return want.is_file()


def _linux_llama_cpp():
    arch = platform.machine().lower()
    if arch in ("aarch64", "arm64"):
        kind = "arm64"
    elif arch in ("x86_64", "amd64"):
        kind = "x64"
    else:
        print(f"[runtime] unsupported linux arch {arch}", flush=True)
        return False
    tgz = MODELS_DIR / f"llama-{LLAMA_CPP_VER}-{kind}.tar.gz"
    _download(LLAMA_CPP_URL.format(arch=kind), tgz)
    root = MODELS_DIR / "llama-cpp"
    # fresh tree every boot: llama-server links shared libs (.so) that must
    # stay next to it — the whole archive is extracted so RPath resolves
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as t:
        t.extractall(root)
    tgz.unlink(missing_ok=True)
    # trim: the release ships ~70 MB of example binaries (llama-cli, bench,
    # quantize...) we never run. Keep llama-server + its shared .so libs.
    for p in list(root.rglob("*")):
        if p.is_file() and p.name != "llama-server" and \
                not p.name.startswith("lib"):
            p.unlink()
    try:
        du = shutil.disk_usage(MODELS_DIR)
        print(f"[runtime] after extract: {du.free/1e6:.0f} MB free", flush=True)
    except OSError:
        pass
    exe = next((p for p in root.rglob("llama-server")), None)
    if exe:
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR
                  | stat.S_IXGRP | stat.S_IXOTH)
        # drop the stale llamafile APE (freed ~368 MB, avoids ENOEXEC if chosen)
        stale = MODELS_DIR / "llamafile"
        if stale.is_file():
            try:
                stale.unlink()
            except OSError:
                pass
        print(f"[runtime] ready: {exe}", flush=True)
        return True
    print("[runtime] llama-server not found in archive", flush=True)
    return False


def _ensure_llamafile():
    if find_llamafile():
        return True
    if sys.platform == "win32":
        url = f"{BASE}/{LLAMAFILE_VER}/llamafile-{LLAMAFILE_VER}.zip"
        zip_dest = MODELS_DIR / f"llamafile-{LLAMAFILE_VER}.zip"
        _download(url, zip_dest)
        with zipfile.ZipFile(zip_dest) as z:
            target = next((n for n in z.namelist()
                           if Path(n).name == "llamafile"), None)
            if target is None:
                raise SystemExit(
                    "[runtime] llamafile.zip: executable not found in archive")
            exe = MODELS_DIR / "llamafile.exe"
            with z.open(target) as src, open(exe, "wb") as f:
                shutil.copyfileobj(src, f)
        zip_dest.unlink(missing_ok=True)
    elif sys.platform.startswith("linux"):
        # Normal ELF server (no APE) so container kernels without binfmt_misc
        # can exec it — this is what fails on Belmo with llamafile.
        if not _linux_llama_cpp():
            raise SystemExit("[runtime] could not install llama-server")
    else:
        # macOS (and anything else): llamafile's single binary works natively.
        url = f"{BASE}/{LLAMAFILE_VER}/llamafile-{LLAMAFILE_VER}"
        dest = MODELS_DIR / "llamafile"
        _download(url, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR
                   | stat.S_IXGRP | stat.S_IXOTH)
    return find_llamafile() is not None


def _prune_tmp():
    """Free disk when we live on a temp filesystem (Belmo: /tmp is small and
    NOT wiped between boots automatically — old model variants/archives pile
    up across redeploys and eventually cause instant ENOSPC)."""
    if not str(DATA_ROOT).startswith(tempfile.gettempdir()):
        return  # repo-backed host (local dev): keep everything cached
    try:
        du = shutil.disk_usage(MODELS_DIR)
        print(f"[runtime] tmp disk: {du.free/1e6:.0f} MB free of "
              f"{du.total/1e6:.0f} MB", flush=True)
    except OSError:
        pass
    for p in list(MODELS_DIR.glob("*.gguf")):
        if p.name != MODEL_FILE:
            try:
                p.unlink()
                print(f"[runtime] pruned old model {p.name}", flush=True)
            except OSError:
                pass
    for p in list(MODELS_DIR.glob("*.part")) + \
             list(MODELS_DIR.glob("llama-*.tar.gz")):
        try:
            p.unlink()
        except OSError:
            pass


def ensure_runtime():
    _prune_tmp()
    ok = _ensure_gguf() and _ensure_llamafile()
    if not ok:
        raise SystemExit("[runtime] could not obtain the model + llamafile "
                         "runtime. Check network access.")
    return True