# LightBrain

A small, fully local English LLM that trains and runs on a weak CPU-only PC
(4 GB RAM or less). It includes long-term memory, retrieval-augmented
generation (RAG) over your own documents, and a tool-calling agent loop.

It is **not** a 1B model: a true 1B-parameter model cannot train on a CPU-only
laptop. Instead the default is a ~7M-parameter transformer that you can
realistically train on your PC. Scaling to ~1B is config-only (`--n_layer`,
`--n_embd`, `--vocab`), but that size needs a GPU.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Train on your CPU

```bash
python train.py --data samples --epochs 3 --grad_accum 4
```

- Drop any `.txt` / `.md` files into a folder and point `--data` at it.
- Defaults are sized for weak CPUs; lower `--vocab 4096 --n_embd 192 --n_layer 4` to train even faster.
- Checkpoints, tokenizer, and config save to `runs/checkpoints/model*`.
- Resume with `--resume`.

## Chat (memory + RAG + agents)

```bash
python chat.py
```

## Web chat platform (HTML + CSS + JavaScript + Python)

```bash
python server.py
```

Then open http://localhost:7860 in your browser. The page streams each reply token
by token (Python streams newline-delimited JSON over `POST /api/chat/stream`) and
shows agent tool usage as chips. Drop any `.txt`/`.md` files into `knowledge/` and
ask about them. The assistant responds in English text — no TTS, nothing installed.

| Command (in the chat box) | Action |
|---|---|
| `/remember <text>` | store a fact permanently |
| `/tools` | list available agent tools |
| `/stats` | memory size |
| `/new`, `/help` | clear window / help |

API: `POST /api/chat/stream` `{"message":"hi"}` → streaming NDJSON events
(`text`, `tool`, `done`), `POST /api/remember`, `GET /api/stats`.

## Knowledge base (RAG)

Put documents in the `knowledge/` folder (any `.txt`/`.md`). They are chunked
and indexed on first chat, then retrieved by similarity for every question.

## Agent tools

The model can emit `TOOL: name {"arg": "value"}` lines to call:
`search_memory`, `search_knowledge`, `remember`, `calculator`,
`current_time`, `write_note`. Register your own tools via `ToolRegistry`.

## Deploy on a server (Belmo / any PaaS)

Push this repo to GitHub and deploy from there. The 100 MB+ binaries
cannot be committed (GitHub limit), so the server downloads them **once**
on first boot automatically — just make sure it has network access:

- The GGUF brain (`SmolLM2-360M-Instruct-Q4_K_M.gguf`, ~271 MB) is
  downloaded to `models/` if missing.
- The runtime is downloaded for the server's OS if missing: Linux gets
  llama.cpp's plain-ELF `llama-server` (container-safe), Windows/macOS get
  the llamafile binary.

### Model provider for Vertex AI (recommended)

For serving the model to an orchestrator (e.g. Vertex AI Platform), run the
OpenAI-compatible **provider** (no web UI, no memory/RAG/tools, tiny memory
footprint — fits Belmo's 512 MB free tier):

- App entry point: `python provider.py`, Python `3.12` (`runtime.txt`).
- Set env `API_KEY=<secret>` in the platform dashboard (called `API_KEY`).
- Endpoints: `POST /v1/chat/completions`, `GET /v1/models` (Bearer auth),
  `GET /health` (no auth). Streaming (`"stream": true`) is passed through.
- Model id to use in Vertex: `SmolLM2-360M-Instruct`.
- The provider binds `0.0.0.0` on `$PORT` when `PORT` is set. Only `numpy`
  is pip-installed (`torch` is NOT required, keeping the build small).
- Generate a key locally with `python provider.py --gen-key`.
- The HTTP server starts instantly and loads the 271 MB model in the
  background, so a PaaS health check (Belmo probes within seconds of start)
  passes even on a cold first boot. `/health` reports
  `{"status":"loading"...}` until the brain is ready. A single proxied
  request may run up to `PROVIDER_REQUEST_TIMEOUT` seconds (default 3600;
  raise it on very slow hosts).

### Deploy on Belmo (belmo.io, free tier: 0.5 vCPU / 512 MB)

1. Push this repo to GitHub (`main`).
2. Sign up at belmo.io → dashboard, connect GitHub (installs the Belmo app
   on the repo), and create a **New Service → API**.
3. Pick the `little-brain` repo, branch `main`. Python is auto-detected
   (`requirements.txt` + `runtime.txt` → Python 3.12). The `Procfile`
   (`web: python provider.py`) is the start command; if your console asks
   for one, enter `python provider.py`.
4. Add the env var `API_KEY=<your-key>` (generate one locally with
   `python provider.py --gen-key`).
5. Click **Deploy**. First boot downloads the GGUF + server binary to
   `models/`; `/health` keeps the service alive during that window.
   State (API keys, usage DB) lives in a writable dir — set
   `LIGHTBRAIN_DATA_DIR=/persistent/path` if you have a persistent disk
   (default falls back to /tmp on read-only hosts).
   Vertex uses model id `SmolLM2-360M-Instruct`.

   **Belmo free tier note:** the Starter plan has *no* persistent storage —
   `/app` is read-only and `/tmp` is a small filesystem wiped each cold
   restart. To fit comfortably:
   - set `LIGHTBRAIN_MODEL=SmolLM2-135M-Instruct` to serve the ~100 MB
     SmolLM2-135M instead of the ~271 MB 360M; or
   - upgrade to Pro ($35/mo, 40 GB NVMe) and point `LIGHTBRAIN_DATA_DIR` at
     it so the model downloads once.
   Your URL is `https://<service>.app.belmo.io`.
6. Point an OpenAI-compatible client at
   `https://<service>.app.belmo.io/v1` with that `API_KEY`.

Local test run: `run_provider.bat [key]` (uses a generated key if omitted).

### Chat web server (optional)

Run `server.py` for the browser UI (`127.0.0.1:7860`, `0.0.0.0:$PORT`
when `PORT` is set). Requires `numpy` only.

## Project layout

```
llm/model.py        GPT transformer (trainable + generate + hidden embeddings)
llm/tokenizer.py    word-level tokenizer, trained on your corpus
llm/train_cmd.py    CPU training loop w/ checkpoints + resume
llm/embeddings.py   model embeddings or lightweight TF-IDF fallback
llm/rag.py          document chunking + numpy vector store
llm/memory.py       persistent transcript + capped long-term memory
llm/agents.py       context builder + tool-calling loop (stream-aware)
llm/tools.py        tool registry + built-ins (safe calculator, etc.)
llm/chat_cmd.py     interactive chat REPL (also "load_backend" for reuse)
server.py           stdlib HTTP server: web UI + streaming chat API
provider.py         OpenAI-compatible model provider (Vertex-ready, API key)
web/                index.html, style.css, app.js (streaming + speech)
```