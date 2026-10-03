# Fine-tune 360M into tiny-Vertex — Colab runbook

Goal: LoRA-tune `HuggingFaceTB/SmolLM2-360M-Instruct` on
`sft/tiny_vertex_dialogues.jsonl` (~100 short dialogues in tiny-Vertex's
voice), merge, convert to GGUF Q4_K_M, download. Free T4: ~20-40 min.

## Colab cells (in order)

**1. Install + GPU check**
```
!nvidia-smi -L
!pip install -q -U transformers peft trl accelerate datasets
```

**2. Upload data + script**
Upload these two files from this repo (Files panel → Upload):
- `sft/tiny_vertex_dialogues.jsonl`
- `sft/train_lora_colab.py`

**3. Train**
```
!python train_lora_colab.py
```
Watches: loss should fall steadily (e.g. 2.x → ~1.x). If loss explodes or
prints NaN, stop: lower `SFT_LR=1e-4` and rerun.

**4. Convert merged model → GGUF → Q4_K_M** (no compile: prebuilt binary)
```
!git clone --depth 1 https://github.com/ggml-org/llama.cpp
!pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
!python llama.cpp/convert_hf_to_gguf.py tinyvertex-360m-merged --outfile tinyvertex-360m-f16.gguf
!curl -sL -o llama-quant.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b11179/llama-b11179-bin-ubuntu-x64.tar.gz
!tar xzf llama-quant.tar.gz --wildcards '*/quantize' --strip-components=2
!./quantize tinyvertex-360m-f16.gguf tinyvertex-360m-Q4_K_M.gguf Q4_K_M
!ls -la *.gguf
```

**5. Download**
Files panel → download `tinyvertex-360m-Q4_K_M.gguf` (~270MB).
Drop it into `littlebrain/models/` as e.g. `tinyvertex-360m-Q4_K_M.gguf`,
then point the app at it (filename without `.gguf` becomes the model id).

## Keep it honest

- 100 examples teach *style* (short, warm, friend-voice), not knowledge.
  After training, re-run the same 5 probes: hey / i am sad / name /
  mom-grief / advice-seeking. Style should now come from weights, and the
  ban-guard in `llm/gguf_backend.py` becomes a safety net instead of the
  main steering.
- If style barely moves: add 100+ more dialogues in YOUR users' words
  (copy real good exchanges from `runs/memory/transcript.jsonl`), raise to
  5 epochs. Data beats hyperparameters here.
- If replies get repetitive or dumb: overfit — drop to 2 epochs, LR 1e-4.

## PC fallback (your i3, 8GB, no GPU)

Same script runs on CPU in fp32: `python train_lora_colab.py` overnight
(~4-8h for 100 examples × 3 epochs). Works, just slow. Colab first choice.
