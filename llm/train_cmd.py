"""CPU-friendly training: `python -m llm.train_cmd --data samples`."""
import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .data import WindowDataset, load_texts
from .model import GPT
from .tokenizer import Tokenizer

RNG = random.Random()


def build_parser():
    p = argparse.ArgumentParser(description="Train LightBrain on your text.")
    p.add_argument("--data", nargs="+", default=["samples"],
                   help="folders/files with .txt .md .py .csv .json")
    p.add_argument("--out", default="runs/checkpoints/model", help="checkpoint prefix")
    p.add_argument("--resume", action="store_true", help="resume from --out")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda/mps")
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--block", type=int, default=128, help="sequence window")
    p.add_argument("--vocab", type=int, default=8192, help="max vocab words")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=200)
    return p


def pick_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def lr_at(step, warmup, total, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if total <= warmup:
        return lr
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    args = build_parser().parse_args()
    device = pick_device(args.device)
    mcfg = ModelConfig(n_layer=args.n_layer, n_embd=args.n_embd,
                       n_head=args.n_head, block_size=args.block,
                       vocab_size=args.vocab)
    tcfg = TrainConfig(lr=args.lr, batch_size=args.batch,
                       grad_accum=args.grad_accum, epochs=args.epochs,
                       block_size=args.block, max_steps=args.max_steps,
                       log_every=args.log_every, save_every=args.save_every)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok_path = out.with_name("tokenizer.json")

    tokenizer = Tokenizer.load(tok_path) if (args.resume and tok_path.exists()) else None
    texts = load_texts(args.data)
    if not texts:
        raise SystemExit("No readable text found under " + ", ".join(args.data))
    if tokenizer is None:
        print("Building tokenizer...")
        tokenizer = Tokenizer()
        tokenizer.train(texts, mcfg.vocab_size)
        mcfg.vocab_size = tokenizer.vocab_size
        tokenizer.save(tok_path)
    else:
        mcfg.vocab_size = tokenizer.vocab_size

    dataset = WindowDataset(texts, tokenizer, mcfg.block_size, device=device)
    model = GPT(mcfg).to(device)
    print(f"vocab={mcfg.vocab_size} params={mcfg.param_count()/1e6:.1f}M "
          f"device={device} corpus_tokens={len(dataset.ids):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                                  weight_decay=tcfg.weight_decay)
    step = 0
    plan_path = out.with_name("model.plan.json")
    if args.resume and plan_path.exists():
        plan = __import__("json").load(open(plan_path, encoding="utf-8"))
        step = plan["step"]
        state = torch.load(out.with_name("model.pt"), map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        print(f"Resumed from step {step}")

    steps_per_epoch = len(dataset) // (tcfg.batch_size * tcfg.grad_accum)
    total = args.max_steps or max(10, steps_per_epoch * tcfg.epochs)
    print(f"steps_per_epoch={steps_per_epoch} total_steps={total}")

    model.train()
    start = time.time()
    for step in range(step, total + 1):
        loss_acc = 0.0
        optimizer.zero_grad()
        for k in range(tcfg.grad_accum):
            x, y = dataset.batch(tcfg.batch_size, RNG)
            _, loss = model(x, y)
            (loss / tcfg.grad_accum).backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, tcfg.warmup_steps, total, tcfg.lr, tcfg.min_lr)
        optimizer.step()

        if step % tcfg.log_every == 0:
            sps = (tcfg.batch_size * tcfg.grad_accum * tcfg.log_every) / max(1e-6, time.time() - start)
            print(f"step {step:>6}/{total} loss {loss_acc/tcfg.grad_accum:.4f} "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} {sps:.0f} tok/s",
                  flush=True)
            start = time.time()
        if step % tcfg.save_every == 0:
            save_checkpoint(model, optimizer, step, out, plan_path, mcfg, device)
    save_checkpoint(model, optimizer, step, out, plan_path, mcfg, device)
    sample(model, tokenizer, device, count=3)
    print("Done. Run: python -m llm.chat_cmd")


def save_checkpoint(model, optimizer, step, out, plan_path, mcfg, device):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()},
               out.with_name("model.pt"))
    mcfg.save(out.with_name("model.json"))
    with open(plan_path, "w", encoding="utf-8") as f:
        __import__("json").dump({"step": step}, f)
    print(f"checkpoint saved -> {out.with_name('model.pt')}")


@torch.no_grad()
def sample(model, tokenizer, device, count=3):
    model.eval()
    for i in range(count):
        seed = ["Once upon a time", "The best thing about", "To learn English,"]
        ids = tokenizer.encode(seed[i % 3])
        start = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(start, 60, temperature=0.9, top_k=40)
        print(f"\n[Sample {i + 1}]\n{tokenizer.decode_text(out[0][len(ids):].tolist())}")
    model.train()


if __name__ == "__main__":
    main()