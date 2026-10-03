"""Rebuild samples/conversation.txt so every turn mirrors the runtime prompt.

Training lines therefore include the same <sys>/<mem>/<knowledge> blocks the
agent prepends at inference, which teaches the small model to handle them
instead of being confused by unseen context.

Pairs live in samples/pairs as "question :: answer" lines
(the file has no extension so the trainer never reads it).

Run:  python samples/make_corpus.py
"""
import random
from pathlib import Path

SYS = "you are lightbrain, a small friendly english assistant. keep answers short and natural."

MEMORIES = [
    "the user likes chocolate",
    "the user's favorite color is green",
    "the user wants to learn english",
    "the user likes football",
]

KNOWLEDGE = [
    "honey can stay good for hundreds of years",
    "water boils at one hundred degrees celsius",
    "the earth revolves around the sun once a year",
    "the moon orbits the earth about every twenty nine days",
    "dogs are loyal and can learn many commands",
    "sleeping about eight hours a night is best for most people",
]


def header():
    parts = []
    if random.random() < 0.7:
        parts.append("<sys>" + SYS + "</sys>")
    if random.random() < 0.25:
        parts.append("<mem>" + random.choice(MEMORIES) + "</mem>")
    if random.random() < 0.25:
        parts.append("<knowledge>" + random.choice(KNOWLEDGE) + "</knowledge>")
    return "".join(parts)


def build(pairs_path, out_path, seed=7):
    random.seed(seed)
    text = Path(pairs_path).read_text(encoding="utf-8")
    pairs = []
    for line in text.splitlines():
        if "::" in line:
            q, a = line.split("::", 1)
            pairs.append((q.strip(), a.strip()))
    if not pairs:
        raise SystemExit("no 'question :: answer' pairs found in " + str(pairs_path))
    random.shuffle(pairs)
    episodes = []
    i = 0
    while i < len(pairs):
        turns = random.randint(1, 4)
        chunk = []
        for _ in range(turns):
            if i >= len(pairs):
                break
            q, a = pairs[i]
            i += 1
            chunk.append(f"<user>{q}</user><assistant>{a}")
        episodes.append(header() + "".join(chunk))
    out = "\n\n".join(episodes) + "\n"
    Path(out_path).write_text(out, encoding="utf-8")
    print(f"wrote {len(episodes)} episodes from {len(pairs)} pairs -> {out_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    build(root / "pairs", root / "conversation.txt")