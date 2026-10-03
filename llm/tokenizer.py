"""Trainable word-level tokenizer built from a corpus."""
import json
import re
from collections import Counter
from pathlib import Path

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"^\w+$", re.UNICODE)

DEFAULT_SPECIAL_TOKENS = [
    "<pad>", "<unk>",
    "<s>", "</s>",
    "<sys>", "</sys>",
    "<user>", "</user>",
    "<assistant>", "</assistant>",
    "<mem>", "</mem>",
    "<knowledge>", "</knowledge>",
    "<tool_result>", "</tool_result>",
    "<end>",
]

_SPECIAL_TO_SYMBOL = {
    name: name.replace("<", "▁").replace(">", "") for name in DEFAULT_SPECIAL_TOKENS
}


class Tokenizer:
    def __init__(self, special_tokens=None):
        self.special_tokens = list(special_tokens or DEFAULT_SPECIAL_TOKENS)
        self.id_to_token = list(self.special_tokens)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.casefold = True

    @property
    def vocab_size(self):
        return len(self.id_to_token)

    def add_token(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    def tokenize_text(self, text):
        """Split text keeping special <tags> as single atomic tokens."""
        if not self.special_tokens:
            return _TOKEN_RE.findall(text)
        specials = set(self.special_tokens)
        parts = re.split(
            "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")",
            text)
        out = []
        for part in parts:
            if part in specials:
                out.append(part)
            elif part:
                out.extend(_TOKEN_RE.findall(part))
        return out

    def encode(self, text):
        ids = []
        for tok in self.tokenize_text(text):
            if self.casefold:
                tok = tok.lower()
            ids.append(self.token_to_id.get(tok, self.token_to_id["<unk>"]))
        return ids

    def decode(self, ids):
        return "".join(
            self.id_to_token[i] if i < len(self.id_to_token) else "<unk>" for i in ids
        )

    def join_text(self, ids):
        """Token ids -> readable text (special tokens stay as ▁ symbols)."""
        out = ""
        for i in ids:
            if i >= len(self.id_to_token):
                continue
            tok = self.id_to_token[i]
            if tok in self.special_tokens:
                if out and not out.endswith(" "):
                    out += " "
                out += _SPECIAL_TO_SYMBOL[tok]
            elif _WORD_RE.fullmatch(tok):
                if out and not out.endswith(" "):
                    out += " "
                out += tok
            else:
                out += tok
        return out.strip()

    def decode_text(self, ids):
        out = self.join_text(ids)
        for name in self.special_tokens:
            out = out.replace(_SPECIAL_TO_SYMBOL[name], name)
        return out.strip()

    def train(self, texts, vocab_size, min_count=1, progress=None):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize_text(text))
        most = [w for w, c in counter.most_common(vocab_size - len(self.special_tokens))
                if c >= min_count]
        for w in most:
            self.add_token(w.lower() if self.casefold else w)
        return len(self.id_to_token)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"special_tokens": self.special_tokens,
                       "tokens": self.id_to_token}, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["special_tokens"])
        tok.id_to_token = list(data["tokens"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        return tok