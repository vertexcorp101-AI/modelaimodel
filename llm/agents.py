"""The agent: context building (memory+RAG) + tool-calling response loop."""
import json
import re

import torch

TOOL_RE = re.compile(r"TOOL:\s*(\w+)\s*(\{.*?\})?", re.DOTALL)

# The model occasionally appends the <sys>/<mem> prompt text to its answer.
# Anything at/after this marker is deemed echoed prompt, not real speech.
SYS_MARK = "you are tiny-vertex"


def cut_prompt_echo(text):
    """Trim a reply at the first occurrence of the echoed system prompt."""
    i = text.find(SYS_MARK)
    if i < 0:
        return text.strip()
    return text[:i].strip()

# Must match the <sys> string used in samples/make_corpus.py exactly.
SYSTEM_PROMPT = ("you are tiny-Vertex, the user's close friend, not an assistant. "
                 "text like a friend texts: 1-2 short casual sentences, warm and comforting. "
                 "never lecture, never use formal phrases. "
                 "use the user's name once you know it. comfort them when they're down.")


class ToolRefusal(Exception):
    pass


def extract_tool_call(text):
    """Return (name, args, text_before_call, match_end) for the last tool call."""
    if not text:
        return None
    matches = list(TOOL_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    name = m.group(1)
    args = {}
    if m.group(2):
        try:
            args = json.loads(m.group(2))
        except ValueError:
            return None
    return name, args, text[:m.start()].rstrip(), m.end()


class Agent:
    def __init__(self, model, tokenizer, embedder, registry, knowledge, memory,
                 device="cpu", max_tokens=150, max_tools=4,
                 include_tools_prompt=False):
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.registry = registry
        self.knowledge = knowledge
        self.memory = memory
        self.device = device
        self.max_tokens = max_tokens
        self.max_tools = max_tools
        self.include_tools_prompt = include_tools_prompt
        self.tool_names = set(registry.tools)

    # -- context --------------------------------------------------------
    def system_prompt(self):
        extra = ("\n" + self.registry.specs_text()) if self.include_tools_prompt else ""
        return SYSTEM_PROMPT + extra

    def _push(self, parts, section, budget):
        """Add a section to the prompt, keeping everything under 'budget' tokens."""
        if not section.strip():
            return budget
        text = section if isinstance(section, str) else str(section)
        n = len(self.tokenizer.encode(text))
        remaining = budget - sum(len(self.tokenizer.encode(p)) for p in parts)
        if n > remaining:
            return budget
        parts.append(text)
        return budget - n

    def build_prompt(self, user_text):
        budget = self.model.cfg.block_size - 12
        parts = []
        self._push(parts, "<sys>" + self.system_prompt() + "</sys>", budget)

        mem_hits = self.memory.recall(user_text, k=1)
        if mem_hits:
            mem_txt = "<mem>" + "\n".join(t[:120] for t, _, _ in mem_hits) + "</mem>"
            self._push(parts, mem_txt, budget)

        if self.knowledge is not None:
            kn_hits = self.knowledge.store.search(self.embedder.embed(user_text), k=1)
            if kn_hits:
                kn_txt = "<knowledge>" + "\n".join(t[:120] for t, _, _ in kn_hits) + "</knowledge>"
                self._push(parts, kn_txt, budget)

        transcript = self._format_transcript(self.memory.recent_turns(2))
        if transcript:
            self._push(parts, transcript, budget)

        self._push(parts, f"<user>{user_text}</user>\n<assistant>", budget)
        return "\n".join(parts)

    @staticmethod
    def _format_transcript(turns):
        s = ""
        for t in turns:
            role = t.get("role")
            if role == "user":
                s += f"<user>{t['text']}</user>\n"
            elif role == "assistant":
                s += f"<assistant>{t['text']}</assistant>\n"
        return s.strip()

    # -- generation -------------------------------------------------------
    @torch.no_grad()
    def _generate(self, prompt):
        ids = self.tokenizer.encode(prompt)
        ids = ids[-(self.model.cfg.block_size - 4):]
        start = torch.tensor([ids], dtype=torch.long, device=self.device)
        user_id = self.tokenizer.token_to_id.get("<user>")
        out_tokens = []
        for tok in self.model.generate_stream(start, self.max_tokens,
                                              temperature=0.2, top_k=4):
            if user_id is not None and tok == user_id:
                break
            out_tokens.append(tok)
        return self.tokenizer.decode_text(out_tokens)

    def respond(self, user_text, verbose=False):
        prompt = self.build_prompt(user_text)
        assistant_part = ""
        tools_used = []
        ctx = self._ctx()
        for _ in range(self.max_tools + 1):
            segment = prompt + assistant_part
            generated = cut_prompt_echo(self._generate(segment) or "")
            if verbose:
                print(f"\n[debug] generated: {generated!r}")
            call = extract_tool_call(generated)
            if call is None:
                reply = cut_prompt_echo(assistant_part + generated)
                if not reply or reply == "<assistant>":
                    reply = "I'm not sure how to answer that yet."
                return reply, tools_used
            name, args, prefix, _ = call
            if name not in self.tool_names:
                return (assistant_part + generated).strip(), tools_used
            observation = self.registry.call(name, args, ctx)
            assistant_part += (prefix + f"\nTOOL: {name} "
                               + (json.dumps(args) if args else "{}")
                               + f"\n<tool_result>{observation}</tool_result>\n")
            tools_used.append((name, args))
        return cut_prompt_echo(assistant_part + generated), tools_used

    def _ctx(self):
        return {"memory": self.memory, "knowledge": self.knowledge,
                "embedder": self.embedder}

    def stream_respond(self, user_text, verbose=False):
        """Yield NDJSON-ready events while generating: text deltas and tools.

        Events: {"type":"text","text":...} | {"type":"tool","name":...,"args":...}
        finally {"type":"done","text":full_reply,"tools":[...]}
        """
        prompt = self.build_prompt(user_text)
        assistant_part = ""
        tools_used = []
        ctx = self._ctx()
        for _ in range(self.max_tools + 1):
            segment = prompt + assistant_part
            ids = self.tokenizer.encode(segment)
            ids = ids[-(self.model.cfg.block_size - 4):]
            start = torch.tensor([ids], dtype=torch.long, device=self.device)
            buf = list(ids)
            p_len = len(self.tokenizer.join_text(buf))
            prev = self.tokenizer.join_text(buf)
            gen_ids = []
            generated = ""
            used_tool = False
            stopped = False
            user_id = self.tokenizer.token_to_id.get("<user>")
            for tok in self.model.generate_stream(start, self.max_tokens,
                                                  temperature=0.2, top_k=4):
                if user_id is not None and tok == user_id:
                    break
                buf.append(tok)
                gen_ids.append(tok)
                s = self.tokenizer.join_text(buf)
                delta = s[len(prev):] if s.startswith(prev) else s
                prev = s
                generated = self.tokenizer.join_text(gen_ids)
                call = extract_tool_call(generated)
                if call is not None:
                    name, args, prefix, end = call
                    if not generated[end:].strip() and name in self.tool_names:
                        observation = self.registry.call(name, args, ctx)
                        assistant_part += (prefix + f"\nTOOL: {name} "
                                           + (json.dumps(args) if args else "{}")
                                           + f"\n<tool_result>{observation}</tool_result>\n")
                        tools_used.append((name, args))
                        used_tool = True
                        yield {"type": "tool", "name": name, "args": args}
                        buf = list(self.tokenizer.encode(segment + assistant_part))
                        prev = self.tokenizer.join_text(buf)
                        gen_ids = []
                        generated = ""
                        break
                    continue
                mark = generated.find(SYS_MARK)
                if mark >= 0:
                    cut = p_len + mark - len(prev)
                    if 0 < cut < len(delta):
                        delta = delta[:cut]
                    elif cut >= 0:
                        delta = ""
                    if delta:
                        yield {"type": "text", "text": delta}
                    stopped = True
                    break
                if delta:
                    yield {"type": "text", "text": delta}
            if stopped:
                reply = cut_prompt_echo(assistant_part + generated) or \
                    "I'm not sure how to answer that yet."
                yield {"type": "done", "text": reply, "tools": tools_used}
                return
            if not used_tool:
                reply = cut_prompt_echo(assistant_part + generated) or \
                    "I'm not sure how to answer that yet."
                yield {"type": "done", "text": reply, "tools": tools_used}
                return
        yield {"type": "done", "text": cut_prompt_echo(assistant_part + generated)
               or "I'm not sure how to answer that yet.", "tools": tools_used}

