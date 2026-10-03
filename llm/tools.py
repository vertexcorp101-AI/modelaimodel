"""Agent tools: registry + built-ins (safe calculator, memory, retrieval, notes)."""
import ast
import operator as op
import time
from pathlib import Path

SAFE_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos,
}
SAFE_FUNCS = {"abs": abs, "round": round, "min": min, "max": max}

OPS = set()


def safe_calc(expr):
    """Evaluate arithmetic without executing arbitrary Python."""
    tree = ast.parse(expr, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in SAFE_FUNCS:
            return SAFE_FUNCS[node.func.id](*[visit(a) for a in node.args])
        raise ValueError(f"unsupported expression: {ast.dump(node)}")

    return visit(tree)


class Tool:
    def __init__(self, name, description, func, args_hint):
        self.name = name
        self.description = description
        self.func = func
        self.args_hint = args_hint


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, description, func, args_hint=""):
        self.tools[name] = Tool(name, description, func, args_hint)
        return self

    def specs_text(self):
        lines = []
        for t in self.tools.values():
            hint = f'  args: {t.args_hint}' if t.args_hint else ""
            lines.append(f"- {t.name}: {t.description}{hint}")
        return "\n".join(lines)

    def call(self, name, args, ctx):
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = tool.func(ctx, **{k: v for k, v in (args or {}).items()})
        except Exception as exc:
            return f"Error: {exc}"
        return str(result)[:4000]


def build_default_registry():
    reg = ToolRegistry()

    def search_memory(ctx, query=""):
        hits = ctx["memory"].recall(query, k=3)
        if not hits:
            return "No stored memories match."
        return "\n".join(f"{s:.2f} | {t}" for t, s, _ in hits)

    def search_knowledge(ctx, query=""):
        hits = ctx["knowledge"].store.search(ctx["embedder"].embed(query), k=3)
        if not hits:
            return "No knowledge matches."
        return "\n".join(f"{s:.2f} | {t}" for t, s, _ in hits)

    def remember(ctx, text=""):
        return "saved" if ctx["memory"].remember(text, source="tool") else "failed"

    def calculator(ctx, expression=""):
        return repr(safe_calc(expression))

    def current_time(ctx, ):
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def write_note(ctx, title="", text=""):
        p = Path("notes") / f"{title or 'note'}.txt"
        p.parent.mkdir(exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return f"note saved to {p}"

    reg.register("search_memory", "Search long-term personal memories", search_memory,
                 '{"query": "..."}')
    reg.register("search_knowledge", "Search the knowledge folder documents",
                 search_knowledge, '{"query": "..."}')
    reg.register("remember", "Store an important fact permanently", remember,
                 '{"text": "..."}')
    reg.register("calculator", "Evaluate arithmetic", calculator,
                 '{"expression": "2+2"}')
    reg.register("current_time", "Current date/time", current_time, "")
    reg.register("write_note", "Save a text note", write_note,
                 '{"title": "...", "text": "..."}')
    return reg