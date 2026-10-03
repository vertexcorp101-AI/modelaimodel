import json
import sys
import urllib.request

body = json.dumps({"message": sys.argv[1] if len(sys.argv) > 1 else "Hello"}).encode()
req = urllib.request.Request("http://127.0.0.1:7860/api/chat/stream",
                             data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    stream = r.read().decode("utf-8")
for line in stream.splitlines():
    if not line.strip():
        continue
    try:
        ev = json.loads(line)
        if ev.get("type") == "text":
            print(ev.get("text", ""), end="", flush=True)
        elif ev.get("type") == "tool":
            print("\n[tool]", ev.get("name"), ev.get("args"))
        elif ev.get("type") == "done":
            print("\n[done]", ev.get("tools"))
    except ValueError:
        pass
print()