"""LightBrain web server (Python stdlib only).

Run:  python server.py          -> http://localhost:7860
Endpoints:
  GET  /                     web chat UI
  GET  /static/...           css/js assets
  POST /api/chat/stream      {message} -> streaming newline-delimited JSON
  POST /api/remember         {message} -> store a memory
  GET  /api/stats            memory statistics
"""
import argparse
import json
import mimetypes
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm.chat_cmd import load_backend, serve_commands

WEB_DIR = Path(__file__).resolve().parent / "web"
# Local use: 127.0.0.1:7860. On a PaaS (Belmo etc.) the platform sets $PORT
# and the app must bind 0.0.0.0 so its reverse proxy can reach us.
HOST = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
PORT = int(os.environ.get("PORT") or "7860")
AGENT_LOCK = threading.Lock()


def write_chunked(wfile, events):
    for ev in events:
        line = (json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8")
        wfile.write(b"%x\r\n" % len(line))
        wfile.write(line)
        wfile.write(b"\r\n")
        wfile.flush()
    wfile.write(b"0\r\n\r\n")
    wfile.flush()


class ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    app = None

    def log_message(self, fmt, *args):
        pass

    def _serve_file(self, path):
        path = path.resolve()
        root = WEB_DIR.resolve()
        if not str(path).startswith(str(root)) or not path.is_file():
            self.send_error(404)
            return
        ctype, _ = mimetypes.guess_type(str(path))
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            return {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html", "/chat"):
            self._serve_file(WEB_DIR / "index.html")
            return
        if path.startswith("/static/"):
            self._serve_file(WEB_DIR / path[len("/static/"):])
            return
        if path in ("/api/stats", "/api/health"):
            self._json(self.app["backend"]["memory"].stats())
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        backend = self.app["backend"]
        agent = backend["agent"]
        memory = backend["memory"]
        data = self._read_json()
        message = str(data.get("message", "")).strip()

        if not message:
            self._json({"type": "error", "text": "Empty message"}, 400)
            return

        if parsed.path == "/api/chat/stream":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            cmds = serve_commands(message, memory, self.app["backend"]["registry"])
            if cmds is not None:
                write_chunked(self.wfile, [{"type": "done", "text": cmds, "tools": []}])
                return
            final_reply = ""
            try:
                def events():
                    nonlocal final_reply
                    with AGENT_LOCK:
                        for ev in agent.stream_respond(message):
                            if ev.get("type") == "done":
                                final_reply = ev.get("text", "")
                            yield ev
                        memory.add_exchange(message, final_reply)
                write_chunked(self.wfile, events())
            except BrokenPipeError:
                pass
            except Exception as exc:
                try:
                    write_chunked(self.wfile, [
                        {"type": "error", "text": f"Server error: {exc}"}])
                except BrokenPipeError:
                    pass
            return

        if parsed.path == "/api/remember":
            ok = memory.remember(message, source="user")
            self._json({"ok": ok, "message": "remembered" if ok else "failed"})
            return

        if parsed.path == "/api/stats":
            self._json(memory.stats())
            return

        self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description="LightBrain web server")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--model", default="runs/checkpoints/model")
    parser.add_argument("--knowledge", default="knowledge")
    parser.add_argument("--memory", default="runs/memory")
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    from llm.runtime import ensure_runtime
    ensure_runtime()

    backend = load_backend(args.model, args.knowledge, args.memory,
                           max_tokens=args.max_tokens)
    app = {"backend": backend}
    httpd = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    ChatHandler.app = app

    print(f"\nLightBrain web chat running at  http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        httpd.shutdown()


if __name__ == "__main__":
    main()