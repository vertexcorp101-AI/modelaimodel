"""OpenAI-compatible model provider for LightBrain.

Serves the local SmolLM2-360M GGUF (via llamafile) to ANY caller to a standard
OpenAI endpoint, so Vertex AI Platform can orchestrate/forward to it.

Endpoints:
  POST /v1/chat/completions     (Bearer key)  proxy to llamafile
  GET  /v1/models               (Bearer key)  list the served model
  GET  /health                  (open)        liveness probe
  GET  /admin ...               (admin)       dashboard (login + side nav)

Auth:
  * Model keys: created in the dashboard (or seeded from env API_KEY).
    Any active key works on /v1/*. Stored hashed (sha256) in state/provider.db.
  * Admin: ADMIN_USERNAME / ADMIN_PASSWORD env (defaults: 'admin' / random
    printed at first boot). Session via HttpOnly cookie.

Run:  python provider.py                 (localhost:3000)
      python provider.py --port 9000
"""
import argparse
import atexit
import datetime as _dt
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm.gguf_backend import (
    DATA_ROOT, LlamaServer, detect_gpu_layers, find_gguf, find_llamafile,
)
from llm.runtime import ensure_runtime

MODEL_ID = os.environ.get("LIGHTBRAIN_MODEL", "Qwen2.5-3B-Instruct")
MAX_TOKENS = 512  # client max_tokens is clamped to this (shared vCPU safety)
# llama.cpp context. Must be >= the whole prompt: vertex_system_prompt.txt alone
# is ~8.9k tokens, before history, RAG and tool schemas. 32768 is Qwen2.5-3B's
# native max_position_embeddings; on a T4 it costs 36 KB/token = 1.21 GB of KV
# cache, which fits alongside the 1.96 GB weights. Raise via PROVIDER_CTX.
CTX_TOKENS = int(os.environ.get("PROVIDER_CTX", "32768"))
# How long one proxied request may run. Slow CPU hosts (Belmo 0.5 vCPU) can
# need many minutes to eval a big prompt; raise via PROVIDER_REQUEST_TIMEOUT.
REQUEST_TIMEOUT = int(os.environ.get("PROVIDER_REQUEST_TIMEOUT", "3600"))
HTML_ESCAPE = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})

def _pick_state_dir():
    """Writable location for the SQLite state. Belmo etc. mount the app dir
    (/app) read-only, so keep state next to the (already writable) models
    dir, or honour an explicit override."""
    env = os.environ.get("LIGHTBRAIN_STATE_DIR")
    if env:
        return Path(env)
    return DATA_ROOT / "state"


STATE_DIR = _pick_state_dir()
DB_PATH = STATE_DIR / "provider.db"
ADMIN_USER = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASSWORD") or secrets.token_urlsafe(12)
SESSION_TTL_S = int(os.environ.get("ADMIN_SESSION_TTL", "43200"))

_BOOT_TS = time.time()
_SESSIONS = {}  # token -> expiry (in-memory; fine for one provider process)


def _now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _db():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    # no WAL: on tiny container disks (Belmo tmpfs) WAL files can trip
    # disk-I/O errors; the default delete journal + busy_timeout is enough
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA synchronous=OFF")
    return con


def _init_db():
    con = _db()
    try:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                key_prefix TEXT NOT NULL,
                key_suffix TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used_at TEXT
            );
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                key_prefix TEXT,
                model TEXT,
                path TEXT,
                status INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                latency_ms INTEGER,
                error TEXT
            );
            """
        )
        con.commit()
    finally:
        con.close()


def _seed_env_key():
    """Make env API_KEY work on /v1 AND visible in the dashboard."""
    key = (os.environ.get("API_KEY") or os.environ.get("LIGHTBRAIN_API_KEY")
           or "").strip()
    if not key:
        return None
    con = _db()
    try:
        row = con.execute(
            "SELECT id FROM api_keys WHERE key_hash = ?", (sha256hex(key),)
        ).fetchone()
        if not row:
            con.execute(
                "INSERT INTO api_keys (name, key_hash, key_prefix, key_suffix,"
                " active, created_at) VALUES (?, ?, ?, ?, 1, ?)",
                ("default", sha256hex(key), key[:12], key[-4:], _now_iso()),
            )
            con.commit()
    finally:
        con.close()
    return key


def sha256hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _key_ok(token):
    if not token:
        return False
    h = sha256hex(token)
    con = _db()
    try:
        row = con.execute(
            "SELECT id, key_prefix FROM api_keys WHERE key_hash = ? AND active = 1",
            (h,),
        ).fetchone()
        if row:
            con.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                (_now_iso(), row["id"]),
            )
            con.commit()
            return True
    finally:
        con.close()
    return False


def _record_request(key_prefix, model, path, status, ptok, ctok, ms, err):
    try:
        con = _db()
        try:
            con.execute(
                "INSERT INTO requests (ts, key_prefix, model, path, status,"
                " prompt_tokens, completion_tokens, latency_ms, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_now_iso(), key_prefix, model, path, status, ptok, ctok,
                 ms, (err or "")[:300]),
            )
            con.commit()
        finally:
            con.close()
    except Exception:
        pass


def _admin_session_ok(token):
    exp = _SESSIONS.get(token or "")
    return bool(exp and exp > time.time())


def _new_session():
    tok = secrets.token_urlsafe(32)
    _SESSIONS[tok] = time.time() + SESSION_TTL_S
    return tok


class ProviderHandler(BaseHTTPRequestHandler):
    server_version = "LightBrainProvider/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[provider] {fmt % args}", flush=True)

    # ---- routing ---------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self.index_page()
        elif path == "/health":
            ready = bool(getattr(self.server, "ready", False))
            self.send_json(200, {"status": "ready" if ready else "loading",
                                 "model": MODEL_ID})
        elif path == "/v1/models":
            if not self.authorized():
                return
            self.send_json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": 0,
                 "owned_by": "lightbrain", "permission": [], "root": MODEL_ID}]})
        elif path.startswith("/admin/api/"):
            try:
                self.admin_api_get(path)
            except Exception:
                self.send_json(503, {"ok": False, "error": "storage error"})
        elif path.startswith("/admin"):
            self.serve_dashboard()
        else:
            self.send_error_json(404, "not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            if not self.authorized():
                return
            self.proxy_chat()
        elif path.startswith("/admin/api/"):
            try:
                self.admin_api_post(path)
            except Exception:
                self.send_json(503, {"ok": False, "error": "storage error"})
        elif path.startswith("/admin"):
            self.send_json(404, {"ok": False, "error": "not found"})
        else:
            self.send_error_json(404, "not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers",
                         "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ---- auth ------------------------------------------------------------

    def authorized(self):
        auth = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        if token and (_key_ok(token) or
                      (self.server.api_key and hmac.compare_digest(
                          token, self.server.api_key))):
            return True
        self.send_error_json(401, "invalid api key")
        return False

    def _session_cookie(self):
        raw = self.headers.get("Cookie", "")
        for part in raw.split(";"):
            k, _, v = part.strip().partition("=")
            if k == "lb_admin":
                return v
        return None

    def _require_admin(self):
        if _admin_session_ok(self._session_cookie()):
            return True
        self.send_json(401, {"ok": False, "error": "authentication_required",
                             "login_url": "/admin"})
        return False

    # ---- dashboard pages / api -------------------------------------------

    def serve_dashboard(self):
        html = Path(__file__).resolve().parent / "web" / "admin.html"
        if not html.is_file():
            self.send_error_json(500, "admin.html missing")
            return
        data = html.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            return None

    def admin_api_get(self, path):
        if not self._require_admin():
            return
        if path == "/admin/api/session":
            self.send_json(200, {"ok": True, "user": ADMIN_USER})
        elif path == "/admin/api/status":
            self.send_json(200, self.status_payload())
        elif path == "/admin/api/keys":
            con = _db()
            try:
                rows = con.execute(
                    "SELECT id, name, key_prefix, key_suffix, active,"
                    " created_at, last_used_at FROM api_keys"
                    " ORDER BY id DESC"
                ).fetchall()
            finally:
                con.close()
            self.send_json(200, {"ok": True, "keys": [dict(r) for r in rows]})
        elif path == "/admin/api/usage":
            con = _db()
            try:
                rows = con.execute(
                    "SELECT id, ts, key_prefix, model, path, status,"
                    " prompt_tokens, completion_tokens, latency_ms, error"
                    " FROM requests ORDER BY id DESC LIMIT 50"
                ).fetchall()
                agg = con.execute(
                    "SELECT COUNT(*) n, SUM(prompt_tokens) pt,"
                    " SUM(completion_tokens) ct,"
                    " SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) errs"
                    " FROM requests"
                ).fetchone()
            finally:
                con.close()
            self.send_json(200, {"ok": True, "requests": [dict(r) for r in rows],
                                 "agg": dict(agg) or {}})
        else:
            self.send_json(404, {"ok": False, "error": "not found"})

    def admin_api_post(self, path):
        if path == "/admin/api/login":
            self.admin_login()
            return
        if not self._require_admin():
            return
        body = self._body()
        if body is None:
            self.send_json(400, {"ok": False, "error": "invalid json"})
            return
        if path == "/admin/api/logout":
            tok = self._session_cookie()
            if tok and tok in _SESSIONS:
                del _SESSIONS[tok]
            self.send_json(200, {"ok": True})
            return
        if path == "/admin/api/keys":
            self.admin_create_key(body)
            return
        if path.startswith("/admin/api/test"):
            self.admin_test(body)
            return
        parts = path.split("/")
        if len(parts) >= 5 and parts[3] == "keys":
            kid = parts[4]
            action = parts[5] if len(parts) > 5 else ""
            if action == "revoke":
                self.admin_set_key(kid, 0)
            elif action == "activate":
                self.admin_set_key(kid, 1)
            elif action == "delete":
                self.admin_delete_key(kid)
            else:
                self.send_json(404, {"ok": False, "error": "not found"})
            return
        self.send_json(404, {"ok": False, "error": "not found"})

    def admin_login(self):
        body = self._body() or {}
        u = str(body.get("username") or "")
        p = str(body.get("password") or "")
        if (hmac.compare_digest(u, ADMIN_USER)
                and hmac.compare_digest(p, ADMIN_PASS)):
            tok = _new_session()
            payload = json.dumps({"ok": True, "user": ADMIN_USER}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            # single send_response: a second one would emit a duplicate
            # status line and browsers would drop the session cookie
            self.send_header("Set-Cookie",
                             f"lb_admin={tok}; Path=/; HttpOnly;"
                             f" SameSite=Lax; Max-Age={SESSION_TTL_S}")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_json(401, {"ok": False, "error": "invalid credentials"})

    def admin_create_key(self, body):
        name = str(body.get("name") or "").strip()[:60] or "api key"
        key = secrets.token_urlsafe(32)
        con = _db()
        try:
            con.execute(
                "INSERT INTO api_keys (name, key_hash, key_prefix, key_suffix,"
                " active, created_at) VALUES (?, ?, ?, ?, 1, ?)",
                (name, sha256hex(key), key[:12], key[-4:], _now_iso()),
            )
            con.commit()
        finally:
            con.close()
        self.send_json(200, {"ok": True, "key": key, "name": name,
                             "key_prefix": key[:12], "key_suffix": key[-4:]})

    def admin_set_key(self, kid, active):
        try:
            kid = int(kid)
        except (TypeError, ValueError):
            self.send_json(400, {"ok": False, "error": "bad id"})
            return
        con = _db()
        try:
            con.execute("UPDATE api_keys SET active = ? WHERE id = ?",
                        (active, kid))
            con.commit()
        finally:
            con.close()
        self.send_json(200, {"ok": True, "active": bool(active)})

    def admin_delete_key(self, kid):
        try:
            kid = int(kid)
        except (TypeError, ValueError):
            self.send_json(400, {"ok": False, "error": "bad id"})
            return
        con = _db()
        try:
            con.execute("DELETE FROM api_keys WHERE id = ?", (kid,))
            con.commit()
        finally:
            con.close()
        self.send_json(200, {"ok": True})

    def status_payload(self):
        con = _db()
        try:
            n_keys = con.execute(
                "SELECT COUNT(*) n FROM api_keys WHERE active = 1"
            ).fetchone()["n"]
            n_req = con.execute("SELECT COUNT(*) n FROM requests").fetchone()["n"]
        finally:
            con.close()
        ready = bool(getattr(self.server, "ready", False))
        llama = getattr(self.server, "llama", None)
        return {
            "ok": True,
            "model": MODEL_ID,
            "ready": ready,
            "device": getattr(llama, "device", None) if llama else None,
            "gpu_layers": getattr(llama, "gpu_layers", None) if llama else None,
            "uptime_s": int(time.time() - _BOOT_TS),
            "base_url": f"http://{self.server.server_address[0]}"
                        f":{self.server.server_address[1]}/v1",
            "ctx": getattr(llama, "ctx", None) if llama else None,
            "gguf": str(getattr(llama, "gguf", "") or ""),
            "active_keys": n_keys,
            "total_requests": n_req,
            "boot_error": getattr(self.server, "boot_error", None),
            "request_timeout_s": REQUEST_TIMEOUT,
            "max_tokens": MAX_TOKENS,
            "started_at": _dt.datetime.fromtimestamp(
                _BOOT_TS).isoformat(timespec="seconds"),
        }

    def admin_test(self, body):
        if not getattr(self.server, "ready", False) or not self.server.llama_url:
            self.send_json(503, {"ok": False,
                                 "error": "model still loading, retry shortly"})
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self.send_json(400, {"ok": False, "error": "messages required"})
            return
        max_tokens = min(int(body.get("max_tokens") or 128), MAX_TOKENS)
        data = {"model": MODEL_ID, "messages": messages,
                "max_tokens": max_tokens, "stream": False,
                "temperature": 0.7, "top_p": 0.95}
        req = urllib.request.Request(
            self.server.llama_url + "/v1/chat/completions",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
            payload = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "replace")
            _record_request(None, MODEL_ID, "/admin/api/test", e.code,
                            None, None, int((time.time() - t0) * 1000), err)
            self.send_json(e.code, {"ok": False, "error": err[:300]})
            return
        except Exception as e:
            _record_request(None, MODEL_ID, "/admin/api/test", 503,
                            None, None, int((time.time() - t0) * 1000), str(e))
            self.send_json(503, {"ok": False, "error": str(e)})
            return
        ms = int((time.time() - t0) * 1000)
        usage = payload.get("usage") or {}
        try:
            reply = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            reply = ""
        _record_request(None, MODEL_ID, "/admin/api/test", 200,
                        usage.get("prompt_tokens"), usage.get("completion_tokens"),
                        ms, None)
        self.send_json(200, {"ok": True, "reply": reply, "usage": usage,
                             "latency_ms": ms, "model": MODEL_ID})

    # ---- openai proxy ----------------------------------------------------

    def index_page(self):
        """Small status page: base URL + model + API key (dev only)."""
        host, port = self.server.server_address[:2]
        base = f"http://{host}:{port}/v1"
        if self.server.show_key:
            key = self.server.api_key
            key_line = (f"<tr><td>API key</td><td><input readonly "
                        f"class=\"kk\" value=\"{key}\"></td></tr>")
        else:
            key_line = ("<tr><td>API key</td><td>kept secret "
                        "(set via API_KEY env)</td></tr>")
        dash = (f"<p><a href=\"/admin\">Open the admin dashboard &rarr;</a></p>")
        body = f"""<!doctype html><html><head><meta charset="utf-8"><title>
LightBrain provider</title><style>
body{{font-family:system-ui,max-width:44rem;margin:3rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}}
.kk{{width:100%;font-family:monospace}}
a{{color:#06c}}</style></head><body>
<h1>LightBrain provider</h1>
{dash}
<p>OpenAI-compatible endpoint serving <b>{MODEL_ID}</b> locally
via llamafile (no torch). Point a client like Vertex AI Platform at:</p>
<pre>  POST {base}/chat/completions   (Bearer key)
  GET  {base}/models             (Bearer key)
  GET  /health                   (no key)</pre>
<table>
<tr><th>Key</th><th>Value</th></tr>
<tr><td>Base URL</td><td><code>{base}</code></td></tr>
<tr><td>Model id</td><td><code>{MODEL_ID}</code></td></tr>
{key_line}
</table>
</body></html>"""
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, code, message):
        self.send_json(code, {"error": {
            "message": message, "type": "invalid_request_error",
            "code": "invalid_request"}})

    def proxy_chat(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw)
        except ValueError:
            self.send_error_json(400, "invalid json body")
            return
        if not isinstance(data, dict):
            self.send_error_json(400, "body must be a json object")
            return

        stream = bool(data.get("stream"))
        # Always advertise the alias llama.cpp was started with, whatever the
        # client asked for. setdefault() let an upstream default leak through
        # (e.g. "openrouter/free"), which llama-server rejects with a 400.
        key = MODEL_ID
        data["model"] = key
        # Clamp: the limit was documented but never enforced, so one caller
        # asking for 8k could stall a shared host for minutes.
        try:
            data["max_tokens"] = max(1, min(int(data.get("max_tokens") or MAX_TOKENS), MAX_TOKENS))
        except (TypeError, ValueError):
            data["max_tokens"] = MAX_TOKENS

        if not getattr(self.server, "ready", False) or not self.server.llama_url:
            self.send_json(503, {"error": {
                "message": "model still loading, retry shortly",
                "type": "server_error", "code": "model_not_ready"}})
            return

        auth = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        con = _db()
        try:
            row = con.execute(
                "SELECT key_prefix FROM api_keys WHERE key_hash = ? AND active = 1",
                (sha256hex(token),),
            ).fetchone()
        finally:
            con.close()
        kprefix = row["key_prefix"] if row else (
            (token or "")[:12] if token else None)

        req = urllib.request.Request(
            self.server.llama_url + "/v1/chat/completions",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
        except urllib.error.HTTPError as e:
            try:
                err = e.read().decode("utf-8", "replace")
            except Exception:
                err = "{}"
            _record_request(kprefix, key, "/v1/chat/completions", e.code,
                            None, None, int((time.time() - t0) * 1000), err)
            self.send_json(e.code, json.loads(err) if err else {"error": {}})
            return
        except Exception as e:
            _record_request(kprefix, key, "/v1/chat/completions", 503,
                            None, None, int((time.time() - t0) * 1000), str(e))
            self.send_error_json(503, "model unavailable")
            return

        ms = int((time.time() - t0) * 1000)
        ctype = resp.headers.get("Content-Type") or (
            "text/event-stream" if stream else "application/json")
        self.send_response(resp.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        ptok = ctok = None
        try:
            # Small read granule so SSE token chunks cross the tunnel as they
            # are produced instead of waiting for large buffers to fill.
            while True:
                chunk = resp.read(2048)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            resp.close()
        _record_request(kprefix, key, "/v1/chat/completions", 200,
                        ptok, ctok, ms, None)


def get_api_key():
    """Returns (key, from_env). The key is shown on the status page only when
    it was generated here (dev/local), never on a real server where API_KEY
    was explicitly configured."""
    key = (os.environ.get("API_KEY")
           or os.environ.get("LIGHTBRAIN_API_KEY") or "").strip()
    if key:
        return key, True
    key = secrets.token_urlsafe(32)
    print(f"[provider] no API_KEY set - generated key: {key}", flush=True)
    print("[provider] set API_KEY=<key> to pin a key (e.g. in Belmo env).",
          flush=True)
    return key, False


def boot_brain(args, httpd):
    """Download/load the model + llamafile IN THE BACKGROUND so the HTTP
    server (and /health) is up immediately. PaaS hosts (Belmo) health-check
    the app within seconds of process start; a 271 MB first-boot download
    must not block it."""
    print("[provider] ensuring runtime (model + llamafile)...", flush=True)
    attempt = 1
    while True:
        try:
            ensure_runtime()
            gguf = find_gguf()
            exe = find_llamafile()
            if not gguf or not exe:
                print("[provider] missing models/*.gguf or models/llamafile",
                      flush=True)
                httpd.boot_error = "model or llamafile missing"
                time.sleep(60)
                continue
            llama = LlamaServer(gguf, exe, ctx=args.ctx)
            if llama.start():
                httpd.llama = llama
                httpd.llama_url = llama.url
                httpd.ready = True
                atexit.register(llama.stop)
                print(f"[provider] brain ready: {gguf.name} "
                      f"({llama.device}, {llama.gpu_layers} layers offloaded)",
                      flush=True)
                return
            httpd.boot_error = "llama-server failed to start"
        except (Exception, SystemExit) as e:
            # ENOSPC on a cramped Belmo worker: don't die — /health keeps
            # returning "loading" so the platform won't kill-loop us, and we
            # retry until the pod lands on a worker that has disk.
            # SystemExit is listed explicitly because ensure_runtime() raises
            # it for an unknown model name, and it is a BaseException — so
            # `except Exception` used to miss it, killing this boot thread and
            # leaving /health stuck on "loading" forever.
            print(f"[provider] boot attempt {attempt} failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            httpd.boot_error = f"{type(e).__name__}: {e}"
        attempt += 1
        time.sleep(90)


def main():
    p = argparse.ArgumentParser(description="LightBrain OpenAI-compatible provider")
    p.add_argument("--host", default=os.environ.get("HOST") or
                   ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"))
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("PORT") or 3000))
    p.add_argument("--gen-key", action="store_true",
                   help="generate an API key and exit")
    p.add_argument("--ctx", type=int, default=CTX_TOKENS,
                   help="llamafile context size (KV cache RAM). Must fit the "
                        "whole prompt; defaults to PROVIDER_CTX/32768")
    args = p.parse_args()

    if args.gen_key:
        print(secrets.token_urlsafe(32))
        return

    _init_db()
    api_key, key_from_env = get_api_key()
    _seed_env_key()
    if not key_from_env:
        # store the generated dev key so it shows up in the dashboard
        try:
            con = _db()
            try:
                con.execute(
                    "INSERT OR IGNORE INTO api_keys (name, key_hash,"
                    " key_prefix, key_suffix, active, created_at)"
                    " VALUES ('dev', ?, ?, ?, 1, ?)",
                    (sha256hex(api_key), api_key[:12], api_key[-4:], _now_iso()),
                )
                con.commit()
            finally:
                con.close()
        except Exception:
            pass

    httpd = ThreadingHTTPServer((args.host, args.port), ProviderHandler)
    httpd.api_key = api_key
    httpd.llama_url = None
    httpd.ready = False
    httpd.boot_error = None
    httpd.llama = None
    httpd.show_key = not key_from_env
    httpd.daemon_threads = True
    httpd.allow_reuse_address = True

    import threading
    threading.Thread(target=boot_brain, args=(args, httpd),
                     daemon=True, name="brain-boot").start()
    print(f"[provider] listening on http://{args.host}:{args.port} "
          "(model loading in background)", flush=True)
    print(f"[provider] admin dashboard at /admin "
          f"(login: {ADMIN_USER} / "
          f"{'(set ADMIN_PASSWORD)' if os.environ.get('ADMIN_PASSWORD') else ADMIN_PASS})",
          flush=True)
    print("[provider] POST /v1/chat/completions | GET /v1/models | GET /health",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if httpd.llama is not None:
            httpd.llama.stop()


if __name__ == "__main__":
    main()