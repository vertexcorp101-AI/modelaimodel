"use strict";

const chat = document.getElementById("chat");
const msg = document.getElementById("msg");
const sendBtn = document.getElementById("send");
const meta = document.getElementById("meta");

let busy = false;
let lastReply = "";

function clean(text) {
  return (text || "")
    .replace(/<\/?(?:s|pad|end|user|assistant|sys|mem|knowledge)>/g, " ")
    .replace(/▁[a-z_]+/g, " ")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = "bubble " + role;
  if (text) el.textContent = text;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

async function send() {
  const text = msg.value.trim();
  if (!text || busy) return;

  if (text === "/new") {
    chat.innerHTML = "";
    lastReply = "";
    meta.textContent = "new session started (history stays in memory)";
    msg.value = "";
    return;
  }
  if (text === "/help") {
    bubble("assistant", "Commands: /remember <text> saves a fact, /stats shows memory,\n/tools lists agent tools, /new clears this window.");
    msg.value = "";
    return;
  }
  if (text.startsWith("/stats")) {
    const s = await api("/api/stats", { message: text });
    bubble("assistant", "long-term memory: " + s.long_term_entries + " entries\nconversation lines: " + s.transcript_lines);
    msg.value = "";
    return;
  }
  if (text.startsWith("/remember ")) {
    await api("/api/remember", { message: text.slice(10).trim() });
    bubble("assistant", "Saved to long-term memory.");
    msg.value = "";
    return;
  }

  busy = true;
  sendBtn.disabled = true;
  const userEl = bubble("user", text);
  const asEl = bubble("assistant", "");
  asEl.classList.add("caret");

  const t0 = performance.now();
  let streaming = true;
  const doneInfo = { tools: [], time: 0 };

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok || !res.body) throw new Error("HTTP " + res.status);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let ev;
        try { ev = JSON.parse(line); } catch { continue; }
        if (ev.type === "text" && ev.text) {
          asEl.textContent = clean(asEl.textContent + ev.text);
          chat.scrollTop = chat.scrollHeight;
        } else if (ev.type === "tool") {
          const chip = document.createElement("span");
          chip.className = "toolchip";
          chip.textContent = "tool: " + ev.name;
          asEl.appendChild(document.createTextNode(" "));
          asEl.appendChild(chip);
          doneInfo.tools.push(ev.name);
        } else if (ev.type === "done") {
          lastReply = ev.text || asEl.textContent;
          doneInfo.time = performance.now() - t0;
        } else if (ev.type === "error") {
          lastReply = ev.text;
        }
      }
    }
    streaming = false;
    asEl.classList.remove("caret");
    if (lastReply !== undefined && lastReply) asEl.textContent = clean(lastReply);
    meta.textContent = (doneInfo.tools.length ? "tools: " + doneInfo.tools.join(", ") + "  ·  " : "") +
      (doneInfo.time / 1000).toFixed(1) + "s  ·  reply saved to memory";
  } catch (err) {
    streaming = false;
    asEl.classList.remove("caret");
    asEl.textContent = "⚠ error: " + err.message;
  }
  busy = false;
  sendBtn.disabled = false;
  msg.value = "";
  msg.focus();
}

sendBtn.addEventListener("click", send);
msg.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });

document.querySelectorAll(".chip").forEach((c) => {
  c.addEventListener("click", () => {
    msg.value = c.dataset.cmd;
    send();
  });
});

msg.focus();