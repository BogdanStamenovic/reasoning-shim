"""OpenAI-compatible shim in front of llama-server for Qwen3.5-template thinking models
(first used for piccolo-gorgone:9b behind gpu-share).

Two jobs, both on /v1/chat/completions; every other path is passed through untouched.

1. Per-turn reasoning budget. llama-server honours a per-request `thinking_budget_tokens`
   (only while its global --reasoning-budget is -1). When the budget runs out its sampler
   forces `<budget message></think>`, so the model must leave the think block. We pick the
   budget from the shape of the turn, so trivial chat gets a small one and hard prompts
   keep the full one. A client that sets thinking_budget_tokens itself wins.

2. Stranded tool-call repair. The template opens every assistant turn with `<think>\n`;
   anything before `</think>` is parsed as reasoning. When the model writes
   `<tool_call>...</tool_call>` before closing the block, llama-server returns empty content
   and no tool_calls, and the harness takes it as a final answer. We lift the first
   complete call(s) out of reasoning_content into proper tool_calls.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

UPSTREAM = os.environ.get("SHIM_UPSTREAM", "http://127.0.0.1:11434").rstrip("/")
HOST = os.environ.get("SHIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("SHIM_PORT", "11450"))
# Budgets in tokens; -1 = unlimited. Measured in piccolo-tune, see README.
B_TRIVIAL = int(os.environ.get("SHIM_BUDGET_TRIVIAL", "96"))
B_TOOL = int(os.environ.get("SHIM_BUDGET_TOOL", "768"))
B_DEFAULT = int(os.environ.get("SHIM_BUDGET_DEFAULT", "-1"))
REPAIR = os.environ.get("SHIM_REPAIR", "1") == "1"
BUDGETS = os.environ.get("SHIM_BUDGETS", "1") == "1"

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*(?:</tool_call>|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)

GREETING = re.compile(r"^\s*(hi|hello|hey|yo|thanks?( you)?|thx|ok(ay)?|cool|nice|good (morning|night|evening)|bye|"
                      r"how are you|what'?s up|sup)\b[\s!.?,:)]*\w{0,12}[\s!.?)]*$", re.I)
HARD_HINT = re.compile(r"(prove|derive|design|algorithm|optimi[sz]e|debug|why does|step by step|integer solutions|"
                       r"complexity|refactor|architecture|```|\bdef |\bclass |function|equation|\d+\s*[-+*/^]\s*\d+.*\d)", re.I)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def classify(messages: list[dict]) -> str:
    """'tool' after a tool result, 'trivial' for small talk, else 'default'.

    Deliberately conservative: a false 'trivial' costs answer quality, a false 'default'
    only costs tokens, so anything that is not obviously small talk stays 'default'.
    """
    if not messages:
        return "default"
    last = messages[-1]
    if last.get("role") == "tool":
        return "tool"
    if last.get("role") != "user":
        return "default"
    t = _text(last.get("content")).strip()
    if t.startswith("<tool_response>"):
        return "tool"
    if len(t) <= 60 and GREETING.match(t) and not HARD_HINT.search(t):
        return "trivial"
    return "default"


# llama-server ignores reasoning_effort; clients such as Hermes send it (e.g. "none" on
# side calls like session titles, which otherwise burn ~5k reasoning tokens on 7 words).
EFFORT = {"none": 0, "minimal": 0, "low": 256, "medium": 1024}


def pick_budget(body: dict) -> int | None:
    if not BUDGETS:
        return None
    # Hermes sends thinking_budget_tokens=-1 itself on side calls; treat -1 as "unset"
    if body.get("thinking_budget_tokens", -1) != -1:
        return None
    kw = body.get("chat_template_kwargs") or {}
    if kw.get("enable_thinking") is False:
        return None
    eff = str(body.get("reasoning_effort") or "").lower()
    if eff in EFFORT:
        return EFFORT[eff]
    return {"trivial": B_TRIVIAL, "tool": B_TOOL, "default": B_DEFAULT}[classify(body.get("messages", []))]


def _coerce(val: str, schema: dict | None):
    typ = (schema or {}).get("type")
    if typ in (None, "string"):
        return val
    try:
        return json.loads(val)
    except Exception:
        return val


def extract_calls(reasoning: str, tools: list[dict] | None):
    """Return (reasoning_before_first_call, [openai tool_call dicts]) or (reasoning, [])."""
    if "<tool_call>" not in reasoning:
        return reasoning, []
    schemas = {}
    for t in tools or []:
        f = t.get("function", {})
        schemas[f.get("name")] = (f.get("parameters") or {}).get("properties", {})
    calls = []
    first = None
    for m in CALL_RE.finditer(reasoning):
        name, inner = m.group(1), m.group(2)
        if schemas and name not in schemas:
            continue
        first = m.start() if first is None else first
        props = schemas.get(name, {})
        args = {k: _coerce(v, props.get(k)) for k, v in PARAM_RE.findall(inner)}
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if not calls:
        return reasoning, []
    return reasoning[:first].rstrip(), calls


def repair_message(msg: dict, tools) -> bool:
    if not REPAIR or msg.get("tool_calls") or (msg.get("content") or "").strip():
        return False
    rc = msg.get("reasoning_content") or ""
    before, calls = extract_calls(rc, tools)
    if not calls:
        return False
    msg["reasoning_content"] = before
    msg["content"] = ""
    msg["tool_calls"] = calls
    return True


client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10))
STATS = {"requests": 0, "stranded": 0, "repaired": 0, "budget": {}, "upstream": UPSTREAM}


_DUMP_N = [0]


def _dump(req: dict, resp: dict):
    d = os.environ.get("SHIM_DUMP")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    _DUMP_N[0] += 1
    with open(os.path.join(d, f"{int(time.time())}-{_DUMP_N[0]:04d}.json"), "w") as f:
        json.dump({"request": req, "response": resp}, f)


def _log(ev: dict):
    if os.environ.get("SHIM_LOG"):
        with open(os.environ["SHIM_LOG"], "a") as f:
            f.write(json.dumps({"t": round(time.time(), 1), **ev}) + "\n")


async def chat(request: Request):
    body = await request.json()
    STATS["requests"] += 1
    budget = pick_budget(body)
    cls = classify(body.get("messages", []))
    if budget is not None:
        body["thinking_budget_tokens"] = budget
        STATS["budget"][cls] = STATS["budget"].get(cls, 0) + 1
    tools = body.get("tools")
    if body.get("stream"):
        return StreamingResponse(_stream(body, tools, cls, budget), media_type="text/event-stream")
    r = await client.post(UPSTREAM + "/v1/chat/completions", json=body)
    if r.status_code != 200:
        _dump(body, {"upstream_status": r.status_code, "upstream_body": r.text[:4000]})
        _log({"cls": cls, "stream": False, "upstream_error": r.status_code, "err": r.text[:300]})
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
    data = r.json()
    stranded = any(not (c.get("message", {}).get("content") or "").strip() and not c.get("message", {}).get("tool_calls")
                   and "<tool_call>" in (c.get("message", {}).get("reasoning_content") or "") for c in data.get("choices", []))
    _dump(body, data)
    repaired = False
    for ch in data.get("choices", []):
        if repair_message(ch.get("message", {}), tools):
            ch["finish_reason"] = "tool_calls"
            repaired = True
    if stranded:
        STATS["stranded"] += 1
    if repaired:
        STATS["repaired"] += 1
    _log({"cls": cls, "budget": budget, "stream": False, "stranded": stranded, "repaired": repaired,
          "n_msgs": len(body.get("messages", [])), "n_tools": len(tools or []), "usage": data.get("usage")})
    return JSONResponse(data)


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


async def _stream(body, tools, cls, budget):
    """Pass deltas through; once reasoning shows `<tool_call>`, hold reasoning back.
    At the end, if no real tool call or content arrived, emit the lifted call(s)."""
    reasoning = ""
    held = False
    saw_content = saw_calls = False
    last_chunk = None
    async with client.stream("POST", UPSTREAM + "/v1/chat/completions", json=body) as r:
        if r.status_code != 200:
            err = await r.aread()
            _dump(body, {"upstream_status": r.status_code, "upstream_body": err.decode(errors="replace")[:4000]})
            _log({"cls": cls, "stream": True, "upstream_error": r.status_code, "err": err.decode(errors="replace")[:300]})
            yield err
            return
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            chunk = json.loads(payload)
            last_chunk = chunk
            choices = chunk.get("choices") or []
            if not choices:
                yield _sse(chunk)
                continue
            d = choices[0].get("delta", {})
            fin = choices[0].get("finish_reason")
            if d.get("content"):
                saw_content = True
            if d.get("tool_calls"):
                saw_calls = True
            if d.get("reasoning_content"):
                reasoning += d["reasoning_content"]
                if not held and "<tool_call" in reasoning[-(len(d["reasoning_content"]) + 12):]:
                    held = True
                if held:
                    d.pop("reasoning_content")
            if fin is not None:
                stranded = held and not saw_content and not saw_calls
                if stranded:
                    STATS["stranded"] += 1
                _dump(body, {"stream_reasoning": reasoning, "saw_content": saw_content, "saw_calls": saw_calls})
                if stranded and REPAIR:
                    _, calls = extract_calls(reasoning, tools)
                    if calls:
                        STATS["repaired"] += 1
                        for i, c in enumerate(calls):
                            c2 = {"index": i, **c}
                            yield _sse({**chunk, "choices": [{"index": 0, "delta": {"tool_calls": [c2]},
                                                              "finish_reason": None}]})
                        choices[0]["finish_reason"] = "tool_calls"
                        _log({"cls": cls, "budget": budget, "stream": True, "stranded": True, "repaired": True, "n_msgs": len(body.get("messages", []))})
                        yield _sse(chunk)
                        continue
                    # nothing parseable: release what we held so the client sees it
                    d["reasoning_content"] = reasoning[reasoning.find("<tool_call"):]
                _log({"cls": cls, "budget": budget, "stream": True, "stranded": stranded, "repaired": False, "n_msgs": len(body.get("messages", []))})
            if d or fin is not None or "usage" in chunk:
                yield _sse(chunk)
    yield b"data: [DONE]\n\n"


async def stats(_):
    return JSONResponse(STATS)


async def passthrough(request: Request):
    url = UPSTREAM + request.url.path + (("?" + request.url.query) if request.url.query else "")
    r = await client.request(request.method, url, content=await request.body(),
                             headers={k: v for k, v in request.headers.items() if k.lower() in ("content-type", "authorization")})
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


app = Starlette(routes=[
    Route("/v1/chat/completions", chat, methods=["POST"]),
    Route("/shim/stats", stats),
    Route("/{path:path}", passthrough, methods=["GET", "POST", "PUT", "DELETE"]),
])


CHECK_PROMPTS = ["hi", "thanks!", "hello!", "ok cool", "good morning", "bye!", "nice, that worked", "you there?"]
# Thinking that leaked past a forced </think>: planning bullets / headings, or a reply that
# starts mid-sentence (lowercase or punctuation), or one that ran to max_tokens.
_LEAK = re.compile(r"(\*\*\s*(draft|analy|option|plan|step|determine|goal)|^\s*[*\-]\s{2,}|options?:|thinking process|analy[sz]e the|drafting|self-correction)", re.I | re.M)
_MIDSENT = re.compile(r"^\s*[a-z\"',.:)]")


def looks_leaked(text: str, finish: str | None) -> bool:
    return finish == "length" or not text.strip() or bool(_LEAK.search(text[:400])) or bool(_MIDSENT.match(text))


def main(argv=None):
    """reasoning-shim serve | status | check  (exit 0 ok, 1 runtime failure, 2 usage)."""
    import argparse
    import sys
    import urllib.request
    ap = argparse.ArgumentParser(prog="reasoning-shim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run in the foreground (the systemd unit does this)")
    sub.add_parser("status", help="print the running shim's counters")
    ck = sub.add_parser("check", help="send trivial prompts through the shim; exit 1 if any reply leaks thinking")
    ck.add_argument("--model", default=os.environ.get("SHIM_CHECK_MODEL", "piccolo-gorgone:9b"))
    ck.add_argument("-n", type=int, default=5, help="how many trivial prompts to send (max %d)" % len(CHECK_PROMPTS))
    a = ap.parse_args(argv)
    base = f"http://{HOST}:{PORT}"
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
        return 0
    try:
        if a.cmd == "status":
            with urllib.request.urlopen(base + "/shim/stats", timeout=5) as r:
                print(json.dumps(json.load(r), indent=1))
            return 0
        results, bad = [], 0
        for prompt in CHECK_PROMPTS[:a.n]:
            body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400}).encode()
            req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
            t = time.time()
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            ch = d["choices"][0]; m = ch["message"]
            c = m.get("content") or ""
            leaked = looks_leaked(c, ch.get("finish_reason"))
            bad += leaked
            results.append({"prompt": prompt, "ok": not leaked, "secs": round(time.time() - t, 1),
                            "reasoning_chars": len(m.get("reasoning_content") or ""),
                            "completion_tokens": d.get("usage", {}).get("completion_tokens"), "reply": c[:100]})
        print(json.dumps({"leaked": f"{bad}/{len(results)}", "results": results}, indent=1))
        return 1 if bad else 0
    except Exception as e:
        print(f"reasoning-shim: {base} unreachable or upstream failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
