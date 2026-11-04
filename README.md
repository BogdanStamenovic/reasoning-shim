# reasoning-shim

An OpenAI-compatible proxy that sits in front of llama-server and fixes two problems Qwen3.5-template thinking models have in agent harnesses: they think far too long, and they sometimes write their tool call inside an unclosed `<think>` block, so the call is lost.

First built for `piccolo-gorgone:9b` (a Qwen3.5-9B merge) behind gpu-share, driven by Hermes Agent. The measurements and harness live in `piccolo-tune`.

## How it works

Only `POST /v1/chat/completions` is touched; every other path is passed through unchanged.

**1. Per-turn reasoning budget.** llama-server accepts a per-request `thinking_budget_tokens` (only while its global `--reasoning-budget` is -1, the default). When the budget runs out, its sampler forces `</think>` and the model has to answer. The shim picks the budget per request:

| Request | Budget |
|---|---|
| client sent `thinking_budget_tokens` ≥ 0 | left alone |
| `chat_template_kwargs.enable_thinking: false` | left alone |
| `reasoning_effort: none` / `minimal` | 0 |
| `reasoning_effort: low` | 256 |
| `reasoning_effort: medium` | 1024 |
| last message is a tool result | `SHIM_BUDGET_TOOL` (768) |
| obvious small talk ("hi", "thanks!") | `SHIM_BUDGET_TRIVIAL` (96) |
| anything else, incl. `reasoning_effort: high` | `SHIM_BUDGET_DEFAULT` (-1, unlimited) |

`reasoning_effort` matters because llama-server ignores it. Hermes Agent sends `none` on its session-title side call and `medium` on every agent turn; without the shim those run unbounded.

**2. Stranded tool-call repair.** The template opens every assistant turn with `<think>\n`, and llama-server parses everything before `</think>` as `reasoning_content`. If the model writes `<tool_call><function=…>…</function></tool_call>` before closing the block, the response has empty content and no `tool_calls`, and the harness treats it as a final answer. When that happens, and only then (no content, no real tool calls), the shim parses the first complete call(s) out of the reasoning and returns them as proper `tool_calls` with `finish_reason: tool_calls`. Streaming and non-streaming are both handled. Calls to tools not in the request's `tools` list are ignored. Argument types follow the tool's JSON schema.

## Measured (piccolo-gorgone:9b, RTX 4060, 2026-11-04)

Small samples (n = 2–9 per cell): read these as a screen, not a benchmark.

| What | Without shim | With shim |
|---|---|---|
| "hi" / "thanks!" reasoning tokens | 508–987 | 96 |
| Hermes session-title side call | ~5.7k reasoning tokens, ~140 s | 0 reasoning tokens |
| Hard Hermes 0.22.6 task (fix 3 bugs, 4 tests), success | 3/3 | 6/6 |
| same, median wall time per run | 253 s | 66 s |
| Hermes 0.21.4, same task, stranded calls | 6 in 3 runs, 0/3 solved | 7 caught and repaired in one run, agent kept going |
| Hard math, 1024-token cap vs uncapped | — | 4/4 correct capped; uncapped answered 1 of 4 within 8192 tokens |

## Install

```bash
ownbox install reasoning-shim          # setup asks for the upstream URL when run in a terminal
# non-interactive:
RSHIM_UPSTREAM=http://archserver:11434 ownbox install reasoning-shim --yes
```

Setup creates a venv, writes `~/.config/reasoning-shim/env` (only if missing), installs and starts the systemd user unit `reasoning-shim.service`, listening on `127.0.0.1:11450`. Point the client at `http://127.0.0.1:11450/v1`.

`ownbox update reasoning-shim` reinstalls dependencies, keeps the config and restarts. `ownbox uninstall reasoning-shim` stops and removes the unit; it asks before deleting config and logs (kept when there's no terminal).

## Usage

```bash
reasoning-shim status     # counters: requests, stranded, repaired, budget per class
reasoning-shim check      # one "hi" through the shim; prints reasoning size and the reply
reasoning-shim serve      # foreground (what the unit runs)
```

Configuration is environment variables in `~/.config/reasoning-shim/env`. Edit, then `systemctl --user restart reasoning-shim`. `SHIM_BUDGETS=0` or `SHIM_REPAIR=0` turns either feature off; `SHIM_DUMP=DIR` writes full request/response bodies for debugging (off by default, because they contain prompts).

## Limitations

- **The trivial-prompt classifier is a regex.** It catches greetings and thanks, but misses most of the long tail: "What is 2+2?" and "Say hello in French." still get unlimited reasoning (574–1601 tokens measured). Clients that send `reasoning_effort` are not affected. A learned router is planned in piccolo-tune.
- **The default budget is unlimited.** On hard prompts piccolo can reason past 8192 tokens without ever answering. A 1024–2048 cap fixed that in a small test (4/4 correct), but a cap could also cut off a genuinely long problem, so it is not on by default. Set `SHIM_BUDGET_DEFAULT=2048` if runaways matter more to you.
- **Budget cut-offs are abrupt.** llama-server's `--reasoning-budget-message` (text injected before the forced `</think>`) is a server flag, not a per-request field, so the shim cannot set it.
- **Repair only covers the Qwen3.5 XML call format** (`<tool_call><function=NAME><parameter=P>…`). Other templates pass through unrepaired.
- **Repair was proven live on Hermes 0.21.4 only, and that run did not finish** (the host ran out of RAM), so end-to-end task success with repair has not been shown. Hermes 0.22.6+ almost never triggers the bug: 0 stranded calls in ~120 runs.
- **No auth and no TLS.** It binds to localhost by default; don't expose it.
- **The metadata log grows without rotation**: about 200 bytes per request.
