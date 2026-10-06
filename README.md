# Agentic LLM Harness — from scratch in pure Python

A rebuild of the harness from Tech With Tim's
[Build Your Own Agentic Harness in Python](https://www.youtube.com/watch?v=H5o1P8RMiMw).
No agent framework: just the OpenAI SDK (to talk to *any* OpenAI-compatible server),
the MCP SDK, and Pydantic.

> **Note:** the author's own code download is gated behind a community sign-up, so this
> is a reconstruction from the video's walkthrough, not a copy of his files.

**Core idea:** an LLM only turns text into text. The *harness* is the code around it that
gives it hands: it runs the tools the model asks for, loops until the task is done,
remembers things, and routes between models.

## Stages — each file is standalone and builds on the previous one

| File | Adds | Key idea |
|---|---|---|
| `v0_chat.py` | Calling an LLM | `messages` in → text out |
| `v1_tools.py` | Tool calling | Model *requests* a tool via a JSON schema; we run it. One round only — deliberately limited |
| `v2_loop.py` | Agentic loop + `write_file` | `while` the model keeps requesting tools, run them and feed results back |
| `v3_mcp.py` | MCP servers (async) | Discover tools from external servers; to the model they look like local tools |
| `v4_memory.py` | Persistent memory | `memory.md` injected into the system prompt + `save_memory` tool |
| `harness.py` | Everything + model routing + `/commands` | Model registry, `mcp_servers.json`, tested loop |

## Quick start

```bash
uv sync
ollama pull qwen3.5:1.7b          # the default in config.py (the video uses qwen3.5:4b)
uv run python harness.py
```

Try: *"Create one.txt and two.txt, then combine them into combined.txt"*,
*"What time is it in Tokyo?"*, *"Remember that my name is Tim"* — then restart and ask your name.

Commands inside `harness.py`: `/models` `/model <name>` `/tools` `/memory` `/help` `/quit`.

## Choosing a model

Defaults target local Ollama. Override with environment variables:

```bash
export HARNESS_MODEL=qwen3.5:4b            # the video's model
export HARNESS_SMALL_MODEL=qwen3.5:2b      # default for both is qwen3.5:1.7b
# Or use a hosted OpenAI-compatible API for the v0–v4 scripts:
export HARNESS_BASE_URL=https://api.openai.com/v1 HARNESS_API_KEY=sk-... HARNESS_MODEL=gpt-4.1-mini
```

`harness.py` additionally registers `gpt` (needs `OPENAI_API_KEY`) and `claude`
(needs `ANTHROPIC_API_KEY`, via Anthropic's OpenAI-compatible endpoint) — edit `MODELS`
to add your own. Switch live with `/model <name>`. You need a model that supports **tool calling**.

## MCP servers

`mcp_servers.json` uses the same `{"name": {"command", "args", "env"}}` shape other
harnesses use. The defaults (`mcp-server-time`, `mcp-server-fetch`) are launched through
`uvx`, so nothing needs installing. Add any stdio MCP server there.

## Tests

```bash
uv run pytest
```

26 offline tests: sandboxing, the multi-round loop (with a scripted fake model),
error handling, step limit, history rollback, model switching, commands, and a real
stdio MCP round-trip against a tiny local server (`tests/echo_server.py`).

## Where this goes beyond the video

- **Workspace sandbox** — file tools reject `../` and absolute paths that escape `workspace/`.
- **Tool errors go back to the model** as text instead of crashing the loop, so it can retry.
- **`MAX_STEPS` limit** (default 25, `HARNESS_MAX_STEPS`) so a confused small model can't loop forever.
- **History rollback** if a model call fails mid-turn (e.g. Ollama not running).
- **Append-only memory** written in the third person (first-person notes like "My name is Tim"
  made a small model think *it* was Tim).
- MCP tool names that collide with local tools are skipped with a warning.
- Targets `mcp` 2.x (`Tool.input_schema`, `is_error`).

## Not included (on purpose)

No delete tool (the video's agent couldn't delete either), no vector-store memory,
no streaming, no HTTP/SSE MCP transports.
