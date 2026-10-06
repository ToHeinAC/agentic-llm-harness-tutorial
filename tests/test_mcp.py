"""Real stdio MCP round-trip against tests/echo_server.py (no network)."""

import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import harness
from conftest import FakeClient, call, say

SERVER = str(Path(__file__).parent / "echo_server.py")
SERVERS = {"echo": harness.McpServer(command=sys.executable, args=[SERVER])}


async def _connect(state, servers=SERVERS):
    stack = AsyncExitStack()
    await stack.__aenter__()
    await harness.connect_mcp(stack, servers, state)
    return stack


def test_discovers_and_calls_mcp_tools(state):
    async def go():
        stack = await _connect(state)
        try:
            assert set(state.mcp_sessions) == {"shout", "explode"}
            names = {s["function"]["name"] for s in state.tool_schemas}
            assert {"shout", "list_files"} <= names  # local + MCP side by side
            assert await harness.call_tool("shout", {"text": "hi"}, state) == "HI"
            assert (await harness.call_tool("explode", {}, state)).startswith("Error:")
        finally:
            await stack.aclose()

    asyncio.run(go())


def test_agent_uses_mcp_tool_through_the_loop(state):
    async def go():
        stack = await _connect(state)
        try:
            client = FakeClient(call("shout", {"text": "abc"}), say("it said ABC"))
            assert await harness.run_agent("shout abc", state, client, "m") == "it said ABC"
            assert [m["content"] for m in state.messages if m["role"] == "tool"] == ["ABC"]
        finally:
            await stack.aclose()

    asyncio.run(go())


def test_broken_server_is_skipped_not_fatal(state, capsys):
    servers = {
        "bad": harness.McpServer(command="definitely-not-a-real-binary"),
        **SERVERS,
    }

    async def go():
        stack = await _connect(state, servers)
        await stack.aclose()

    asyncio.run(go())
    assert "'bad' failed to start" in capsys.readouterr().out
    assert "shout" in state.mcp_sessions


def test_name_collision_with_local_tool_is_skipped(state, capsys, monkeypatch):
    monkeypatch.setitem(harness.LOCAL_TOOLS, "shout", lambda text: "local")

    async def go():
        stack = await _connect(state)
        await stack.aclose()

    asyncio.run(go())
    assert "shout" not in state.mcp_sessions
    assert "name already in use" in capsys.readouterr().out
