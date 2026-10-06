"""V3 - MCP servers.

Instead of hard-coding every tool, we let the harness *discover* tools from MCP
(Model Context Protocol) servers at start-up. For each server we:

  1. launch it (here: as a child process speaking over stdin/stdout, "stdio"),
  2. ask it which tools it offers,
  3. translate each tool into the same JSON schema format we already use.

To the model, local tools and MCP tools look identical. Only `call_tool` knows the
difference. MCP is network/process I/O, so the code becomes async.

The two example servers are started with `uvx`, so nothing needs installing.

Run:  uv run python v3_mcp.py
"""

import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

from config import API_KEY, BASE_URL, MAX_STEPS, MODEL

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

SYSTEM_PROMPT = "You are a helpful personal assistant. Use your tools when they help."

MCP_SERVERS = {
    "time": {"command": "uvx", "args": ["mcp-server-time"]},
    "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
}


# --- Local tools (unchanged from v2) ------------------------------------------------
def _resolve(file_name: str) -> Path:
    path = (WORKSPACE / file_name).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError(f"'{file_name}' is outside the workspace")
    return path


def list_files() -> str:
    names = sorted(p.name for p in WORKSPACE.iterdir() if p.is_file() and p.name != ".gitkeep")
    return "\n".join(names) if names else "(workspace is empty)"


def read_file(file_name: str) -> str:
    return _resolve(file_name).read_text(encoding="utf-8")


def write_file(file_name: str, content: str) -> str:
    path = _resolve(file_name)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {file_name}"


LOCAL_TOOLS = {"list_files": list_files, "read_file": read_file, "write_file": write_file}

LOCAL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the workspace directory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {"file_name": {"type": "string", "description": "Name of the file to read"}},
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file in the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of the file to write"},
                    "content": {"type": "string", "description": "Full text content of the file"},
                },
                "required": ["file_name", "content"],
            },
        },
    },
]


# --- MCP: discover remote tools ------------------------------------------------------
async def connect_mcp(stack: AsyncExitStack) -> tuple[dict[str, ClientSession], list[dict]]:
    """Connect to every server in MCP_SERVERS.

    Returns (tool name -> session that owns it, schemas for the model). The
    `stack` owns the open connections and closes them all when the program ends.
    """
    sessions: dict[str, ClientSession] = {}
    schemas: list[dict] = []

    for server_name, cfg in MCP_SERVERS.items():
        try:
            params = StdioServerParameters(command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env"))
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools = (await session.list_tools()).tools
            for tool in tools:
                sessions[tool.name] = session
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,  # MCP already uses JSON Schema
                        },
                    }
                )
            print(f"  MCP server '{server_name}' connected")
        except Exception as exc:  # noqa: BLE001 - one broken server shouldn't kill the harness
            print(f"  MCP server '{server_name}' failed to start: {exc}")

    return sessions, schemas


async def call_tool(name: str, args: dict, mcp_sessions: dict[str, ClientSession]) -> str:
    """Route a tool call to a local function or to the MCP server that owns the tool."""
    if name in LOCAL_TOOLS:
        return str(LOCAL_TOOLS[name](**args))
    if name in mcp_sessions:
        result = await mcp_sessions[name].call_tool(name, args)
        # MCP results are a list of content blocks; keep the text ones.
        text = "\n".join(block.text for block in result.content if getattr(block, "text", None))
        return f"Error: {text}" if result.is_error else text
    raise KeyError(f"unknown tool '{name}'")


def assistant_message(message) -> dict:
    entry: dict = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
    return entry


async def run_agent(user_input: str, messages: list[dict], mcp_sessions: dict, tool_schemas: list[dict]) -> str:
    """The same loop as v2 - now with `await`s."""
    messages.append({"role": "user", "content": user_input})

    for _ in range(MAX_STEPS):
        response = await client.chat.completions.create(model=MODEL, messages=messages, tools=tool_schemas)
        message = response.choices[0].message
        messages.append(assistant_message(message))

        if not message.tool_calls:
            return message.content or ""

        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
                print(f"  [tool] {call.function.name}({args})")
                result = await call_tool(call.function.name, args, mcp_sessions)
            except Exception as exc:  # noqa: BLE001
                result = f"Error: {type(exc).__name__}: {exc}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return f"(stopped after {MAX_STEPS} steps without a final answer)"


async def main() -> None:
    print(f"V3 - MCP with {MODEL}  (type 'q' to quit)")
    async with AsyncExitStack() as stack:
        print("Connecting to MCP servers...")
        mcp_sessions, mcp_schemas = await connect_mcp(stack)
        tool_schemas = LOCAL_TOOL_SCHEMAS + mcp_schemas
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        while True:
            try:
                user_input = (await asyncio.to_thread(input, "\nyou> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in {"q", "quit", "exit"}:
                break
            if user_input:
                print(f"\nassistant> {await run_agent(user_input, messages, mcp_sessions, tool_schemas)}")


if __name__ == "__main__":
    asyncio.run(main())
