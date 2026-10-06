"""V4 - Persistent memory.

An LLM only knows what is in the messages we send it, so it forgets everything
when the program restarts. The simplest fix, and what most harnesses do:

  * keep memory in a plain file (memory.md),
  * inject its contents into the system prompt on every start,
  * give the model a save_memory tool so it can decide what is worth keeping.

(Fancier harnesses search the memory with RAG / vector stores; the principle is
identical - text goes in the prompt.)

Run:  uv run python v4_memory.py
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

MEMORY_FILE = Path(__file__).parent / "memory.md"

SYSTEM_PROMPT = """You are a helpful personal assistant. Use your tools when they help.

You have persistent memory that survives restarts. When the user tells you something
worth remembering (name, preferences, facts about them), call save_memory. Write each
note in the third person about "the user" (e.g. "The user's name is Tim"), never as "I"/"my".

# What you remember about the user
{memory}"""

MCP_SERVERS = {
    "time": {"command": "uvx", "args": ["mcp-server-time"]},
    "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
}


# --- Memory --------------------------------------------------------------------------
def load_memory() -> str:
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("# Memory\n", encoding="utf-8")
    return MEMORY_FILE.read_text(encoding="utf-8")


def save_memory(note: str) -> str:
    """Append one note. Append-only so a weak model can't wipe what it already knows."""
    load_memory()  # make sure the file exists
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {note.strip()}\n")
    return "Saved to memory."


def system_message() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory())}


# --- Local tools (v2 + save_memory) ----------------------------------------------------
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


LOCAL_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "save_memory": save_memory,
}

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
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save a short fact to persistent memory so it is remembered in future sessions.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string", "description": "One third-person sentence about the user, e.g. \"The user's name is Tim\""}},
                "required": ["note"],
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
    print(f"V4 - memory with {MODEL}  (type 'q' to quit)")
    async with AsyncExitStack() as stack:
        print("Connecting to MCP servers...")
        mcp_sessions, mcp_schemas = await connect_mcp(stack)
        tool_schemas = LOCAL_TOOL_SCHEMAS + mcp_schemas
        messages = [system_message()]  # memory is injected here, once per session

        while True:
            try:
                user_input = (await asyncio.to_thread(input, "\nyou> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in {"q", "quit", "exit"}:
                break
            if user_input:
                print(f"\nassistant> {await run_agent(user_input, messages, mcp_sessions, tool_schemas)}")
                messages[0] = system_message()  # pick up anything save_memory just wrote


if __name__ == "__main__":
    asyncio.run(main())
