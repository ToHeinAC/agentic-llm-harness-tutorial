"""The full agentic harness: every piece from v0-v4, plus model routing and commands.

    LLM (the brain)  +  harness (this file)  =  agent

  * agentic loop        ask -> run requested tools -> feed results back -> repeat
  * local tools         list_files / read_file / write_file in a sandboxed workspace
  * MCP servers         tools discovered at start-up from mcp_servers.json
  * persistent memory   memory.md injected into the system prompt, save_memory tool
  * model routing       several models registered; switch live with /model
  * slash commands      /models /model <name> /tools /memory /help /quit

Run:  uv run python harness.py
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from pydantic import BaseModel

import config

ROOT = Path(__file__).parent
WORKSPACE = ROOT / "workspace"
MEMORY_FILE = ROOT / "memory.md"
MCP_CONFIG = ROOT / "mcp_servers.json"


# --- Model registry (model routing) ------------------------------------------------------
class ModelConfig(BaseModel):
    """One entry in the registry. Anything OpenAI-compatible fits."""

    model: str
    base_url: str
    api_key: str | None = None  # literal key (local servers)
    api_key_env: str | None = None  # or: name of the env var holding it (hosted APIs)

    def resolve_key(self) -> str | None:
        return self.api_key if self.api_key is not None else os.getenv(self.api_key_env or "")


MODELS: dict[str, ModelConfig] = {
    "local": ModelConfig(model=config.MODEL, base_url=config.BASE_URL, api_key=config.API_KEY),
    "local-small": ModelConfig(model=config.SMALL_MODEL, base_url=config.BASE_URL, api_key=config.API_KEY),
    "gpt": ModelConfig(
        model="gpt-4.1-mini", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY"
    ),
    # Anthropic exposes an OpenAI-compatible endpoint, so the same client works.
    "claude": ModelConfig(
        model="claude-sonnet-5-5", base_url="https://api.anthropic.com/v1/", api_key_env="ANTHROPIC_API_KEY"
    ),
}
DEFAULT_MODEL = "local"


def make_client(cfg: ModelConfig) -> AsyncOpenAI | None:
    """Build a client for a registry entry, or None if its API key isn't available."""
    key = cfg.resolve_key()
    return AsyncOpenAI(base_url=cfg.base_url, api_key=key) if key else None


# --- MCP config --------------------------------------------------------------------------
class McpServer(BaseModel):
    command: str
    args: list[str] = []
    env: dict[str, str] | None = None


def load_mcp_config(path: Path = MCP_CONFIG) -> dict[str, McpServer]:
    """Read {"name": {"command": ..., "args": [...]}} - same shape other harnesses use."""
    if not path.exists():
        return {}
    return {name: McpServer(**spec) for name, spec in json.loads(path.read_text(encoding="utf-8")).items()}


# --- Memory ------------------------------------------------------------------------------
def load_memory() -> str:
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("# Memory\n", encoding="utf-8")
    return MEMORY_FILE.read_text(encoding="utf-8")


def save_memory(note: str) -> str:
    """Append one note. Append-only so a weak model can't wipe what it already knows."""
    load_memory()
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {note.strip()}\n")
    return "Saved to memory."


SYSTEM_PROMPT = """You are a helpful personal assistant running inside a small agent harness.
Use your tools when they help. Files live in a workspace directory; you cannot delete files.

You have persistent memory that survives restarts. When the user tells you something
worth remembering (name, preferences, facts about them), call save_memory. Write each
note in the third person about "the user" (e.g. "The user's name is Tim"), never as "I"/"my".

# What you remember about the user
{memory}"""


def system_message() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory())}


# --- Local tools -------------------------------------------------------------------------
def _resolve(file_name: str) -> Path:
    """Map a model-supplied name to a path inside WORKSPACE (blocks ../ escapes)."""
    WORKSPACE.mkdir(exist_ok=True)
    path = (WORKSPACE / file_name).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError(f"'{file_name}' is outside the workspace")
    return path


def list_files() -> str:
    WORKSPACE.mkdir(exist_ok=True)
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


def _schema(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
        },
    }


LOCAL_TOOL_SCHEMAS = [
    _schema("list_files", "List the files in the workspace directory."),
    _schema(
        "read_file",
        "Read a text file from the workspace directory.",
        {"file_name": {"type": "string", "description": "Name of the file to read"}},
        ["file_name"],
    ),
    _schema(
        "write_file",
        "Create or overwrite a text file in the workspace directory.",
        {
            "file_name": {"type": "string", "description": "Name of the file to write"},
            "content": {"type": "string", "description": "Full text content of the file"},
        },
        ["file_name", "content"],
    ),
    _schema(
        "save_memory",
        "Save a short fact to persistent memory so it is remembered in future sessions.",
        {"note": {"type": "string", "description": "One third-person sentence about the user, e.g. \"The user's name is Tim\""}},
        ["note"],
    ),
]


# --- Harness state & MCP connection ------------------------------------------------------
@dataclass
class State:
    model_name: str = DEFAULT_MODEL
    messages: list[dict] = field(default_factory=list)
    mcp_sessions: dict[str, ClientSession] = field(default_factory=dict)  # tool name -> owning session
    mcp_schemas: list[dict] = field(default_factory=list)

    @property
    def tool_schemas(self) -> list[dict]:
        return LOCAL_TOOL_SCHEMAS + self.mcp_schemas


async def connect_mcp(stack: AsyncExitStack, servers: dict[str, McpServer], state: State) -> None:
    """Launch each server, discover its tools, and register them in `state`."""
    for server_name, server in servers.items():
        try:
            params = StdioServerParameters(command=server.command, args=server.args, env=server.env)
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            for tool in (await session.list_tools()).tools:
                if tool.name in LOCAL_TOOLS or tool.name in state.mcp_sessions:
                    print(f"  ! skipping MCP tool '{tool.name}' from '{server_name}': name already in use")
                    continue
                state.mcp_sessions[tool.name] = session
                state.mcp_schemas.append(
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


async def call_tool(name: str, args: dict, state: State) -> str:
    """Route a tool call to a local function or to the MCP server that owns the tool."""
    if name in LOCAL_TOOLS:
        return str(LOCAL_TOOLS[name](**args))
    if name in state.mcp_sessions:
        result = await state.mcp_sessions[name].call_tool(name, args)
        text = "\n".join(block.text for block in result.content if getattr(block, "text", None))
        return f"Error: {text}" if result.is_error else text
    raise KeyError(f"unknown tool '{name}'")


# --- The agentic loop --------------------------------------------------------------------
def assistant_message(message) -> dict:
    """Turn the SDK's response object into a plain message dict for the history."""
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


async def run_agent(user_input: str, state: State, client: AsyncOpenAI, model: str) -> str:
    """Run one user turn to completion: loop until the model stops asking for tools.

    If the model call fails (server down, bad key, ...) the history is rolled back to
    where it was, so a half-finished turn never poisons the conversation.
    """
    checkpoint = len(state.messages)
    state.messages.append({"role": "user", "content": user_input})

    try:
        for _ in range(config.MAX_STEPS):
            response = await client.chat.completions.create(
                model=model, messages=state.messages, tools=state.tool_schemas
            )
            message = response.choices[0].message
            state.messages.append(assistant_message(message))

            if not message.tool_calls:
                return message.content or ""

            for call in message.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    print(f"  [tool] {call.function.name}({args})")
                    result = await call_tool(call.function.name, args, state)
                except Exception as exc:  # noqa: BLE001 - let the model see the failure and adapt
                    result = f"Error: {type(exc).__name__}: {exc}"
                state.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    except BaseException:
        del state.messages[checkpoint:]
        raise

    return f"(stopped after {config.MAX_STEPS} steps without a final answer)"


# --- Slash commands ----------------------------------------------------------------------
HELP = """Commands:
  /models          list registered models (* = active)
  /model <name>    switch model
  /tools           list available tools (local + MCP)
  /memory          show memory.md
  /help            this message
  /quit            exit
Anything else is sent to the agent."""


def handle_command(line: str, state: State) -> str | None:
    """Run a slash command. Returns text to print, or None to quit."""
    command, _, arg = line.strip().partition(" ")
    arg = arg.strip()

    if command in {"/quit", "/exit", "/q"}:
        return None
    if command == "/help":
        return HELP
    if command == "/models":
        rows = []
        for name, cfg in MODELS.items():
            marker = "*" if name == state.model_name else " "
            status = "" if cfg.resolve_key() else "  (no API key)"
            rows.append(f" {marker} {name:<12} {cfg.model}{status}")
        return "\n".join(rows)
    if command == "/model":
        if arg not in MODELS:
            return f"Unknown model '{arg}'. Try /models."
        if not MODELS[arg].resolve_key():
            return f"Can't use '{arg}': set {MODELS[arg].api_key_env} first."
        state.model_name = arg
        return f"Switched to {arg} ({MODELS[arg].model})."
    if command == "/tools":
        local = {s["function"]["name"] for s in LOCAL_TOOL_SCHEMAS}
        return "\n".join(
            f"  {'local' if s['function']['name'] in local else 'mcp  '}  {s['function']['name']}"
            f" - {s['function']['description'].splitlines()[0][:70] if s['function']['description'] else ''}"
            for s in state.tool_schemas
        )
    if command == "/memory":
        return load_memory()
    return f"Unknown command '{command}'. Try /help."


# --- Main loop ---------------------------------------------------------------------------
async def main() -> None:
    print("Starting harness")
    state = State()
    async with AsyncExitStack() as stack:
        await connect_mcp(stack, load_mcp_config(), state)
        state.messages = [system_message()]
        print(f"Model: {state.model_name} ({MODELS[state.model_name].model})  |  tools: {len(state.tool_schemas)}  |  /help for commands")

        while True:
            try:
                line = (await asyncio.to_thread(input, "\nyou> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.startswith("/"):
                output = handle_command(line, state)
                if output is None:
                    break
                print(output)
                continue

            client = make_client(MODELS[state.model_name])
            if client is None:
                print(f"No API key for '{state.model_name}'. Use /model to pick another.")
                continue
            try:
                reply = await run_agent(line, state, client, MODELS[state.model_name].model)
            except Exception as exc:  # noqa: BLE001 - e.g. server down; keep the session alive
                print(f"\n[error] {type(exc).__name__}: {exc}")
                continue
            print(f"\nassistant> {reply}")
            state.messages[0] = system_message()  # pick up anything save_memory just wrote

    print("Bye.")


if __name__ == "__main__":
    asyncio.run(main())
