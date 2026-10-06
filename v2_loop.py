"""V2 - The agentic loop.

The only change from v1: instead of one round of tool calls, we loop.

    while True:
        ask the model
        if it requested no tools -> that's the final answer, stop
        otherwise run the tools, append the results, go around again

That loop *is* an agent. It may run once, or 20 times, depending on the task.
We also add a write_file tool so the agent can produce something.

Run:  uv run python v2_loop.py
"""

import json
from pathlib import Path

from openai import OpenAI

from config import API_KEY, BASE_URL, MAX_STEPS, MODEL

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

SYSTEM_PROMPT = "You are a helpful personal assistant. Use your tools when they help."


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


TOOLS = {"list_files": list_files, "read_file": read_file, "write_file": write_file}

TOOL_SCHEMAS = [
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


def run_tool(name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
        print(f"  [tool] {name}({args})")
        return str(TOOLS[name](**args))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {type(exc).__name__}: {exc}"


def run_agent(user_input: str, messages: list[dict]) -> str:
    """Run the agentic loop for one user turn. `messages` is the live conversation."""
    messages.append({"role": "user", "content": user_input})

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        message = response.choices[0].message
        messages.append(assistant_message(message))

        if not message.tool_calls:  # no more actions requested -> final answer
            return message.content or ""

        for call in message.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return f"(stopped after {MAX_STEPS} steps without a final answer)"


def main() -> None:
    print(f"V2 - agentic loop with {MODEL}  (type 'q' to quit)")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # kept across turns
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"q", "quit", "exit"}:
            break
        if user_input:
            print(f"\nassistant> {run_agent(user_input, messages)}")


if __name__ == "__main__":
    main()
