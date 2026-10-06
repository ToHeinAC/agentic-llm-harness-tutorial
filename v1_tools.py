"""V1 - Tool calling (single round).

An LLM cannot run code. It can only *say* "I want to call list_files()". Our code
then runs the function and feeds the result back. To make that possible we:

  1. write ordinary Python functions (the tools),
  2. describe them to the model with a JSON schema,
  3. execute whatever tool calls come back, and send the results to the model.

LIMITATION (on purpose): we only do ONE round of tool calls. If the model wants a
second round, we never run it and the answer comes back empty. v2 fixes that.

Run:  uv run python v1_tools.py
"""

import json
from pathlib import Path

from openai import OpenAI

from config import API_KEY, BASE_URL, MODEL

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# The agent may only touch files in here.
WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

SYSTEM_PROMPT = "You are a helpful personal assistant. Use your tools when they help."


# --- 1. The tools: plain Python functions ------------------------------------------
def _resolve(file_name: str) -> Path:
    """Map a model-supplied name to a path inside WORKSPACE (blocks ../ escapes)."""
    path = (WORKSPACE / file_name).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError(f"'{file_name}' is outside the workspace")
    return path


def list_files() -> str:
    names = sorted(p.name for p in WORKSPACE.iterdir() if p.is_file() and p.name != ".gitkeep")
    return "\n".join(names) if names else "(workspace is empty)"


def read_file(file_name: str) -> str:
    return _resolve(file_name).read_text(encoding="utf-8")


# name -> function: how *we* find the code when the model asks for a tool by name
TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
}

# --- 2. The schema: how the model learns the tools exist ----------------------------
# Frameworks generate this for you; from scratch we write it by hand.
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
]


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


def run_tool(name: str, arguments: str) -> str:
    """Execute one tool call requested by the model. Errors go back to the model as text."""
    try:
        args = json.loads(arguments or "{}")
        print(f"  [tool] {name}({args})")
        return str(TOOLS[name](**args))
    except Exception as exc:  # noqa: BLE001 - the model should see any failure and adapt
        return f"Error: {type(exc).__name__}: {exc}"


# --- 3. One round of: ask -> (maybe) run tools -> ask again --------------------------
def chat(user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
    message = response.choices[0].message

    if not message.tool_calls:
        return message.content or ""

    # The model said what it wants; remember that, run it ourselves, report back.
    messages.append(assistant_message(message))
    for call in message.tool_calls:
        result = run_tool(call.function.name, call.function.arguments)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    final = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
    # If the model asks for yet another tool here, content is empty - the v1 limitation.
    return final.choices[0].message.content or ""


def main() -> None:
    print(f"V1 - tool calling with {MODEL}  (type 'q' to quit)")
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"q", "quit", "exit"}:
            break
        if user_input:
            print(f"\nassistant> {chat(user_input)}")


if __name__ == "__main__":
    main()
