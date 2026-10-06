"""V0 - Calling an LLM.

An LLM is a function from text to text. This is the whole interface: send a list
of messages, get a message back. Everything else in this tutorial (tools, loops,
MCP, memory) is code we wrap *around* this one call.

Run:  uv run python v0_chat.py
"""

from openai import OpenAI

from config import API_KEY, BASE_URL, MODEL

# The OpenAI SDK speaks the OpenAI wire protocol, which Ollama, vLLM, LM Studio,
# Anthropic, ... all implement. Changing provider = changing base_url/key/model.
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def chat(user_input: str) -> str:
    """Send one user message, return the model's text reply."""
    response = client.chat.completions.create(
        model=MODEL,
        # `messages` is the entire conversation the model gets to see:
        #   system    - standing instructions / context not coming from the user
        #   user      - what the human asked
        #   assistant - what the model said earlier (used for multi-turn history)
        #   tool      - results of tool calls (introduced in v1)
        messages=[
            {"role": "system", "content": "You are a helpful personal assistant."},
            {"role": "user", "content": user_input},
        ],
    )
    # The response also carries token usage and tool calls; here we only want the text.
    return response.choices[0].message.content or ""


def main() -> None:
    print(f"V0 - plain chat with {MODEL}  (type 'q' to quit)")
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
