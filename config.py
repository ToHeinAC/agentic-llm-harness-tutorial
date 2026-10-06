"""Shared settings for every stage of the tutorial.

Defaults point at a local Ollama server (OpenAI-compatible API on :11434).
To use a hosted model instead, set the environment variables, e.g. for OpenAI:

    export HARNESS_BASE_URL=https://api.openai.com/v1
    export HARNESS_API_KEY=sk-...
    export HARNESS_MODEL=gpt-4.1-mini
"""

import os

BASE_URL = os.getenv("HARNESS_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("HARNESS_API_KEY", "ollama")  # Ollama ignores the key, the SDK requires one

# The video uses qwen3.5:4b (`ollama pull qwen3.5:4b`). Any tool-calling model works.
#MODEL = os.getenv("HARNESS_MODEL", "gemma4:e2b")
MODEL = os.getenv("HARNESS_MODEL", "qwen3.5:1.7b")
SMALL_MODEL = os.getenv("HARNESS_SMALL_MODEL", "qwen3.5:1.7b")

# Safety net for the agentic loop: a confused small model must not loop forever.
MAX_STEPS = int(os.getenv("HARNESS_MAX_STEPS", "25"))
