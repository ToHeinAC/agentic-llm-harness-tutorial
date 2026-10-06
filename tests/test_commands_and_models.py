import json

import pytest

import harness


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_models_lists_registry_and_marks_active(state):
    out = harness.handle_command("/models", state)
    assert "* local " in out
    for name in harness.MODELS:
        assert name in out
    assert "(no API key)" in out  # gpt / claude


def test_switch_model(state):
    assert "Switched to local-small" in harness.handle_command("/model local-small", state)
    assert state.model_name == "local-small"


def test_switch_to_hosted_model_requires_key(state, monkeypatch):
    assert "OPENAI_API_KEY" in harness.handle_command("/model gpt", state)
    assert state.model_name == "local"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert "Switched to gpt" in harness.handle_command("/model gpt", state)
    assert state.model_name == "gpt"


def test_unknown_model_and_command(state):
    assert "Unknown model" in harness.handle_command("/model nope", state)
    assert "Unknown command" in harness.handle_command("/wat", state)


def test_quit_returns_none(state):
    assert harness.handle_command("/quit", state) is None


def test_tools_and_memory_commands(state):
    assert "local  list_files" in harness.handle_command("/tools", state)
    harness.save_memory("likes tea")
    assert "likes tea" in harness.handle_command("/memory", state)


def test_make_client_needs_key(monkeypatch):
    assert harness.make_client(harness.MODELS["gpt"]) is None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = harness.make_client(harness.MODELS["gpt"])
    assert client is not None and "api.openai.com" in str(client.base_url)
    assert harness.make_client(harness.MODELS["local"]) is not None


def test_load_mcp_config(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"t": {"command": "uvx", "args": ["x"]}, "u": {"command": "python"}}))
    cfg = harness.load_mcp_config(path)
    assert cfg["t"].args == ["x"] and cfg["u"].args == [] and cfg["u"].env is None
    assert harness.load_mcp_config(tmp_path / "missing.json") == {}


def test_shipped_mcp_config_is_valid():
    assert set(harness.load_mcp_config()) == {"time", "fetch"}
