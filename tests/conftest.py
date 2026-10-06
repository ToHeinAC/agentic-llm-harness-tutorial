import json
from types import SimpleNamespace

import pytest

import harness


class FakeClient:
    """Stands in for AsyncOpenAI: replays scripted assistant messages, records requests."""

    def __init__(self, *messages):
        self._messages = list(messages)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        # Snapshot: the harness keeps mutating the same list afterwards.
        self.requests.append({**kwargs, "messages": json.loads(json.dumps(kwargs["messages"]))})
        if not self._messages:
            raise AssertionError("FakeClient ran out of scripted replies")
        return SimpleNamespace(choices=[SimpleNamespace(message=self._messages.pop(0))])


def say(text: str):
    return SimpleNamespace(content=text, tool_calls=None)


def call(name: str, arguments="{}", id="call_1", content=None):
    fn = SimpleNamespace(name=name, arguments=arguments if isinstance(arguments, str) else json.dumps(arguments))
    return SimpleNamespace(content=content, tool_calls=[SimpleNamespace(id=id, function=fn)])


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the harness at a throw-away workspace and memory file."""
    monkeypatch.setattr(harness, "WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr(harness, "MEMORY_FILE", tmp_path / "memory.md")
    return tmp_path


@pytest.fixture
def state(sandbox):
    s = harness.State()
    s.messages = [harness.system_message()]
    return s
