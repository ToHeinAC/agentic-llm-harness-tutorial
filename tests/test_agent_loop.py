import asyncio

import pytest

import config
import harness
from conftest import FakeClient, call, say


def run(coro):
    return asyncio.run(coro)


def test_plain_answer_makes_one_request(state):
    client = FakeClient(say("hi there"))
    assert run(harness.run_agent("hello", state, client, "m")) == "hi there"
    assert len(client.requests) == 1
    assert client.requests[0]["model"] == "m"
    assert [m["role"] for m in state.messages] == ["system", "user", "assistant"]


def test_loops_through_multiple_tool_rounds(state, sandbox):
    """The v1 failure case: list -> read -> write -> answer needs 4 model calls."""
    harness.write_file("a.txt", "AAA")
    client = FakeClient(
        call("list_files", id="1"),
        call("read_file", {"file_name": "a.txt"}, id="2"),
        call("write_file", {"file_name": "out.txt", "content": "AAA!"}, id="3"),
        say("done"),
    )
    assert run(harness.run_agent("combine", state, client, "m")) == "done"
    assert len(client.requests) == 4
    assert (harness.WORKSPACE / "out.txt").read_text() == "AAA!"

    # Results are fed back with the matching tool_call_id, and assistant tool_calls kept in history.
    tool_msgs = [m for m in state.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["1", "2", "3"]
    assert tool_msgs[0]["content"] == "a.txt"
    assert tool_msgs[1]["content"] == "AAA"
    assert any(m.get("tool_calls") for m in state.messages if m["role"] == "assistant")


def test_tool_errors_are_returned_to_the_model_not_raised(state):
    client = FakeClient(
        call("read_file", {"file_name": "missing.txt"}, id="1"),
        call("nope", id="2"),
        call("read_file", "{not json", id="3"),
        call("../../x", id="4"),
        say("recovered"),
    )
    assert run(harness.run_agent("go", state, client, "m")) == "recovered"
    results = [m["content"] for m in state.messages if m["role"] == "tool"]
    assert results[0].startswith("Error: FileNotFoundError")
    assert results[1].startswith("Error: KeyError")
    assert results[2].startswith("Error: JSONDecodeError")
    assert results[3].startswith("Error: KeyError")


def test_step_limit_stops_runaway_loops(state, monkeypatch):
    monkeypatch.setattr(config, "MAX_STEPS", 3)
    client = FakeClient(*[call("list_files", id=str(i)) for i in range(10)])
    out = run(harness.run_agent("loop forever", state, client, "m"))
    assert "stopped after 3 steps" in out
    assert len(client.requests) == 3


def test_failed_model_call_rolls_back_history(state):
    class Boom(FakeClient):
        async def _create(self, **kwargs):
            raise ConnectionError("ollama is down")

    client = Boom()
    client.chat.completions.create = client._create
    before = list(state.messages)
    with pytest.raises(ConnectionError):
        run(harness.run_agent("hello", state, client, "m"))
    assert state.messages == before


def test_conversation_history_is_sent_on_later_turns(state):
    client = FakeClient(say("one"), say("two"))
    run(harness.run_agent("first", state, client, "m"))
    run(harness.run_agent("second", state, client, "m"))
    sent = client.requests[1]["messages"]
    assert [m["content"] for m in sent[1:]] == ["first", "one", "second"]


def test_tool_schemas_are_sent_with_every_request(state):
    client = FakeClient(say("x"))
    run(harness.run_agent("hi", state, client, "m"))
    assert {t["function"]["name"] for t in client.requests[0]["tools"]} >= {"list_files", "save_memory"}
