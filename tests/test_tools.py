import pytest

import harness


def test_write_read_list_roundtrip(sandbox):
    assert harness.list_files() == "(workspace is empty)"
    harness.write_file("a.txt", "hello")
    harness.write_file("b.txt", "world")
    assert harness.read_file("a.txt") == "hello"
    assert harness.list_files().splitlines() == ["a.txt", "b.txt"]


@pytest.mark.parametrize("name", ["../escape.txt", "/etc/passwd", "sub/../../x"])
def test_workspace_is_sandboxed(sandbox, name):
    with pytest.raises(ValueError, match="outside the workspace"):
        harness.read_file(name)
    with pytest.raises(ValueError, match="outside the workspace"):
        harness.write_file(name, "x")
    assert not (sandbox / "escape.txt").exists()


def test_memory_is_created_appended_and_injected(sandbox):
    assert harness.load_memory() == "# Memory\n"
    harness.save_memory("User is called Tim")
    harness.save_memory("  Tim is 26  ")
    assert harness.MEMORY_FILE.read_text().splitlines() == ["# Memory", "- User is called Tim", "- Tim is 26"]
    prompt = harness.system_message()["content"]
    assert "User is called Tim" in prompt and "Tim is 26" in prompt


def test_every_local_tool_has_a_schema_and_vice_versa():
    schema_names = {s["function"]["name"] for s in harness.LOCAL_TOOL_SCHEMAS}
    assert schema_names == set(harness.LOCAL_TOOLS)
    for s in harness.LOCAL_TOOL_SCHEMAS:
        params = s["function"]["parameters"]
        assert set(params["required"]) <= set(params["properties"])
