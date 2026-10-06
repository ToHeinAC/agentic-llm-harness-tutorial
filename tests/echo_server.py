"""Tiny stdio MCP server used by the tests (no network, no uvx)."""

from mcp.server.mcpserver import MCPServer

server = MCPServer("echo")


@server.tool()
def shout(text: str) -> str:
    """Return the text in upper case."""
    return text.upper()


@server.tool()
def explode() -> str:
    """Always fails."""
    raise RuntimeError("boom")


if __name__ == "__main__":
    server.run()
