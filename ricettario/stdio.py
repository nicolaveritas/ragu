"""Same MCP tools, same core, over stdio — for local MCP dev (Claude Desktop, mcp dev).

Run: uv run python -m ricettario.stdio
"""

from ricettario.adapters.mcp import mcp

if __name__ == "__main__":
    mcp.run()
