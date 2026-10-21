"""Drive MC with your own agent: an MCP server (stdio + streamable HTTP) over :mod:`main_character_comfy.service`.

    python -m main_character_comfy.agent                       # stdio (Claude Desktop, Cursor, any local client)
    python -m main_character_comfy.agent --http                # http://127.0.0.1:8765/mcp, this PC only
    python -m main_character_comfy.agent --http --remote       # bearer token required (for a tunnel / tailnet)
"""

from .server import build_server, INSTRUCTIONS  # noqa: F401
