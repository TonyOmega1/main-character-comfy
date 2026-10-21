"""python -m main_character_comfy.agent [--http] [--remote] ...  (see --help)"""

from __future__ import annotations

import argparse
import os
import sys

from ..service import MCService

HELP = """\
Drive MC with your own agent (MCP).

  python -m main_character_comfy.agent                    stdio: Claude Desktop, Cursor, Grok Bot on this PC
  python -m main_character_comfy.agent --http             http://127.0.0.1:8765/mcp (this PC only)
  python -m main_character_comfy.agent --http --remote    bearer token required; expose 127.0.0.1:8765 with
                                                          tailscale serve or a cloudflare tunnel, never a router port
"""


def main(argv: list[str] | None = None) -> int:
    from .http import DEFAULT_PORT, check_exposure, load_or_create_token, serve, token_file

    ap = argparse.ArgumentParser(prog="mc-agent", description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    ap.add_argument("--host", default=os.environ.get("MC_AGENT_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MC_AGENT_PORT", DEFAULT_PORT)))
    ap.add_argument("--remote", action="store_true", help="require a bearer token (for tunnels / tailnets / phones)")
    ap.add_argument("--token", help="bearer token (default: MC_AGENT_TOKEN, else outputs/_agent/agent_token.txt)")
    ap.add_argument("--new-token", action="store_true", help="make a fresh token (replaces the saved one)")
    ap.add_argument("--allow-lan", action="store_true", help="allow a non-loopback --host (token still required)")
    ap.add_argument("--public-url", help="base URL agents reach this server at, for download links "
                                         "(e.g. https://mypc.example.ts.net)")
    ap.add_argument("--backend", choices=["comfy", "mock"], help="default: $MC_BACKEND or comfy")
    ap.add_argument("--show-token", action="store_true", help="print the saved token and exit")
    ap.add_argument("--start-comfy", nargs="?", const="", default=None, metavar="COMFYUI_DIR",
                    help="start ComfyUI first if it isn't running (always --reserve-vram 1.5 --max-upload-size 500; "
                         "run_nvidia_gpu_mc.bat when the folder has it). Folder default: $MC_COMFYUI_DIR / $COMFYUI_DIR")
    a = ap.parse_args(argv)
    if a.start_comfy is not None and a.backend != "mock":
        from ..comfy_launch import DEFAULT_URL, ensure_running

        root = a.start_comfy or os.environ.get("MC_COMFYUI_DIR") or os.environ.get("COMFYUI_DIR") or None
        print("mc agent connector: " + ensure_running(root, os.environ.get("COMFYUI_URL", DEFAULT_URL)),
              file=sys.stderr)

    svc = MCService(backend_kind=a.backend)
    if a.show_token:
        t, _ = load_or_create_token(svc, create=False)
        print(t or "(no token yet: run with --remote --new-token)")
        return 0
    if not a.http:
        from .server import build_server

        build_server(svc).run("stdio")
        return 0
    if a.new_token and token_file(svc).exists():
        token_file(svc).unlink()
    token = a.token
    created = False
    if not token and (a.remote or a.new_token):
        token, created = load_or_create_token(svc, create=True)
    elif not token:
        token, _ = load_or_create_token(svc, create=False) if os.environ.get("MC_AGENT_TOKEN") else (None, False)
    check_exposure(a.host, token, a.remote, a.allow_lan)
    where = f"http://{a.host}:{a.port}/mcp"
    print(f"mc agent connector: {where}  ({'bearer token required' if token else 'this pc only, no token'})",
          file=sys.stderr)
    if created:
        print(f"new token (saved to {token_file(svc)}):\n  {token}\nput it in your agent's connector settings as "
              "'Authorization: Bearer <token>'. keep it secret.", file=sys.stderr)
    elif token:
        print(f"token: from {'MC_AGENT_TOKEN' if os.environ.get('MC_AGENT_TOKEN') else token_file(svc)} "
              "(--show-token prints it)", file=sys.stderr)
    serve(svc, host=a.host, port=a.port, token=token, public_url=a.public_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
