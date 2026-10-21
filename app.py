"""Launch the MC web UI.  python app.py [--backend mock|comfy] [--theme noir] [--port 7860] [--share] [--agent [--remote]]"""

import argparse
import os

from main_character_comfy import themes
from main_character_comfy.config import settings
from main_character_comfy.ui import build_ui, library_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["comfy", "mock"], help="default: $MC_BACKEND or comfy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public Gradio link (your GPU serves it; be careful)")
    ap.add_argument("--theme", help="visual theme from main_character_comfy/themes (default: $MC_THEME or 'noir')")
    ap.add_argument("--agent", action="store_true",
                    help="also serve the agent connector (MCP over HTTP) at http://127.0.0.1:8765/mcp, sharing this "
                         "window's gpu queue")
    ap.add_argument("--agent-port", type=int, default=int(os.environ.get("MC_AGENT_PORT", "8765")))
    ap.add_argument("--remote", action="store_true", help="with --agent: require the bearer token (for a tunnel)")
    a = ap.parse_args()
    if a.agent:
        from main_character_comfy.agent.http import load_or_create_token, serve_in_thread
        from main_character_comfy.service import MCService

        svc = MCService(backend_kind=a.backend)
        token, created = load_or_create_token(svc, create=a.remote)
        if a.remote and created:
            print(f"agent connector token (saved in outputs/_agent/agent_token.txt): {token}")
        serve_in_thread(svc, host="127.0.0.1", port=a.agent_port, token=token if a.remote else None)
        print(f"agent connector: http://127.0.0.1:{a.agent_port}/mcp" + (" (bearer token required)" if a.remote else ""))
    if a.theme:
        os.environ["MC_THEME"] = a.theme
    build_ui(a.backend).queue(default_concurrency_limit=1).launch(
        server_name=a.host, server_port=a.port, share=a.share, inbrowser=True, favicon_path=str(themes.FAVICON),
        # your renders (share bar) + bundled OFL fonts + logo/icons
        allowed_paths=[str(settings().output_dir), str(library_dir()), str(themes.FONTS), str(themes.BRAND)],
    )
