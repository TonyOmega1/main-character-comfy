"""Streamable HTTP transport for the MC MCP server, with MC's security rules.

* Local mode (default): binds 127.0.0.1, so only programs on this PC can connect. A token is optional.
* Remote mode (``--remote``): a bearer token is REQUIRED (``Authorization: Bearer <token>``; ``?token=`` works for
  clients that can't set headers). It still binds 127.0.0.1: expose it with Tailscale (``tailscale serve``) or a
  Cloudflare tunnel, never by opening a port on your router. Binding a LAN address needs ``--allow-lan`` too.
* ``/files/<sig>/<exp>/<path>`` serves MC outputs through signed links that expire after 24 h (HMAC with a key
  derived from the token), so a phone can open a result without the token in the URL.
* ``PUT /upload/<kind>/<filename>`` (token-protected) takes big uploads (clips) that don't fit in a tool call.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote

from ..service import MAX_UPLOAD_BYTES, MCService, AccessError
from ..pipeline import InputError
from .server import build_server, sign

DEFAULT_PORT = 8765
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def is_loopback(host: str) -> bool:
    if host in LOOPBACK:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def token_file(service: MCService) -> Path:
    from ..jobs import agent_dir

    return agent_dir(service.output_dir) / "agent_token.txt"


def load_or_create_token(service: MCService, *, create: bool) -> tuple[str | None, bool]:
    """(token, newly_created). Order: MC_AGENT_TOKEN env, the token file, a new one (when ``create``)."""
    env = os.environ.get("MC_AGENT_TOKEN", "").strip()
    if env:
        return env, False
    f = token_file(service)
    if f.exists():
        t = f.read_text(encoding="utf-8").strip()
        if t:
            return t, False
    if not create:
        return None, False
    t = "mc_" + secrets.token_urlsafe(32)
    f.write_text(t, encoding="utf-8")
    try:
        os.chmod(f, 0o600)
    except OSError:  # pragma: no cover - Windows ACLs
        pass
    return t, True


class TokenGate:
    """ASGI middleware: bearer token on everything except /health and signed /files/ links."""

    def __init__(self, app, token: str | None) -> None:
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.token:
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if path == "/health" or path.startswith("/files/"):
            return await self.app(scope, receive, send)
        given = ""
        for k, v in scope.get("headers", []):
            if k == b"authorization":
                val = v.decode("latin-1")
                given = val[7:].strip() if val.lower().startswith("bearer ") else ""
        if not given:
            q = parse_qs(scope.get("query_string", b"").decode())
            given = (q.get("token") or [""])[0]
        if given and hmac.compare_digest(given.encode(), self.token.encode()):
            return await self.app(scope, receive, send)
        body = b'{"error":"mc: missing or wrong bearer token"}'
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": body})


def build_http_app(service: MCService, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                   token: str | None = None, public_url: str | None = None):
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse

    secret = (hashlib.sha256(b"mc-links|" + token.encode()).digest() if token else secrets.token_bytes(32))
    base = (public_url or os.environ.get("MC_AGENT_PUBLIC_URL") or f"http://{host}:{port}").rstrip("/")
    mcp = build_server(service, public_url=lambda: base, link_secret=secret)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request):
        return JSONResponse({"ok": True, "app": "mc", "auth": "token" if token else "local"})

    @mcp.custom_route("/files/{sig}/{exp}/{path:path}", methods=["GET"])
    async def files(request: Request):
        sig, exp, rel = request.path_params["sig"], request.path_params["exp"], unquote(request.path_params["path"])
        try:
            ok = int(exp) > time.time() and hmac.compare_digest(sig, sign(secret, rel, int(exp)))
        except ValueError:
            ok = False
        if not ok:
            return JSONResponse({"error": "link expired or invalid"}, status_code=403)
        try:
            p = service.resolve(rel)
        except (AccessError, InputError):
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(p, filename=p.name)

    @mcp.custom_route("/upload/{kind}/{filename}", methods=["PUT", "POST"])
    async def upload(request: Request):
        size = int(request.headers.get("content-length") or 0)
        if size > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "too big"}, status_code=413)
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > MAX_UPLOAD_BYTES:
                return JSONResponse({"error": "too big"}, status_code=413)
        try:
            info = service.save_upload(request.path_params["kind"], request.path_params["filename"], bytes(buf))
        except (InputError, AccessError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(info)

    if token:  # the token is the gate; tunnels / tailnets send their own Host header
        sec = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    else:
        hosts = [f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")] + ["127.0.0.1", "localhost"]
        origins = [f"http://{h}:{port}" for h in ("127.0.0.1", "localhost")]
        sec = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts,
                                        allowed_origins=origins)
    app = mcp.streamable_http_app(streamable_http_path="/mcp", transport_security=sec, host=host,
                                  max_request_body_size=64 * 2 ** 20)
    return TokenGate(app, token), base


def check_exposure(host: str, token: str | None, remote: bool, allow_lan: bool) -> None:
    if remote and not token:
        raise SystemExit("mc agent: --remote needs a token (set MC_AGENT_TOKEN, pass --token, or --new-token)")
    if not is_loopback(host):
        if not token:
            raise SystemExit(f"mc agent: refusing to listen on {host} without a token. Use --remote (token "
                             "required) and keep 127.0.0.1 behind Tailscale or a Cloudflare tunnel.")
        if not allow_lan:
            raise SystemExit(f"mc agent: {host} is not this PC only. Prefer 127.0.0.1 + `tailscale serve` or a "
                             "Cloudflare tunnel; if you really mean it, add --allow-lan (never port-forward this).")


def serve(service: MCService, *, host: str, port: int, token: str | None, public_url: str | None = None,
          log_level: str = "warning"):
    """Blocking: run the HTTP server (uvicorn)."""
    import uvicorn

    app, _ = build_http_app(service, host=host, port=port, token=token, public_url=public_url)
    uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level=log_level)).run()


def serve_in_thread(service: MCService, **kw) -> threading.Thread:
    """For ``app.py --agent``: the MCP server in the same process as the MC window."""
    t = threading.Thread(target=serve, args=(service,), kwargs=kw, name="mc-agent-http", daemon=True)
    t.start()
    return t
