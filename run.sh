#!/usr/bin/env bash
# MC launcher (Linux/macOS).
#   ./run.sh            real renders (needs ComfyUI running on :8188)
#   ./run.sh --mock     UI demo without a GPU
#   ./run.sh --agent    also serve the agent connector (MCP) at http://127.0.0.1:8765/mcp (add --remote for a token)
#   ./run.sh --check    is ComfyUI up, and which model files are still missing for your GPU tier
#   COMFYUI_DIR=~/ComfyUI ./run.sh
#                       also start ComfyUI if it isn't running, always with --reserve-vram 1.5 --max-upload-size 500
#                       (override: MC_RESERVE_VRAM=2 for streaming, MC_MAX_UPLOAD_MB=1024; MC_COMFY_LAUNCHER=<script>)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
export COMFY_BASE_URL=${COMFY_BASE_URL:-http://127.0.0.1:8189}
if [ "${1:-}" = "--mock" ]; then
  exec .venv/bin/python app.py --backend mock
fi
export COMFYUI_URL=${COMFYUI_URL:-http://127.0.0.1:8188}
if [ "${1:-}" = "--check" ]; then
  exec .venv/bin/python -m main_character_comfy.doctor --live "$COMFYUI_URL"
fi
if [ -n "${COMFYUI_DIR:-}" ] && ! curl -fsS "$COMFYUI_URL/system_stats" >/dev/null 2>&1; then
  LA=(-m main_character_comfy.comfy_launch --root "$COMFYUI_DIR")
  [ -n "${MC_MAX_UPLOAD_MB:-}" ] && LA+=(--max-upload-size "$MC_MAX_UPLOAD_MB")
  [ -n "${MC_RESERVE_VRAM:-}" ] && LA+=(--reserve-vram "$MC_RESERVE_VRAM")
  .venv/bin/python "${LA[@]}"
  for _ in $(seq 1 180); do curl -fsS "$COMFYUI_URL/system_stats" >/dev/null 2>&1 && break; sleep 1; done
fi
if ! curl -fsS "$COMFY_BASE_URL/api/v2/health" >/dev/null 2>&1; then
  echo "Starting comfy-api-proxy ($COMFYUI_URL -> $COMFY_BASE_URL)..."
  .venv/bin/comfy-api-proxy start --comfyui "$COMFYUI_URL" --port "${COMFY_BASE_URL##*:}" || \
    (nohup .venv/bin/comfy-api-proxy run --comfyui "$COMFYUI_URL" > .proxy.log 2>&1 &)
  for _ in $(seq 1 20); do curl -fsS "$COMFY_BASE_URL/api/v2/health" >/dev/null 2>&1 && break; sleep 0.5; done
fi
exec .venv/bin/python app.py --backend comfy "$@"
