#!/usr/bin/env bash
# Wrapper around the worker-comfyui /start.sh that runs our model-
# ensure step first. Keeps the base image's ComfyUI + handler bootstrap
# intact — we just inject a download step before it.
set -euo pipefail

echo "[start_with_models] running model ensure step…"
/opt/venv/bin/python /usr/local/bin/ensure_models.py || true
echo "[start_with_models] handing off to /start.sh"

# Exec the base image's startup so the handler runs as PID 1 and the
# RunPod queue poller starts.
exec /start.sh
