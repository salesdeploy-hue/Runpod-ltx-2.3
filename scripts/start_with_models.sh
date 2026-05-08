#!/usr/bin/env bash
# Wrapper around the worker-comfyui /start.sh.
#
# Order matters:
#   1. Apply the Gemma loader monkey-patch BEFORE ComfyUI imports the
#      LTXV node modules (otherwise the patch never takes effect).
#   2. Run model-ensure (downloads weights to /runpod-volume on first
#      cold start, no-op when cached).
#   3. Inject low-VRAM ComfyUI args into the COMFY_ARGS env var that
#      worker-comfyui's /start.sh forwards to ``python main.py``.
#   4. Hand off to /start.sh so the RunPod handler runs as PID 1.
set -euo pipefail

echo "[start] LOW_VRAM_MODE=${LOW_VRAM_MODE:-bf16} reserve=${COMFY_RESERVE_VRAM:-5}GB"

# Step 1 — patch the Gemma loader if Q4 mode is requested. Importing
# the module triggers the monkey-patch via its module-level call.
echo "[start] applying gemma loader patch (mode=${LOW_VRAM_MODE:-bf16})…"
/opt/venv/bin/python /usr/local/bin/gemma_loader_patch.py || true

# Step 2 — ensure weights present on /comfyui/models (volume-backed
# when /runpod-volume is mounted). FAIL FAST if anything's missing
# after download — a worker that boots without weights only fails
# later at workflow validation, where the error message is opaque.
echo "[start] running model ensure step…"
if ! /opt/venv/bin/python /usr/local/bin/ensure_models.py; then
  echo "[start] ✗ model ensure step failed — aborting boot. Check the"
  echo "[start]   download errors above (gated repo? bad HF_TOKEN? volume full?)"
  exit 1
fi

# Step 3 — augment ComfyUI args. worker-comfyui's /start.sh uses
# COMFY_ARGS verbatim, so we append rather than replace.
RESERVE="${COMFY_RESERVE_VRAM:-5}"
EXTRA_ARGS="--reserve-vram ${RESERVE}"
# In q4 mode, Gemma + LTX still peak around 18-22GB so --normalvram is
# fine on 32GB+. On 24GB cards (experimental), --lowvram forces extra
# offloading at a small speed cost.
if [ "${LOW_VRAM_MODE:-bf16}" = "q4" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --lowvram"
fi
export COMFY_ARGS="${COMFY_ARGS:-} ${EXTRA_ARGS}"
echo "[start] COMFY_ARGS=${COMFY_ARGS}"

# Pre-import the patch in the same Python process tree so the monkey-
# patch survives into ComfyUI's import. We do this by setting
# PYTHONSTARTUP — every Python invocation in the container will
# pre-import gemma_loader_patch (no-op when LOW_VRAM_MODE=bf16).
export PYTHONSTARTUP=/usr/local/bin/gemma_loader_patch.py

echo "[start] handing off to /start.sh"
exec /start.sh
