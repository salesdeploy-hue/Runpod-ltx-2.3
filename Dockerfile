# AdsXFlow LTX-2.3 RunPod Serverless worker — ComfyUI-based.
#
# This image stays SLIM (~3 GB). Weight downloads (~50 GB total) happen
# at first worker boot via the start hook, persisted on the attached
# network volume so subsequent cold starts mount-and-go.
#
# Why slim: baking 50 GB of safetensors into the image hit RunPod's
# build timeout. Lazy-download to /runpod-volume on first call avoids
# that entirely while still giving cold-start-once economics when a
# volume is attached.
#
# Architecture:
#   FROM worker-comfyui (handler + ComfyUI server + websocket plumbing)
#   + Lightricks ComfyUI-LTXVideo nodes (small, ~50 MB)
#   + start hook that ensures models are present on /comfyui/models
#     (symlinked from /runpod-volume/models when volume is attached)

FROM runpod/worker-comfyui:5.8.5-base-cuda12.8.1

ENV DEBIAN_FRONTEND=noninteractive

# ─── Install Lightricks/ComfyUI-LTXVideo custom nodes ──────────────────
# Provided by the base image's helper. Tiny — just the node code,
# no model weights.
RUN comfy-node-install ComfyUI-LTXVideo

# Extra Python deps the LTX nodes import at runtime (av for video
# encode, imageio for fallbacks, hf hub for the lazy download in
# our start hook).
RUN /opt/venv/bin/pip install --no-cache-dir \
    "av>=13.0.0" \
    "imageio[ffmpeg]>=2.36.0" \
    "imageio-ffmpeg>=0.5.1" \
    "huggingface_hub>=0.26.0"

# ─── Lazy-download startup hook ────────────────────────────────────────
# The hook runs once per worker boot (before ComfyUI's server starts):
#   - If /runpod-volume is mounted (network volume attached), models
#     persist there across cold starts → only the first ever boot
#     pays the ~5-10 min download.
#   - If no volume, models go to /comfyui/models on the container
#     disk → re-downloaded on each cold start (works but slow).
COPY scripts/ensure_models.py /usr/local/bin/ensure_models.py
COPY scripts/start_with_models.sh /usr/local/bin/start_with_models.sh
RUN chmod +x /usr/local/bin/ensure_models.py /usr/local/bin/start_with_models.sh

# Override the base image's CMD/ENTRYPOINT to run our hook first,
# then exec the original startup script (which boots ComfyUI +
# the runpod handler).
CMD ["/usr/local/bin/start_with_models.sh"]

# Worker-side knobs.
ENV REFRESH_WORKER=false \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    LTX_REPO=Lightricks/LTX-2.3 \
    LTX_FP8_REPO=Lightricks/LTX-2.3-fp8 \
    GEMMA_REPO=Lightricks/gemma-3-12b-it-qat-q4_0-unquantized
