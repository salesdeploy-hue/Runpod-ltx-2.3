# AdsXFlow LTX-2.3 RunPod Serverless worker — ComfyUI-based.
#
# Why ComfyUI vs the native ltx_pipelines path:
#   - ComfyUI's memory manager actually offloads Gemma + idle layers
#     between transformer blocks. Native ltx_pipelines pinned both
#     the LTX 22B AND the Gemma 12B in VRAM at all times → 138 GB
#     OOMs even on H200 (140 GB) at our render params.
#   - ComfyUI fits LTX-2.3 22B on **32 GB GPUs** (per Lightricks'
#     ComfyUI-LTXVideo README). RTX 5090 / L40S / A100 are all in.
#   - Workflow JSON exposes every LTX param (resolution, frames,
#     fp8, LoRA, IC-LoRA, etc.) without us hand-coding a handler
#     for each variant.
#
# Strategy: layer LTX-2.3 ComfyUI nodes + weights on top of RunPod's
# published worker-comfyui base. The base provides ComfyUI itself
# plus the runpod handler that polls the prediction queue.

FROM runpod/worker-comfyui:5.8.5-base-cuda12.8.1

ENV DEBIAN_FRONTEND=noninteractive

# ─── Install Lightricks/ComfyUI-LTXVideo custom nodes ──────────────────────
# These are the official LTX-Video / LTX-2.3 nodes from Lightricks.
# `comfy-node-install` is provided by the base image.
RUN comfy-node-install ComfyUI-LTXVideo

# Some LTX-2.3 nodes need extra Python deps (av, openimageio, etc.)
# beyond what the base image ships. Install into ComfyUI's venv.
RUN /opt/venv/bin/pip install --no-cache-dir \
    "av>=13.0.0" \
    "imageio[ffmpeg]>=2.36.0" \
    "imageio-ffmpeg>=0.5.1" \
    "huggingface_hub>=0.26.0"

# ─── Pre-download LTX-2.3 model weights ────────────────────────────────────
# Bake the weights into the image so cold starts don't pay the
# HuggingFace bandwidth on every worker boot. Going with the FP8
# variant (~27 GB) instead of the full bf16 (~44 GB) — matches
# what Lightricks/ComfyUI-LTXVideo expects for the 32 GB-GPU floor,
# and ComfyUI's loader handles fp8 weights natively.
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

RUN mkdir -p /comfyui/models/checkpoints \
             /comfyui/models/vae \
             /comfyui/models/text_encoders \
             /comfyui/models/upscale_models

# 1. LTX-2.3 distilled fp8 (~27 GB). The "distilled-fp8" variant runs
#    in 8 inference steps with CFG=1, fits 32 GB GPUs.
RUN /opt/venv/bin/python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='Lightricks/LTX-2.3-fp8', \
                filename='ltx-2.3-22b-distilled-fp8.safetensors', \
                local_dir='/comfyui/models/checkpoints')"

# 2. Spatial upscaler ×2 v1.1 (~1 GB) — used by the two-stage workflow
#    to bring the half-res stage_1 output up to full res.
RUN /opt/venv/bin/python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='Lightricks/LTX-2.3', \
                filename='ltx-2.3-spatial-upscaler-x2-1.1.safetensors', \
                local_dir='/comfyui/models/upscale_models')"

# 3. Gemma 3 12B text encoder (~22.7 GB) — Lightricks' open variant
#    with the multimodal preprocessor_config.json. Lands at the path
#    the LTX-2.3 nodes expect.
RUN /opt/venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Lightricks/gemma-3-12b-it-qat-q4_0-unquantized', \
                  local_dir='/comfyui/models/text_encoders/gemma-3-12b')"

# Worker-side env vars. ComfyUI-Manager's network mode is set on
# container start by the base image's startup script; we just add
# our knobs.
ENV REFRESH_WORKER=false \
    HF_HUB_ENABLE_HF_TRANSFER=0
