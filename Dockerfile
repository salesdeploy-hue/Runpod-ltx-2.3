FROM runpod/base:0.6.2-cuda12.4.1

WORKDIR /workspace

# System deps for video encode + git for cloning LTX-2
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        git \
        git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# uv is the LTX-2 repo's recommended package manager — significantly
# faster than pip for the heavy deps (torch, transformers, etc.).
RUN python -m pip install --upgrade pip uv

# Clone LTX-2 and install the ltx-pipelines + ltx-core packages from
# source. Pinned to a known-good commit; bump after testing a newer
# release. The two `pip install -e` lines avoid `uv sync` here so the
# image only carries inference deps (no dev/test extras).
ARG LTX2_REF=main
RUN git clone --depth 1 --branch ${LTX2_REF} https://github.com/Lightricks/LTX-2.git /workspace/LTX-2 \
    && cd /workspace/LTX-2 \
    && pip install --no-cache-dir -e packages/ltx-core \
    && pip install --no-cache-dir -e packages/ltx-pipelines

# Worker SDK + a few utilities the handler needs
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# ─── Pre-download checkpoints into the image ────────────────────────────
# Baking weights into the image avoids paying ~50GB of HF Hub
# bandwidth on every cold boot. Comment these out for faster image
# rebuilds during dev — the first job will fall back to a runtime
# download (slower first call, same total cost).
#
# Three artefacts:
#   1. LTX-2.3 distilled-1.1 base checkpoint (~44GB bf16)
#   2. Spatial upscaler ×2 1.1 (~1GB)
#   3. Gemma text encoder
#
# `--include` patterns keep us from pulling the whole 22B repo when
# we only need the merged distilled-1.1 safetensors + tokenizer.
RUN mkdir -p /workspace/models /workspace/models/gemma

ARG LTX_REPO=Lightricks/LTX-2.3
ARG GEMMA_REPO=google/gemma-3-1b-pt
ARG HF_TOKEN=""
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN python - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
token = os.environ.get("HF_TOKEN") or None

# Distilled-1.1 — single merged safetensors. The repo lists this as
# `ltx-2.3-22b-distilled-1.1.safetensors` at the root.
hf_hub_download(
    repo_id=os.environ.get("LTX_REPO", "Lightricks/LTX-2.3"),
    filename="ltx-2.3-22b-distilled-1.1.safetensors",
    local_dir="/workspace/models",
    token=token,
)

# Spatial upscaler ×2 v1.1
hf_hub_download(
    repo_id=os.environ.get("LTX_REPO", "Lightricks/LTX-2.3"),
    filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    local_dir="/workspace/models",
    token=token,
)

# Gemma text encoder — LTX-2.3 expects a local path to a Gemma model
# directory containing model weights + tokenizer. gemma-3-1b-pt is
# the smallest variant compatible with LTX-2's gated text connector.
snapshot_download(
    repo_id=os.environ.get("GEMMA_REPO", "google/gemma-3-1b-pt"),
    local_dir="/workspace/models/gemma",
    token=token,
)
PY

COPY handler.py /workspace/handler.py

# Default env points the handler at the baked checkpoints. Operators
# can override any of these on the RunPod endpoint config to point
# at a network volume cache instead.
ENV LTX_CHECKPOINT_PATH=/workspace/models/ltx-2.3-22b-distilled-1.1.safetensors \
    LTX_SPATIAL_UPSAMPLER_PATH=/workspace/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    LTX_GEMMA_ROOT=/workspace/models/gemma \
    LTX_DISTILLED_LORA_PATH=""

CMD ["python", "-u", "/workspace/handler.py"]
