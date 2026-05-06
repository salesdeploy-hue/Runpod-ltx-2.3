FROM runpod/base:1.0.3-cuda1281-ubuntu2204

WORKDIR /workspace

# System deps for video encode + git for cloning LTX-2.
# (ffmpeg + libgl1 are already in the base image but we re-declare so
# this Dockerfile is portable to other bases.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        git \
        git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# The runpod/base image uses `python3` (not `python`) — the previous
# Dockerfile failed at exit 127 because the venv-style `python` symlink
# isn't there. Make a tiny wrapper alias-script for any tooling that
# expects the bare name (LTX-2's optional `uv sync` invokes `python`),
# then use `python3` explicitly elsewhere for clarity.
RUN ln -sf "$(which python3)" /usr/local/bin/python

# Latest pip (the base ships an older one) — kept minimal: no `uv` here.
# `uv sync --frozen` from LTX-2's docs is for a dev env; we install
# `ltx-core` + `ltx-pipelines` editable via plain pip below, which is
# what RunPod's serverless runtime actually needs.
RUN python3 -m pip install --upgrade pip

# Clone LTX-2 and install the ltx-pipelines + ltx-core packages from
# source. Pinned to `main`; bump LTX2_REF after testing a newer tag.
ARG LTX2_REF=main
RUN git clone --depth 1 --branch ${LTX2_REF} https://github.com/Lightricks/LTX-2.git /workspace/LTX-2 \
    && cd /workspace/LTX-2 \
    && python3 -m pip install --no-cache-dir -e packages/ltx-core \
    && python3 -m pip install --no-cache-dir -e packages/ltx-pipelines

# Worker SDK + the few utilities the handler imports directly.
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

# ─── Pre-download checkpoints into the image ────────────────────────────
# Bake the ~47GB of weights into the image at build time so cold-starts
# don't pay the HF Hub bandwidth on every worker boot. Done in three
# separate RUN steps so each download has its own layer — if one fails
# (e.g. transient HF rate-limit) the build can resume from the next
# layer instead of redoing the entire fetch.
#
# Lightricks/LTX-2.3 is *public* (verified via HF API), so the LTX
# downloads need no token. Only Gemma is gated → see step 3.

RUN mkdir -p /workspace/models /workspace/models/gemma

ARG LTX_REPO=Lightricks/LTX-2.3
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 1. LTX-2.3 22B distilled-1.1 base checkpoint (~44GB)
RUN python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='${LTX_REPO}', filename='ltx-2.3-22b-distilled-1.1.safetensors', local_dir='/workspace/models')"

# 2. Spatial upscaler ×2 v1.1 (~1GB)
RUN python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='${LTX_REPO}', filename='ltx-2.3-spatial-upscaler-x2-1.1.safetensors', local_dir='/workspace/models')"

# 3. Gemma 3 1B text encoder (~2GB) — GATED on HuggingFace.
#    Pass HF_TOKEN as a Docker build arg (RunPod console → endpoint
#    settings → Container env). If unset we skip — handler will
#    download lazily on first job using a runtime-injected HF_TOKEN.
ARG HF_TOKEN=""
ENV HF_TOKEN=$HF_TOKEN
ARG GEMMA_REPO=google/gemma-3-1b-pt
RUN if [ -n "$HF_TOKEN" ]; then \
        python3 -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${GEMMA_REPO}', local_dir='/workspace/models/gemma', token=os.environ['HF_TOKEN'])"; \
    else \
        echo "⚠️  HF_TOKEN not set at build time — Gemma will be downloaded lazily on first job. Set HF_TOKEN as a build env var to bake it into the image."; \
    fi

COPY handler.py /workspace/handler.py

# Default env points the handler at the baked checkpoints. Operators
# can override any of these on the RunPod endpoint config to point
# at a network volume cache instead.
ENV LTX_CHECKPOINT_PATH=/workspace/models/ltx-2.3-22b-distilled-1.1.safetensors \
    LTX_SPATIAL_UPSAMPLER_PATH=/workspace/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    LTX_GEMMA_ROOT=/workspace/models/gemma \
    LTX_DISTILLED_LORA_PATH=""

CMD ["python3", "-u", "/workspace/handler.py"]
