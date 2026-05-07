# Slim LTX-2.3 worker image — no weights baked in.
#
# Why slim: the bake-everything image was ~50GB. Cold starts had to
# pull all of that on every worker boot, and parallel pulls saturate
# RunPod's internal network → workers stuck in `initializing` for
# 30+ min. This image is ~5GB. Weights are lazy-downloaded on first
# job into /runpod-volume/models/ (RunPod's standard mount path),
# which persists across cold starts when a network volume is
# attached to the endpoint.
#
# First-ever job (no cached weights): ~5-10 min download + render.
# Every subsequent cold start: ~30s mount + model load + render.

FROM runpod/base:1.0.3-cuda1281-ubuntu2204

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        git \
        git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# runpod/base ships only `python3` (no `python` symlink). Make a
# symlink so any tool expecting the bare `python` works (LTX-2's
# package config invokes `python` in some places).
RUN ln -sf "$(which python3)" /usr/local/bin/python

RUN python3 -m pip install --upgrade pip

# Clone LTX-2 + install ltx-core + ltx-pipelines from source. Pinned
# to `main`; bump LTX2_REF after testing a tag.
ARG LTX2_REF=main
RUN git clone --depth 1 --branch ${LTX2_REF} https://github.com/Lightricks/LTX-2.git /workspace/LTX-2 \
    && cd /workspace/LTX-2 \
    && python3 -m pip install --no-cache-dir -e packages/ltx-core \
    && python3 -m pip install --no-cache-dir -e packages/ltx-pipelines

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

# Default model paths point inside the network volume that the
# operator attaches at endpoint create time (mounted at
# /runpod-volume by RunPod's runtime). The handler creates the
# /runpod-volume/models dir on first run and downloads weights.
# When no volume is attached, falls back to /workspace/models which
# lives on the worker's container disk and is wiped on terminate.
ENV LTX_MODELS_ROOT=/runpod-volume/models \
    LTX_FALLBACK_MODELS_ROOT=/workspace/models \
    LTX_REPO=Lightricks/LTX-2.3 \
    GEMMA_REPO=google/gemma-3-1b-pt \
    HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["python3", "-u", "/workspace/handler.py"]
