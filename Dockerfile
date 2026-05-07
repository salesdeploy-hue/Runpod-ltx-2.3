# AdsXFlow LTX-2.3 RunPod Serverless worker.
#
# Base: official PyTorch image (CUDA 12.4, py3.11, torch 2.4.1).
# Why not runpod/base: runpod/base only ships `python3` (no `python`),
# its torch wheels stack varies by tag, and the GitHub-deploy flow
# kept failing silently against it. The official PyTorch image is
# what every production-tier RunPod template (incl. the public
# HunyuanVideo / Wan-2.1 templates) uses.

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        ffmpeg \
        curl \
        libsm6 \
        libxext6 \
        libgl1 \
        build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Worker-side deps installed FIRST (clean torch + transformers stack
# already supplied by the base image). The base ships torch==2.4.1 +
# matching CUDA wheels — DO NOT touch those.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /workspace/requirements.txt

# Clone LTX-2 + install ltx-core + ltx-pipelines from source with
# --no-deps so they don't yank our torch wheel.
ARG LTX2_REF=main
RUN git clone --depth 1 --branch ${LTX2_REF} https://github.com/Lightricks/LTX-2.git /workspace/LTX-2 \
    && cd /workspace/LTX-2 \
    && pip install --no-cache-dir --no-deps -e packages/ltx-core \
    && pip install --no-cache-dir --no-deps -e packages/ltx-pipelines

# Copy handler + entry point
COPY handler.py /workspace/handler.py

# Default model paths — handler creates the dir on first run.
ENV LTX_MODELS_ROOT=/runpod-volume/models \
    LTX_FALLBACK_MODELS_ROOT=/workspace/models \
    LTX_REPO=Lightricks/LTX-2.3 \
    GEMMA_REPO=google/gemma-3-1b-pt \
    HF_HUB_ENABLE_HF_TRANSFER=1

# RunPod Serverless workers don't strictly need an EXPOSE, but
# matching the canonical examples helps the platform's healthcheck.
EXPOSE 5000

# Light healthcheck: just verify the python process can import torch.
# A heavy model-load healthcheck would block /run during the first
# call. The runpod SDK + container itself is the actual readiness.
HEALTHCHECK --interval=60s --timeout=10s --retries=3 --start-period=120s \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["python", "-u", "/workspace/handler.py"]
