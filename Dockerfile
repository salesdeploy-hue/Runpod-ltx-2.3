# AdsXFlow LTX-2.3 RunPod Serverless worker.
#
# Versions pinned to MATCH Lightricks/LTX-2's uv.lock exactly. Each
# loose pin we tried previously drifted past their tested set and
# broke at runtime (SiglipVisionModel API drift in transformers >=4.55,
# accelerate API changes, etc.).

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

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

# runpod/base ships only `python3`; LTX-2 packages invoke `python`
# in some build paths.
RUN ln -sf "$(which python3)" /usr/local/bin/python

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel

# CRITICAL ORDER:
# 1. Install our pinned requirements FIRST. These match Lightricks'
#    uv.lock exactly so the LTX-2 install step (next) sees its deps
#    already satisfied and doesn't re-resolve to newer breaking
#    versions.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# 2. Now install LTX-2 packages. Use --no-deps because every dep is
#    already pinned in step 1; --no-deps prevents pip's resolver from
#    yanking transformers/accelerate/etc to newer versions.
ARG LTX2_REF=main
RUN git clone --depth 1 --branch ${LTX2_REF} https://github.com/Lightricks/LTX-2.git /workspace/LTX-2 \
    && cd /workspace/LTX-2 \
    && pip install --no-cache-dir --no-deps -e packages/ltx-core \
    && pip install --no-cache-dir --no-deps -e packages/ltx-pipelines

COPY handler.py /workspace/handler.py

ENV LTX_MODELS_ROOT=/workspace/models \
    LTX_FALLBACK_MODELS_ROOT=/workspace/models \
    LTX_REPO=Lightricks/LTX-2.3 \
    GEMMA_REPO=Lightricks/gemma-3-12b-it-qat-q4_0-unquantized \
    HF_HUB_ENABLE_HF_TRANSFER=0

EXPOSE 5000

# Healthcheck verifies CUDA + transformers are importable. If either
# breaks the rolling update never completes — fail fast, not silent.
HEALTHCHECK --interval=60s --timeout=10s --retries=3 --start-period=180s \
    CMD python -c "import torch, transformers; assert torch.cuda.is_available()" || exit 1

CMD ["python", "-u", "/workspace/handler.py"]
