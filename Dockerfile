# AdsXFlow LTX-2.3 RunPod Serverless worker.
#
# Base: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
#
# Why this exact tag (root cause of all the silent worker crashes):
# Lightricks LTX-2's `ltx-core` pyproject.toml pins
#   - torch ~= 2.7  (CUDA 12.9 wheel index)
# Earlier attempts used torch 2.4 / CUDA 12.4 base + `pip install
# --no-deps` to suppress the conflict — that "succeeded" at build
# time but the workers crashed immediately at runtime when
# ltx-pipelines imported torch 2.7-only attributes. CUDA 12.8 ⇄ 12.9
# wheels are forward-compatible, so torch 2.7.1 + CUDA 12.8 is the
# right pairing here (12.9 base images aren't published on Docker Hub
# yet by the official PyTorch project).

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

WORKDIR /workspace

# Worker-side requirements (runpod SDK + transformers / huggingface
# hub etc.). Base image already has torch 2.7.1; do NOT touch it.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /workspace/requirements.txt

# Install LTX-2 packages WITH their deps. Since torch 2.7.1 is
# already present and matches `torch~=2.7`, pip will reuse the
# pre-installed wheel and only fetch the remaining deps (av, einops,
# scipy, etc.). No --no-deps trick — that was the silent-crash bug.
ARG LTX2_REF=main
RUN git clone --depth 1 --branch ${LTX2_REF} https://github.com/Lightricks/LTX-2.git /workspace/LTX-2 \
    && cd /workspace/LTX-2 \
    && pip install --no-cache-dir -e packages/ltx-core \
    && pip install --no-cache-dir -e packages/ltx-pipelines

COPY handler.py /workspace/handler.py

ENV LTX_MODELS_ROOT=/runpod-volume/models \
    LTX_FALLBACK_MODELS_ROOT=/workspace/models \
    LTX_REPO=Lightricks/LTX-2.3 \
    GEMMA_REPO=google/gemma-3-1b-pt \
    HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 5000

# Light healthcheck — verify cuda is reachable. If the GPU isn't
# bound we want to fail fast and surface that in console logs.
HEALTHCHECK --interval=60s --timeout=10s --retries=3 --start-period=180s \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["python", "-u", "/workspace/handler.py"]
