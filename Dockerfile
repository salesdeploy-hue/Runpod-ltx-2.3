# Minimal diagnostic image — verify RunPod can spawn workers from
# our repo at all. NO LTX, NO heavy deps. Just runpod SDK + torch
# (already in base) + the hello-world handler.

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Just the runpod SDK + minimal extras. Skip the entire LTX install
# for this diagnostic pass.
RUN pip install --no-cache-dir runpod==1.7.13

COPY handler.py /workspace/handler.py

EXPOSE 5000
CMD ["python", "-u", "/workspace/handler.py"]
