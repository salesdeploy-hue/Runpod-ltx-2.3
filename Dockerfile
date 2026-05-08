# AdsXFlow LTX-2.3 RunPod Serverless worker — low-VRAM ComfyUI build.
#
# Targets 32GB VRAM cards (RTX 5000 Ada, RTX 5090, L40S, A100 80GB) as
# the floor; experimental support for 24GB (RTX 4090) via bitsandbytes
# 4-bit Gemma quantization and aggressive offloading.
#
# This image stays SLIM (~3 GB). Weight downloads happen at first
# worker boot, persisted on the attached network volume so subsequent
# cold starts mount-and-go.
#
# Architecture:
#   FROM worker-comfyui (handler + ComfyUI server + websocket plumbing)
#   + Lightricks ComfyUI-LTXVideo nodes (low_vram_loaders, tiled VAE)
#   + bitsandbytes for runtime int4 Gemma quant (LOW_VRAM_MODE=true)
#   + start hook that:
#       1. ensures models present on /comfyui/models (symlinked from
#          /runpod-volume/models when a network volume is attached)
#       2. boots ComfyUI with --reserve-vram 5 (leaves headroom for
#          VAE decode + activations)

FROM runpod/worker-comfyui:5.8.5-base-cuda12.8.1

ENV DEBIAN_FRONTEND=noninteractive

# ─── Install Lightricks/ComfyUI-LTXVideo custom nodes ──────────────────
# Provided by the base image's helper. Includes:
#   - LTXVGemmaCLIPModelLoader (text encoder)
#   - LowVRAMCheckpointLoader / LowVRAMAudioVAELoader / LowVRAMLatentUpscaleModelLoader
#   - LTXVTiledVAEDecode / LTXVTiledSampler
#   - IC-LoRA Union Control / HDR LoRA nodes
RUN comfy-node-install ComfyUI-LTXVideo

# Extra Python deps:
#   - av/imageio: video encode/decode in CreateVideo + SaveVideo
#   - huggingface_hub: lazy download in start hook
#   - bitsandbytes: optional 4-bit Gemma quant for the 24GB experimental
#     path (the LTXVGemmaCLIPModelLoader doesn't pass quant config, so
#     we monkey-patch it via gemma_loader_patch.py when LOW_VRAM_MODE=q4)
#   - accelerate: required by transformers for big-model offload
RUN /opt/venv/bin/pip install --no-cache-dir \
    "av>=13.0.0" \
    "imageio[ffmpeg]>=2.36.0" \
    "imageio-ffmpeg>=0.5.1" \
    "huggingface_hub>=0.26.0" \
    "bitsandbytes>=0.45.0" \
    "accelerate>=1.0.0"

# ─── Lazy-download + low-VRAM startup hook ────────────────────────────
COPY scripts/ensure_models.py /usr/local/bin/ensure_models.py
COPY scripts/start_with_models.sh /usr/local/bin/start_with_models.sh
COPY scripts/gemma_loader_patch.py /usr/local/bin/gemma_loader_patch.py
RUN chmod +x /usr/local/bin/ensure_models.py \
              /usr/local/bin/start_with_models.sh

# Override base CMD to run our hook first, then exec the worker-comfyui
# /start.sh with our extra ComfyUI args injected.
CMD ["/usr/local/bin/start_with_models.sh"]

# ─── Worker config ─────────────────────────────────────────────────────
# Everything ships QUANTIZED by default:
#   - LTX 2.3 → fp8 distilled (~22GB on disk, ~35GB peak VRAM)
#   - Gemma 3 12B → unsloth bnb-4bit (~7GB on disk, ~7GB VRAM)
#
# Unsloth's bnb-4bit safetensors include a quantization_config in the
# model's config.json, so HF transformers' Gemma3ForConditionalGeneration
# .from_pretrained engages bitsandbytes automatically — no monkey-patch
# needed (we keep the patch script for legacy LOW_VRAM_MODE=q4 paths
# but it's a no-op now).
#
# Lightricks/* gemma repos are GATED on HF (manual approval). Unsloth's
# repos are PUBLIC. That's why we use unsloth.
ENV REFRESH_WORKER=false \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    LTX_REPO=Lightricks/LTX-2.3 \
    LTX_FP8_REPO=Lightricks/LTX-2.3-fp8 \
    GEMMA_REPO=unsloth/gemma-3-12b-it-bnb-4bit \
    LOW_VRAM_MODE=q4 \
    COMFY_RESERVE_VRAM=5
