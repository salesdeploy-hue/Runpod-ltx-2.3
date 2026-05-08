"""Monkey-patch LTXVGemmaCLIPModelLoader to load Gemma in 4-bit when
LOW_VRAM_MODE=q4. Without this, the loader uses bf16 (~24GB) regardless
of which checkpoint dir we point it at — even if we hand it the
QAT-Q4 weights, transformers will dequantize to bf16 unless we pass
``BitsAndBytesConfig`` at load time.

The patch wraps ``Gemma3ForConditionalGeneration.from_pretrained`` to
inject a 4-bit nf4 config. nf4 (NormalFloat4) is the empirically best
4-bit type for transformer weights; it costs <1% quality vs fp16 on
benchmarks and drops Gemma 12B from 24GB → ~7GB.

Active only when LOW_VRAM_MODE=q4 in env. Otherwise this module is a
no-op (the loader behaves exactly as upstream).

Importing this file has the side-effect of patching — we run it from
start_with_models.sh BEFORE ComfyUI imports the LTXV node modules.
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

_LOW_VRAM_MODE = (os.environ.get("LOW_VRAM_MODE") or "q4").lower().strip()


def _apply_patch() -> None:
    if _LOW_VRAM_MODE != "q4":
        logger.info("[gemma_loader_patch] LOW_VRAM_MODE=%s — no patch", _LOW_VRAM_MODE)
        return

    try:
        import torch  # noqa: F401
        from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration
    except Exception as e:
        logger.warning(
            "[gemma_loader_patch] missing deps for q4 mode (%s) — falling "
            "back to bf16. Install bitsandbytes + accelerate.", e,
        )
        return

    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    original_from_pretrained = Gemma3ForConditionalGeneration.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        # Only patch the LTXV loader's call. The signature is
        # ``from_pretrained(encoder_path, local_files_only=True,
        #   torch_dtype=dtype)`` — when those exact kwargs match,
        # we're in the LTXV code path and we add the quant config.
        if (
            kwargs.get("local_files_only") is True
            and "quantization_config" not in kwargs
        ):
            kwargs["quantization_config"] = nf4
            kwargs.pop("torch_dtype", None)  # bnb manages dtype
            logger.info(
                "[gemma_loader_patch] injecting nf4 quant for Gemma load"
            )
        return original_from_pretrained(*args, **kwargs)

    Gemma3ForConditionalGeneration.from_pretrained = patched_from_pretrained
    logger.info("[gemma_loader_patch] applied: Gemma loads in nf4 (~7GB)")


_apply_patch()


# Allow running as a one-shot script (start hook calls
# ``python -c "import gemma_loader_patch"``).
if __name__ == "__main__":
    print(
        f"[gemma_loader_patch] mode={_LOW_VRAM_MODE} "
        f"(patch {'active' if _LOW_VRAM_MODE == 'q4' else 'no-op'})",
        flush=True,
    )
    sys.exit(0)
