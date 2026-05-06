"""Reference RunPod Serverless handler for LTX-2.3.

LTX-2.3 (released Jan 2026) is the 22B audio-video foundation model
from Lightricks. Vs the 13B LTX-Video 1.x:

  * 22B params instead of 13B → needs ≥48GB VRAM (L40S minimum;
    A100 80GB / H100 80GB recommended for headroom).
  * **Native synchronized audio** — generates a single MP4 with
    audio baked in. No separate VO step needed for talking-head /
    documentary archetypes.
  * Native portrait (9:16) at up to 1080p.
  * Better prompt adherence (gated attention text connector).
  * Uses Lightricks' native `ltx_pipelines.TI2VidTwoStagesPipeline`
    instead of diffusers — diffusers support is "coming soon" but
    the model_index.json is missing as of 2026-05.

I/O contract is unchanged from the LTX 1.x handler — the AdsXFlow
client (`app/services/ltx_video_provider.py`) speaks the same
schema and our `_decode_video_output` accepts the same shapes.
This means the registry tile in Studio just gets a label / cost /
capability flag refresh; no client-side change is required.

Deploy on RunPod:
  1. Push this directory + Dockerfile to a Git repo.
  2. RunPod → Serverless → New Endpoint → Deploy from GitHub.
  3. GPU: **L40S 48GB minimum**, A100 80GB recommended.
     Min workers: 0 (scale-to-zero).
     Max workers: 5.
     Idle timeout: 30s.
     Execution timeout: 300s.
     FlashBoot: ON.
  4. After deploy, copy the endpoint id into AdsXFlow .env as
     `RUNPOD_LTX_ENDPOINT_ID`.

Cost shape (L40S Flex, distilled-1.1 at 8 inference steps):
  - 8s clip render: ~30-40s GPU → ~$0.018 per beat.
  - Full 24s reel: 3 × ~$0.018 = ~$0.055 per reel WITH AUDIO.
  - vs Veo 3.1 Fast ($3.60 per 24s reel): ~65× cheaper.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import runpod  # type: ignore
import torch  # type: ignore


# Singletons survive across warm invocations of the same worker
# container — model load (~5-15s on a warm filesystem cache) only
# happens once per cold start, not once per render.
_PIPELINE = None
_PIPELINE_LOAD_ERROR: str | None = None


# ─── Model paths — set via env vars, defaults match Dockerfile cache ──
#
# The Dockerfile pre-downloads these into /workspace/models so the
# first job after a cold boot doesn't pay the 50GB+ HF Hub fetch.
# Operators can override any of these by setting the env var on the
# RunPod endpoint config (e.g. to point at a network-volume cache).
_CHECKPOINT_PATH = os.environ.get(
    "LTX_CHECKPOINT_PATH",
    "/workspace/models/ltx-2.3-22b-distilled-1.1.safetensors",
)
_SPATIAL_UPSAMPLER_PATH = os.environ.get(
    "LTX_SPATIAL_UPSAMPLER_PATH",
    "/workspace/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
)
_GEMMA_ROOT = os.environ.get(
    "LTX_GEMMA_ROOT",
    "/workspace/models/gemma",
)
# Distilled-1.1 ships as a LoRA over the dev base; specify path(s)
# here when using the full pipeline. Empty when running the
# all-in-one distilled checkpoint above.
_DISTILLED_LORA_PATH = os.environ.get("LTX_DISTILLED_LORA_PATH", "")


def _load_pipeline():
    """Instantiate the LTX-2.3 production pipeline ONCE per container.

    Raises a RuntimeError that gets captured into the response if any
    checkpoint is missing — the operator sees a clear "missing path"
    error instead of a cryptic torch traceback in the worker logs.
    """
    global _PIPELINE, _PIPELINE_LOAD_ERROR
    if _PIPELINE is not None:
        return _PIPELINE
    if _PIPELINE_LOAD_ERROR is not None:
        # Re-raise the cached error so we don't keep retrying a
        # broken load on every job (~30s per attempt on a 22B model).
        raise RuntimeError(_PIPELINE_LOAD_ERROR)

    try:
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline  # type: ignore
    except ImportError as e:
        msg = (
            f"ltx_pipelines is not installed in the container. "
            f"Build the Docker image with the LTX-2 repo cloned + "
            f"`uv sync --frozen` against packages/ltx-pipelines. "
            f"Underlying ImportError: {e}"
        )
        _PIPELINE_LOAD_ERROR = msg
        raise RuntimeError(msg)

    for label, path in (
        ("LTX_CHECKPOINT_PATH", _CHECKPOINT_PATH),
        ("LTX_SPATIAL_UPSAMPLER_PATH", _SPATIAL_UPSAMPLER_PATH),
    ):
        if not Path(path).exists():
            msg = (
                f"{label}={path} not found inside the container. "
                f"Either bake the checkpoint into the image at build "
                f"time (recommended) or mount a network volume "
                f"containing it. See the Dockerfile."
            )
            _PIPELINE_LOAD_ERROR = msg
            raise RuntimeError(msg)
    if not Path(_GEMMA_ROOT).exists():
        msg = (
            f"LTX_GEMMA_ROOT={_GEMMA_ROOT} not found. LTX-2.3 uses "
            f"Gemma as the text encoder; download it at image-build "
            f"time."
        )
        _PIPELINE_LOAD_ERROR = msg
        raise RuntimeError(msg)

    distilled_lora_arg: list = []
    if _DISTILLED_LORA_PATH:
        # ltx_pipelines accepts a list of LoRA paths to stack on top
        # of the base checkpoint. Distilled-1.1 already bakes the
        # LoRA into a merged safetensors above; this is for operators
        # who chose the dev + LoRA layout instead.
        distilled_lora_arg = [p for p in _DISTILLED_LORA_PATH.split(",") if p]

    _PIPELINE = TI2VidTwoStagesPipeline(
        checkpoint_path=_CHECKPOINT_PATH,
        distilled_lora=distilled_lora_arg,
        spatial_upsampler_path=_SPATIAL_UPSAMPLER_PATH,
        gemma_root=_GEMMA_ROOT,
    )
    return _PIPELINE


# ─── Helpers ───────────────────────────────────────────────────────────────


def _decode_image_b64_to_path(b64: str, dest: Path) -> None:
    """Decode the inbound base64 first frame to a JPG file on disk —
    the LTX pipeline's ImageConditioningInput takes a path, not bytes.
    """
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    raw = base64.b64decode(b64, validate=False)
    from PIL import Image  # type: ignore
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.save(dest, format="JPEG", quality=92)


def _parse_resolution(res: str) -> tuple[int, int]:
    """LTX-2.3 requires width + height divisible by 32. Snap to the
    nearest valid pair and clamp to 256-1920."""
    try:
        w, h = res.lower().split("x")
        w_i, h_i = int(w), int(h)
    except Exception:
        return 768, 1280
    w_i = max(256, min(1920, (w_i // 32) * 32))
    h_i = max(256, min(1920, (h_i // 32) * 32))
    return w_i, h_i


def _frames_for_duration(duration_s: int, fps: float) -> int:
    """LTX-2.3 frame-count rule: (num_frames - 1) divisible by 8.

    25fps × 4s = 100 → snap to 97 (96+1) or 105 (104+1). Pick nearest
    that respects the rule and is in the practical 33-241 range
    (~1.3s to ~9.6s of footage).
    """
    raw = max(9, int(round(duration_s * fps)))
    candidates = [n for n in range(max(33, raw - 12), min(241, raw + 12) + 1) if (n - 1) % 8 == 0]
    if not candidates:
        # Fallback — round to nearest valid count globally.
        return ((raw - 1) // 8) * 8 + 1
    return min(candidates, key=lambda n: abs(n - raw))


# ─── Handler ───────────────────────────────────────────────────────────────


def handler(event: dict[str, Any]) -> dict[str, Any]:
    """RunPod entry point. Receives `event['input']` per RunPod
    convention; returns the I/O-contract shape documented at the top
    of `app/services/ltx_video_provider.py`."""
    try:
        inp = event.get("input") or {}
        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            return {"error": "input.prompt is required"}

        negative = (inp.get("negative_prompt") or "").strip() or None
        duration_s = int(inp.get("duration_seconds") or 8)
        fps = float(inp.get("fps") or 25.0)              # LTX 2.3 default
        steps = int(inp.get("num_inference_steps") or 8) # distilled = 8
        guidance = float(inp.get("guidance_scale") or 1.0)  # CFG=1 for distilled
        seed = int(inp.get("seed") or 0)
        resolution = inp.get("resolution") or "768x1280"
        first_frame_b64 = inp.get("first_frame_b64") or ""

        width, height = _parse_resolution(resolution)
        num_frames = _frames_for_duration(duration_s, fps)

        with tempfile.TemporaryDirectory() as work_dir_str:
            work_dir = Path(work_dir_str)

            # Image conditioning input — empty list for text-to-video,
            # one entry for image-to-video (first frame).
            images = []
            if first_frame_b64:
                from ltx_core.components.guiders import (  # type: ignore
                    ImageConditioningInput,
                )
                ff_path = work_dir / "first_frame.jpg"
                _decode_image_b64_to_path(first_frame_b64, ff_path)
                # ImageConditioningInput(path, frame_index, strength, crf)
                # frame_index=0 anchors at the start; strength=1.0 = hard
                # lock (the inbound frame appears verbatim); crf=33 is
                # the video-codec hint for the conditioning pass.
                images = [ImageConditioningInput(str(ff_path), 0, 1.0, 33)]

            output_path = work_dir / "ltx23_output.mp4"

            pipeline = _load_pipeline()

            if seed:
                # The LTX pipeline reads from torch's global RNG; seed
                # both CUDA and CPU streams for reproducibility.
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # Build the pipeline kwargs. We pass `negative_prompt` and
            # `cfg_scale` only when meaningful — the distilled model
            # skips CFG (cfg_scale=1.0) and ignores negative prompts.
            call_kwargs: dict[str, Any] = dict(
                prompt=prompt,
                output_path=str(output_path),
                images=images,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=steps,
            )
            if negative and guidance > 1.01:
                call_kwargs["negative_prompt"] = negative
            if guidance > 1.01:
                # MultiModalGuiderParams is the LTX-2 way of passing
                # CFG; we set it via the dict-friendly kwarg path so
                # the worker doesn't have to import the dataclass.
                call_kwargs["cfg_scale"] = guidance

            pipeline(**call_kwargs)

            if not output_path.exists() or output_path.stat().st_size < 1024:
                return {
                    "error": (
                        f"Pipeline produced no output (or empty file) at "
                        f"{output_path}. Check worker logs."
                    ),
                }

            video_bytes = output_path.read_bytes()

        return {
            "video": {
                "type": "base64",
                "data": base64.b64encode(video_bytes).decode("ascii"),
            },
            "frames": num_frames,
            "seed_used": seed,
            "resolution": f"{width}x{height}",
            "duration_seconds": duration_s,
            "fps": fps,
            "model": "ltx-2.3-22b-distilled-1.1",
            "audio_baked_in": True,
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-3000:],
        }


# Module-level entrypoint — RunPod's GitHub deploy scanner greps the
# default branch for `runpod.serverless.start(` at top-level scope to
# auto-detect the worker's entry point. Wrapping this call in
# `if __name__ == "__main__":` defeats that scan and triggers the
# "Could not find runpod.serverless.start() in your default branch"
# error on the New Endpoint page.
runpod.serverless.start({"handler": handler})
