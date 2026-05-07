"""LTX-2.3 RunPod Serverless handler — slim image + lazy weights.

Why this shape (vs the previous bake-50GB-into-image approach):
  - Image stays small (~5GB) so cold starts pull fast (~30s) instead
    of stalling for 30+ min on a parallel-pull bandwidth fight.
  - Weights download once into /runpod-volume/models/ (RunPod's
    standard mount path for network volumes). Subsequent cold starts
    just remount the volume → re-use the cached weights.
  - No HF_TOKEN required at build time. The runtime env on the
    template provides it; we only hit HF Hub on first run.

I/O contract is unchanged from the prior handler — the AdsXFlow
client (`app/services/ltx_video_provider.py`) speaks the same schema.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import runpod  # type: ignore
import torch  # type: ignore


# ─── Singletons (warm-call reuse) ──────────────────────────────────────────
_PIPELINE = None
_PIPELINE_LOAD_ERROR: str | None = None
_LOAD_LOCK = threading.Lock()


# ─── Resolve where weights live ────────────────────────────────────────────
# Prefer the network volume if attached; fall back to the worker's
# container disk. We compute MODELS_ROOT once at module load.

def _pick_models_root() -> Path:
    primary = Path(os.environ.get("LTX_MODELS_ROOT", "/runpod-volume/models"))
    fallback = Path(os.environ.get("LTX_FALLBACK_MODELS_ROOT", "/workspace/models"))
    # If /runpod-volume exists, use it (volume is mounted) even if
    # /runpod-volume/models doesn't yet — we'll create it.
    parent = primary.parent
    if parent.exists() and parent.is_dir():
        primary.mkdir(parents=True, exist_ok=True)
        return primary
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


MODELS_ROOT = _pick_models_root()


def _file_complete(p: Path, *, min_size_bytes: int) -> bool:
    """Sanity-check a downloaded file. Avoid loading half-pulled
    safetensors — they'd raise opaque parse errors deep inside the
    pipeline and the operator would chase the wrong lead."""
    try:
        return p.exists() and p.stat().st_size >= min_size_bytes
    except OSError:
        return False


def _ensure_weights() -> dict[str, Path]:
    """Download (or reuse cached) LTX + Gemma weights.

    Order:
      1. LTX-2.3 distilled-1.1 base checkpoint (~44GB)
      2. Spatial upscaler ×2 v1.1 (~1GB)
      3. Gemma 3 1B text encoder (~2GB) — gated, needs HF_TOKEN

    Returns a dict of resolved paths. Idempotent — `hf_hub_download`
    is a no-op when the file already exists at the target with
    matching size + sha256.
    """
    from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore

    ltx_repo = os.environ.get("LTX_REPO", "Lightricks/LTX-2.3")
    gemma_repo = os.environ.get("GEMMA_REPO", "google/gemma-3-1b-pt")
    hf_token = os.environ.get("HF_TOKEN") or None

    ckpt_name = "ltx-2.3-22b-distilled-1.1.safetensors"
    upscaler_name = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    gemma_dir = MODELS_ROOT / "gemma"

    ckpt_path = MODELS_ROOT / ckpt_name
    upscaler_path = MODELS_ROOT / upscaler_name

    # 1. LTX 22B distilled-1.1
    if not _file_complete(ckpt_path, min_size_bytes=20 * 1024 ** 3):  # >20GB
        print(f"[handler] downloading {ckpt_name} → {MODELS_ROOT}", flush=True)
        hf_hub_download(
            repo_id=ltx_repo, filename=ckpt_name,
            local_dir=str(MODELS_ROOT),
        )

    # 2. Spatial upscaler
    if not _file_complete(upscaler_path, min_size_bytes=500 * 1024 * 1024):  # >500MB
        print(f"[handler] downloading {upscaler_name}", flush=True)
        hf_hub_download(
            repo_id=ltx_repo, filename=upscaler_name,
            local_dir=str(MODELS_ROOT),
        )

    # 3. Gemma multimodal text encoder. LTX-2.3 needs the multimodal
    #    Gemma 3 (loads `preprocessor_config.json` via Gemma3Processor
    #    for image-conditioning). Default GEMMA_REPO is now
    #    `Lightricks/gemma-3-12b-it-qat-q4_0-unquantized` (~22.7GB) —
    #    open + multimodal — so HF_TOKEN is optional. We pass it if
    #    present (lets operators substitute the gated google/gemma-3-4b-pt),
    #    but no longer fail when absent.
    #    Marker is preprocessor_config.json (the multimodal hint),
    #    not config.json (text-only Gemmas have that too).
    gemma_marker = gemma_dir / "preprocessor_config.json"
    if not gemma_marker.exists():
        print(f"[handler] downloading {gemma_repo} → {gemma_dir}", flush=True)
        gemma_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=gemma_repo,
            local_dir=str(gemma_dir),
            token=hf_token,  # None is fine for ungated repos
        )

    return {
        "checkpoint": ckpt_path,
        "spatial_upsampler": upscaler_path,
        "gemma_root": gemma_dir,
    }


def _load_pipeline():
    """Build the LTX-2.3 production pipeline once per warm container.

    Cached failure: if a load attempt fails, we cache the error and
    re-raise on every subsequent call so we don't keep retrying the
    same expensive load (model load is ~30s on a warm machine).
    """
    global _PIPELINE, _PIPELINE_LOAD_ERROR
    if _PIPELINE is not None:
        return _PIPELINE
    with _LOAD_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        if _PIPELINE_LOAD_ERROR:
            raise RuntimeError(_PIPELINE_LOAD_ERROR)
        try:
            from ltx_pipelines.ti2vid_two_stages import (  # type: ignore
                TI2VidTwoStagesPipeline,
            )
        except ImportError as e:
            msg = (
                "ltx_pipelines is not installed. The Dockerfile must "
                f"`pip install -e packages/ltx-pipelines`. ImportError: {e}"
            )
            _PIPELINE_LOAD_ERROR = msg
            raise RuntimeError(msg)

        try:
            paths = _ensure_weights()
        except Exception as e:
            msg = f"Weight download failed: {type(e).__name__}: {e}"
            _PIPELINE_LOAD_ERROR = msg
            raise RuntimeError(msg)

        try:
            # The TI2VidTwoStagesPipeline signature requires BOTH
            # `distilled_lora` AND `loras`. We discovered this the
            # hard way — first error was 'missing loras', then after
            # adding loras the next error was 'missing distilled_lora'.
            # distilled-1.1 ships pre-merged into the base safetensors,
            # so both LoRA lists are empty. Operators stacking
            # additional adapters can populate `loras` at runtime.
            _PIPELINE = TI2VidTwoStagesPipeline(
                checkpoint_path=str(paths["checkpoint"]),
                distilled_lora=[],
                loras=[],
                spatial_upsampler_path=str(paths["spatial_upsampler"]),
                gemma_root=str(paths["gemma_root"]),
            )
        except Exception as e:
            msg = f"Pipeline construction failed: {type(e).__name__}: {e}"
            _PIPELINE_LOAD_ERROR = msg
            raise RuntimeError(msg)
        return _PIPELINE


# ─── Frame / image helpers ─────────────────────────────────────────────────


def _decode_image_b64_to_path(b64: str, dest: Path) -> None:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    raw = base64.b64decode(b64, validate=False)
    from PIL import Image  # type: ignore
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.save(dest, format="JPEG", quality=92)


def _parse_resolution(res: str) -> tuple[int, int]:
    try:
        w, h = res.lower().split("x")
        w_i, h_i = int(w), int(h)
    except Exception:
        return 768, 1280
    w_i = max(256, min(1920, (w_i // 32) * 32))
    h_i = max(256, min(1920, (h_i // 32) * 32))
    return w_i, h_i


def _frames_for_duration(duration_s: int, fps: float) -> int:
    """LTX 2.3: (num_frames - 1) divisible by 8."""
    raw = max(9, int(round(duration_s * fps)))
    candidates = [n for n in range(max(33, raw - 12), min(241, raw + 12) + 1) if (n - 1) % 8 == 0]
    if not candidates:
        return ((raw - 1) // 8) * 8 + 1
    return min(candidates, key=lambda n: abs(n - raw))


# ─── Handler ───────────────────────────────────────────────────────────────


def handler(event: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    try:
        inp = event.get("input") or {}
        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            return {"error": "input.prompt is required"}

        negative = (inp.get("negative_prompt") or "").strip() or None
        duration_s = int(inp.get("duration_seconds") or 8)
        fps = float(inp.get("fps") or 25.0)
        steps = int(inp.get("num_inference_steps") or 8)
        guidance = float(inp.get("guidance_scale") or 1.0)
        seed = int(inp.get("seed") or 0)
        resolution = inp.get("resolution") or "768x1280"
        first_frame_b64 = inp.get("first_frame_b64") or ""

        width, height = _parse_resolution(resolution)
        num_frames = _frames_for_duration(duration_s, fps)

        with tempfile.TemporaryDirectory() as work_dir_str:
            work_dir = Path(work_dir_str)

            # ImageConditioningInput is in ltx_pipelines.utils.args (NOT
            # ltx_core.components.guiders — that was an earlier wrong path).
            from ltx_pipelines.utils.args import (  # type: ignore
                ImageConditioningInput,
            )
            from ltx_pipelines.utils.constants import (  # type: ignore
                DEFAULT_NEGATIVE_PROMPT,
                detect_params,
            )
            from ltx_pipelines.utils.media_io import encode_video  # type: ignore
            from ltx_core.model.video_vae import (  # type: ignore
                TilingConfig, get_video_chunks_number,
            )

            images = []
            if first_frame_b64:
                ff_path = work_dir / "first_frame.jpg"
                _decode_image_b64_to_path(first_frame_b64, ff_path)
                # ImageConditioningInput(path, frame_index, strength, crf)
                images = [ImageConditioningInput(str(ff_path), 0, 1.0, 33)]

            t_load_start = time.time()
            pipeline = _load_pipeline()
            t_load_ms = int((time.time() - t_load_start) * 1000)

            # Pull canonical params for this checkpoint. detect_params
            # reads safetensors metadata and picks LTX_2_3_PARAMS (or
            # LTX_2_PARAMS) — already includes the right video/audio
            # guider params and step counts. _ensure_weights() is
            # idempotent + cached, so re-calling it costs nothing.
            paths_dict = _ensure_weights()
            params = detect_params(str(paths_dict["checkpoint"]))
            # Effective inference steps: caller's override or distilled-1.1
            # default of 8 (which is what the operator wants for speed).
            effective_steps = steps if steps else params.num_inference_steps

            output_path = work_dir / "ltx23_output.mp4"
            tiling_config = TilingConfig.default()
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

            t_infer_start = time.time()
            video, audio = pipeline(
                prompt=prompt,
                negative_prompt=negative or DEFAULT_NEGATIVE_PROMPT,
                seed=seed or params.seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=effective_steps,
                video_guider_params=params.video_guider_params,
                audio_guider_params=params.audio_guider_params,
                images=images,
                tiling_config=tiling_config,
                max_batch_size=1,
            )
            # Pipeline returns iterators; encode_video drains them and
            # writes the MP4 to disk (with synchronized audio).
            encode_video(
                video=video,
                fps=fps,
                audio=audio,
                output_path=str(output_path),
                video_chunks_number=video_chunks_number,
            )
            t_infer_ms = int((time.time() - t_infer_start) * 1000)

            if not output_path.exists() or output_path.stat().st_size < 1024:
                return {
                    "error": (
                        f"Pipeline produced no output (or empty file) at "
                        f"{output_path}."
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
            "timing_ms": {
                "model_load_or_lazy_dl": t_load_ms,
                "inference": t_infer_ms,
                "total": int((time.time() - t0) * 1000),
            },
            "models_root": str(MODELS_ROOT),
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-3000:],
            "models_root": str(MODELS_ROOT),
        }


# Module-level entrypoint — RunPod's GitHub deploy scanner greps for
# this string at top-level scope to detect the worker's entry point.
runpod.serverless.start({"handler": handler})
