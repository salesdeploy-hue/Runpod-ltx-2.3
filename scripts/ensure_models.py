"""Lazy-download LTX-2.3 + Gemma weights into ComfyUI's models dir.

Runs once per worker boot via start_with_models.sh, BEFORE ComfyUI
starts. Idempotent — checks for each file's presence + a sane size
threshold and only fetches what's missing.

When /runpod-volume is mounted, we redirect ComfyUI's models dir to
the volume via a symlink, so weights persist across cold starts:
only the first-ever worker pays the ~5-10 min download. Without a
volume, models live on the container disk and re-download per cold
start — works, just slower.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

LTX_FP8_REPO = os.environ.get("LTX_FP8_REPO", "Lightricks/LTX-2.3-fp8")
LTX_REPO = os.environ.get("LTX_REPO", "Lightricks/LTX-2.3")
GEMMA_REPO = os.environ.get(
    "GEMMA_REPO", "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"
)
HF_TOKEN = os.environ.get("HF_TOKEN") or None

COMFY_ROOT = Path("/comfyui")
COMFY_MODELS = COMFY_ROOT / "models"
VOLUME_ROOT = Path("/runpod-volume")
VOLUME_MODELS = VOLUME_ROOT / "models"

# Files we need + min size threshold (bytes) for "looks complete"
REQUIRED = [
    {
        "repo": LTX_FP8_REPO,
        "filename": "ltx-2.3-22b-distilled-fp8.safetensors",
        "subdir": "checkpoints",
        "min_bytes": 20 * 1024 ** 3,
    },
    {
        "repo": LTX_REPO,
        "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "subdir": "upscale_models",
        "min_bytes": 500 * 1024 * 1024,
    },
]


def _redirect_models_to_volume() -> Path:
    """If a network volume is mounted at /runpod-volume, symlink
    /comfyui/models -> /runpod-volume/models so all reads/writes go
    to the persistent volume. Returns the effective models root."""
    if VOLUME_ROOT.is_mount() or VOLUME_ROOT.exists():
        VOLUME_MODELS.mkdir(parents=True, exist_ok=True)
        # Move existing /comfyui/models contents into the volume the
        # first time we see it, so any seed dirs (like ComfyUI's
        # default empty subdirs) are preserved.
        if COMFY_MODELS.exists() and not COMFY_MODELS.is_symlink():
            for child in COMFY_MODELS.iterdir():
                target = VOLUME_MODELS / child.name
                if not target.exists():
                    try:
                        child.rename(target)
                    except Exception:
                        pass
            try:
                COMFY_MODELS.rmdir()
            except OSError:
                # If non-empty for any reason, just remove
                import shutil as _shutil
                _shutil.rmtree(COMFY_MODELS, ignore_errors=True)
        if not COMFY_MODELS.is_symlink():
            COMFY_MODELS.symlink_to(VOLUME_MODELS)
        print(f"[ensure_models] using network volume at {VOLUME_MODELS}", flush=True)
        return VOLUME_MODELS
    print(
        "[ensure_models] no /runpod-volume; using container disk "
        "(weights will re-download on cold start)",
        flush=True,
    )
    COMFY_MODELS.mkdir(parents=True, exist_ok=True)
    return COMFY_MODELS


def _file_complete(p: Path, min_bytes: int) -> bool:
    try:
        return p.exists() and p.stat().st_size >= min_bytes
    except OSError:
        return False


def _download_required(models_root: Path) -> None:
    from huggingface_hub import hf_hub_download  # type: ignore

    for spec in REQUIRED:
        target_dir = models_root / spec["subdir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / spec["filename"]
        if _file_complete(target, spec["min_bytes"]):
            print(
                f"[ensure_models] ✓ cached: {spec['subdir']}/{spec['filename']} "
                f"({target.stat().st_size / 1024**3:.1f} GB)",
                flush=True,
            )
            continue
        print(
            f"[ensure_models] downloading {spec['repo']}/{spec['filename']} → "
            f"{target_dir}",
            flush=True,
        )
        t0 = time.time()
        hf_hub_download(
            repo_id=spec["repo"],
            filename=spec["filename"],
            local_dir=str(target_dir),
            token=HF_TOKEN,
        )
        elapsed = time.time() - t0
        print(
            f"[ensure_models] done in {elapsed:.0f}s "
            f"({target.stat().st_size / 1024**3:.1f} GB)",
            flush=True,
        )


def _download_gemma(models_root: Path) -> None:
    from huggingface_hub import snapshot_download  # type: ignore

    gemma_dir = models_root / "text_encoders" / "gemma-3-12b"
    marker = gemma_dir / "preprocessor_config.json"
    if marker.exists():
        print(
            f"[ensure_models] ✓ cached: text_encoders/gemma-3-12b/", flush=True,
        )
        return
    gemma_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ensure_models] downloading {GEMMA_REPO} → {gemma_dir}", flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=GEMMA_REPO,
        local_dir=str(gemma_dir),
        token=HF_TOKEN,
    )
    print(f"[ensure_models] gemma done in {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    try:
        root = _redirect_models_to_volume()
        _download_required(root)
        _download_gemma(root)
        print("[ensure_models] all weights present", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        # Don't bail the worker boot — log loudly. ComfyUI will surface
        # a clear "checkpoint not found" error on the first /run if a
        # weight is missing, which is recoverable by re-running.
        print(
            f"[ensure_models] ⚠ failed (continuing anyway): "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
