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
# Public, pre-quantized bnb-4bit Gemma. ~7GB on disk, ~7GB VRAM at
# runtime via bitsandbytes auto-engaged from the model's config.json.
# Lightricks's own Q4 repo is gated on HF; unsloth's mirror is public.
GEMMA_REPO = os.environ.get(
    "GEMMA_REPO", "unsloth/gemma-3-12b-it-bnb-4bit"
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
    """Download Gemma encoder. Default is unsloth's public bnb-4bit
    snapshot (~7GB). Auto-quantizes via bitsandbytes when transformers
    loads the model — no monkey-patch required."""
    from huggingface_hub import snapshot_download  # type: ignore

    gemma_dir = models_root / "text_encoders" / "gemma-3-12b"
    marker = gemma_dir / "preprocessor_config.json"
    repo_marker = gemma_dir / f".repo-{GEMMA_REPO.replace('/', '_')}"

    if marker.exists() and repo_marker.exists():
        print(
            f"[ensure_models] ✓ cached: text_encoders/gemma-3-12b/ "
            f"(repo={GEMMA_REPO})",
            flush=True,
        )
        return

    # If a different repo's snapshot is cached, wipe it first — file
    # layouts differ and Lightricks's loader does a recursive glob,
    # which would pick up wrong files.
    if marker.exists() and not repo_marker.exists():
        print(
            f"[ensure_models] cached Gemma is from a different repo — "
            f"re-downloading from {GEMMA_REPO}",
            flush=True,
        )
        import shutil as _shutil
        _shutil.rmtree(gemma_dir, ignore_errors=True)

    gemma_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ensure_models] downloading {GEMMA_REPO} → {gemma_dir}", flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=GEMMA_REPO,
        local_dir=str(gemma_dir),
        token=HF_TOKEN,
    )
    repo_marker.touch()
    print(f"[ensure_models] gemma done in {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    """Ensure all weights are present. Exits non-zero if any required
    file is missing AFTER all downloads attempted, so the worker boot
    aborts loudly instead of silently starting ComfyUI with an empty
    text_encoders dir (the failure mode that bit us on 2026-05-08:
    gated repo download failed silently → workflow validation later
    rejected an empty gemma_path filename list)."""
    download_err: Exception | None = None
    try:
        root = _redirect_models_to_volume()
        _download_required(root)
        _download_gemma(root)
    except Exception as e:  # noqa: BLE001
        download_err = e
        print(
            f"[ensure_models] ⚠ download error: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

    # Verify every required file actually landed. If any are missing
    # we exit non-zero so start_with_models.sh fails fast — far
    # better than booting ComfyUI to handle a request it can't fulfil.
    missing: list[str] = []
    for spec in REQUIRED:
        target = root / spec["subdir"] / spec["filename"]
        if not _file_complete(target, spec["min_bytes"]):
            missing.append(f"{spec['subdir']}/{spec['filename']}")
    gemma_marker = root / "text_encoders" / "gemma-3-12b" / "preprocessor_config.json"
    if not gemma_marker.exists():
        missing.append("text_encoders/gemma-3-12b/preprocessor_config.json")

    if missing:
        print(
            f"[ensure_models] ✗ ABORT: required files missing after download: "
            f"{', '.join(missing)}",
            flush=True,
        )
        if download_err:
            print(
                f"[ensure_models] root cause: {type(download_err).__name__}: "
                f"{download_err}",
                flush=True,
            )
        return 1

    print("[ensure_models] ✓ all weights present", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
