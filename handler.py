"""Diagnostic hello-world handler — verify RunPod Serverless plumbing
works end-to-end before re-layering LTX-2.3.

If workers spawn, reach `ready`, and respond to /run with this handler:
  → the container/image/GPU/network plumbing is fine; the LTX code was
    the crash source. Re-deploy the real handler.

If workers STILL fail to come up with this trivial handler:
  → the issue is upstream of our app code (Dockerfile, image, GPU
    permissions, RunPod account state, etc.) and we need a different
    angle.
"""
from __future__ import annotations

import platform
import sys
import time
from typing import Any

import runpod  # type: ignore


_BOOT_TIME = time.time()
print(f"[diagnostic] handler.py imported. python={sys.version.split()[0]} platform={platform.platform()}", flush=True)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    inp = event.get("input") or {}
    print(f"[diagnostic] handler called with input keys: {list(inp.keys())}", flush=True)
    cuda_info: dict[str, Any] = {"checked": False}
    try:
        import torch  # type: ignore
        cuda_info = {
            "checked": True,
            "torch_version": getattr(torch, "__version__", "?"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None
            ),
        }
    except Exception as e:
        cuda_info = {"checked": True, "error": f"{type(e).__name__}: {e}"}
    return {
        "ok": True,
        "diagnostic": True,
        "echo_prompt": inp.get("prompt", ""),
        "uptime_s": round(time.time() - _BOOT_TIME, 1),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda": cuda_info,
    }


runpod.serverless.start({"handler": handler})
