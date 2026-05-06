"""One-shot RunPod Serverless endpoint creator for LTX-2.3.

Run this AFTER you have:
  1. A Docker image URL containing the LTX-2.3 worker
       (Dockerfile in this directory builds one — push it to Docker
        Hub / ghcr.io / RunPod Hub or connect GitHub).
  2. A RunPod API key
       (RunPod console → Settings → API Keys → Create).

What it does:
  - Creates a template (image config) via POST /v1/templates.
  - Creates an endpoint (worker pool config) via POST /v1/endpoints
    with sane defaults: L40S 48GB primary GPU, scale-to-zero, FlashBoot,
    30s idle, 5-minute execution ceiling.
  - Optionally writes RUNPOD_LTX_ENDPOINT_ID to backend/.env so the
    backend picks it up on the next restart.
  - Idempotent: re-running with the same name updates the existing
    template/endpoint instead of creating duplicates.

Usage:
  export RUNPOD_API_KEY=rp_...
  python deploy.py \\
      --image docker.io/yourname/runpod-ltx-2-3:latest \\
      --name adsxflow-ltx-2-3 \\
      [--gpu L40S | A100_80 | H100_80 | RTX5090] \\
      [--max-workers 5] \\
      [--write-env ../../.env]

Or via env vars:
  RUNPOD_API_KEY=...  RUNPOD_IMAGE=...  python deploy.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


_BASE = "https://rest.runpod.io/v1"

# Friendly aliases → canonical RunPod gpuTypeIds. The ordered list is
# what gets sent: RunPod tries each in order, falling back when the
# preferred GPU isn't available in any region.
_GPU_PRESETS: dict[str, list[str]] = {
    # Cheapest 48GB option that fits LTX-2.3 distilled (22B bf16).
    # Falls back to A100 80GB then H100 80GB if no L40S available.
    "L40S": [
        "NVIDIA L40S",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA H100 80GB HBM3",
    ],
    "A100_80": [
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA L40S",
    ],
    "H100_80": [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 PCIe",
        "NVIDIA A100 80GB PCIe",
    ],
    # NOT recommended for LTX-2.3 (32GB < 44GB the 22B bf16 model needs)
    # — included only as a last-resort path if you've quantized weights.
    "RTX5090": [
        "NVIDIA GeForce RTX 5090",
    ],
}


def _request(method: str, path: str, *, api_key: str, body: dict | None = None) -> dict:
    """Tiny urllib wrapper — keeps this script dependency-free so
    operators can run it without `pip install runpod` if they want."""
    url = f"{_BASE}{path}"
    data: bytes | None = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="ignore")[:600]
        raise SystemExit(
            f"\nRunPod API {method} {path} failed: {e.code} {e.reason}\n"
            f"Response body: {body_txt}\n"
        )
    except urllib.error.URLError as e:
        raise SystemExit(f"\nRunPod API transport error: {e}\n")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit(f"\nRunPod API returned non-JSON: {raw[:300]!r}\n")


def _list_templates(*, api_key: str) -> list[dict]:
    """RunPod's GET /v1/templates returns either a list directly or
    {'items': [...]} depending on tier — handle both."""
    res = _request("GET", "/templates", api_key=api_key)
    if isinstance(res, list):
        return res
    return list(res.get("items") or res.get("templates") or [])


def _list_endpoints(*, api_key: str) -> list[dict]:
    res = _request("GET", "/endpoints", api_key=api_key)
    if isinstance(res, list):
        return res
    return list(res.get("items") or res.get("endpoints") or [])


def _find_by_name(items: list[dict], name: str) -> dict | None:
    for it in items:
        if (it.get("name") or "").strip() == name.strip():
            return it
    return None


def _create_or_update_template(
    *, api_key: str, name: str, image: str, hf_token: str,
) -> str:
    """Returns the templateId. Creates a new one if none with this
    name exists; otherwise reuses (RunPod doesn't let you mutate an
    existing template's image, so for image swaps the operator should
    pass a new --name)."""
    existing = _find_by_name(_list_templates(api_key=api_key), name)
    if existing and existing.get("id"):
        cur_image = existing.get("imageName") or existing.get("image") or ""
        if cur_image != image:
            print(
                f"⚠️  Template '{name}' already exists with image "
                f"{cur_image!r} — RunPod templates are immutable. "
                f"Re-run with --name <new-name> to point at {image}.\n"
                f"Reusing existing templateId={existing['id']}.",
                file=sys.stderr,
            )
        else:
            print(f"✓  Template '{name}' already exists — reusing.")
        return existing["id"]

    body: dict[str, Any] = {
        "name": name,
        "imageName": image,
        "isServerless": True,
        "containerDiskInGb": 80,           # leaves headroom for the model cache
        "volumeInGb": 0,                   # no persistent volume — model is baked
        "volumeMountPath": "/runpod-volume",
    }
    if hf_token:
        body["env"] = {"HF_TOKEN": hf_token}
    res = _request("POST", "/templates", api_key=api_key, body=body)
    tid = res.get("id") or res.get("templateId") or ""
    if not tid:
        raise SystemExit(f"Template create returned no id: {res}")
    print(f"✓  Created template '{name}' → templateId={tid}")
    return tid


def _create_or_update_endpoint(
    *, api_key: str, name: str, template_id: str,
    gpu_type_ids: list[str], max_workers: int,
    execution_timeout_ms: int = 300_000,
    idle_timeout: int = 30,
) -> str:
    existing = _find_by_name(_list_endpoints(api_key=api_key), name)
    body: dict[str, Any] = {
        "name": name,
        "templateId": template_id,
        "gpuTypeIds": gpu_type_ids,
        "gpuCount": 1,
        "workersMin": 0,                   # true scale-to-zero
        "workersMax": max_workers,
        "idleTimeout": idle_timeout,       # seconds before scale-down
        "executionTimeoutMs": execution_timeout_ms,
        "flashboot": True,                 # sub-2s warm cold-start
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
    }
    if existing and existing.get("id"):
        eid = existing["id"]
        # POST /v1/endpoints/{id}/update updates an existing endpoint.
        _request("POST", f"/endpoints/{eid}/update", api_key=api_key, body=body)
        print(f"✓  Updated endpoint '{name}' → endpointId={eid}")
        return eid
    res = _request("POST", "/endpoints", api_key=api_key, body=body)
    eid = res.get("id") or res.get("endpointId") or ""
    if not eid:
        raise SystemExit(f"Endpoint create returned no id: {res}")
    print(f"✓  Created endpoint '{name}' → endpointId={eid}")
    return eid


def _write_env(env_path: str, key: str, value: str) -> None:
    """Append-or-replace a single key in a .env file. We don't pull
    in python-dotenv to keep this script dependency-free."""
    if not os.path.exists(env_path):
        with open(env_path, "w", encoding="utf-8") as fh:
            fh.write(f"{key}={value}\n")
        print(f"✓  Wrote {key} to {env_path}")
        return
    with open(env_path, encoding="utf-8") as fh:
        lines = fh.readlines()
    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        lines.append(f"{key}={value}\n")
    with open(env_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    print(f"✓  {'Updated' if found else 'Appended'} {key} in {env_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--image", default=os.environ.get("RUNPOD_IMAGE", ""),
        help="Docker image URL (e.g. docker.io/youruser/runpod-ltx-2-3:latest).",
    )
    ap.add_argument(
        "--name", default=os.environ.get("RUNPOD_NAME", "adsxflow-ltx-2-3"),
        help="Endpoint + template name (used as a stable handle).",
    )
    ap.add_argument(
        "--gpu", default="L40S", choices=sorted(_GPU_PRESETS.keys()),
        help="GPU preset. L40S = cheapest 48GB; A100_80/H100_80 = faster + headroom.",
    )
    ap.add_argument(
        "--max-workers", type=int, default=5,
        help="Cap on concurrent workers. Min is always 0 (scale-to-zero).",
    )
    ap.add_argument(
        "--idle-timeout", type=int, default=30,
        help="Seconds an idle worker stays warm before scale-down.",
    )
    ap.add_argument(
        "--exec-timeout-ms", type=int, default=300_000,
        help="Max ms for a single render. Default 5 minutes.",
    )
    ap.add_argument(
        "--write-env", default="",
        help="Path to .env to update with RUNPOD_LTX_ENDPOINT_ID + RUNPOD_API_KEY.",
    )
    ap.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace token for the gated Gemma pull at build time. Optional once the image is built.",
    )
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "RUNPOD_API_KEY env var is required. Generate one at "
            "https://www.runpod.io/console/user/settings → API Keys."
        )
    if not args.image:
        raise SystemExit(
            "--image is required (or set RUNPOD_IMAGE). Build the "
            "Dockerfile in this directory and push to Docker Hub / "
            "ghcr.io / RunPod Hub first."
        )

    gpu_type_ids = _GPU_PRESETS[args.gpu]
    print(f"\n→ Deploying RunPod Serverless endpoint")
    print(f"  name      = {args.name}")
    print(f"  image     = {args.image}")
    print(f"  gpu       = {args.gpu} (tries: {', '.join(gpu_type_ids)})")
    print(f"  max workers = {args.max_workers}  ·  idle = {args.idle_timeout}s")
    print()

    template_id = _create_or_update_template(
        api_key=api_key, name=args.name, image=args.image, hf_token=args.hf_token,
    )
    endpoint_id = _create_or_update_endpoint(
        api_key=api_key, name=args.name, template_id=template_id,
        gpu_type_ids=gpu_type_ids, max_workers=args.max_workers,
        execution_timeout_ms=args.exec_timeout_ms,
        idle_timeout=args.idle_timeout,
    )

    print()
    print("─────────────────────────────────────────────────────────────")
    print(f"  RUNPOD_LTX_ENDPOINT_ID = {endpoint_id}")
    print("─────────────────────────────────────────────────────────────")

    if args.write_env:
        _write_env(args.write_env, "RUNPOD_LTX_ENDPOINT_ID", endpoint_id)
        _write_env(args.write_env, "RUNPOD_API_KEY", api_key)
    else:
        print(
            f"\nAdd these to backend/.env:\n"
            f"  RUNPOD_API_KEY={api_key}\n"
            f"  RUNPOD_LTX_ENDPOINT_ID={endpoint_id}\n"
        )

    print(
        f"Smoke-test (after the first worker boots, ~30-60s):\n"
        f"  curl -s -X POST https://api.runpod.ai/v2/{endpoint_id}/health \\\n"
        f"       -H 'authorization: Bearer {api_key[:8]}...'\n"
    )


if __name__ == "__main__":
    main()
