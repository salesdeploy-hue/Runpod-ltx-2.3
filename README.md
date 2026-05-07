# AdsXFlow LTX-2.3 ComfyUI Worker

RunPod Serverless worker for **Lightricks LTX-2.3** built on the
**ComfyUI** runtime — the same stack the LTX community uses to fit
22B params on 32 GB GPUs.

This replaces the previous native `ltx_pipelines` handler approach,
which kept the LTX 22B + Gemma 12B both pinned in VRAM and OOMed
even on H200 (140 GB).

## What's inside

| Layer | Provider |
|---|---|
| Base image | `runpod/worker-comfyui:5.8.5-base-cuda12.8.1` (RunPod's official ComfyUI Serverless worker) |
| LTX nodes | `Lightricks/ComfyUI-LTXVideo` (auto-installed at build) |
| Models | `Lightricks/LTX-2.3-fp8` distilled + `Lightricks/gemma-3-12b-it-qat-q4_0-unquantized` text encoder + spatial upscaler — all baked in |
| Workflow | Driven entirely by JSON sent in the `/run` body |

## API contract

**Input** (POST `/run`):

```json
{
  "input": {
    "workflow": { ...ComfyUI graph JSON... },
    "images": [{ "name": "first_frame.png", "image": "<base64>" }]
  }
}
```

The handler is the stock RunPod ComfyUI handler — see
[`runpod-workers/worker-comfyui`](https://github.com/runpod-workers/worker-comfyui).

**Output** (`/status/{id}` once `COMPLETED`): base64 video bytes from
the workflow's `CreateVideo` (or equivalent) output node.

## Workflow templates

`workflows/ltx-2.3-t2v-i2v-distilled.json` — the official Lightricks
LTX-2.3 single-stage distilled workflow (44 nodes). AdsXFlow's
`app/services/ltx_video_provider.py` parameterises this template at
runtime (prompt, seed, duration, image_url, etc.) and submits it.

## Deploying

1. Push commits to `main`.
2. RunPod Console → Serverless → New Endpoint → Deploy from GitHub →
   pick `salesdeploy-hue/Runpod-ltx-2.3` (this repo) → branch `main`.
3. Hub will read `.runpod/hub.json` for defaults (60 GB container
   disk, 32 GB GPU floor, 1 max worker, 30s idle, scale-to-zero).
4. Optional build env: `HF_TOKEN` if you swap the Gemma repo to
   Google's gated `gemma-3-4b-pt`. Otherwise unused.

## Backup branch

The previous native-pipeline handler is preserved on
`backup/ltx-2.3-native-pipeline` if anyone wants to revisit that
path with fp8 quantization or a 80 GB+ GPU floor.

## Cost shape

| GPU | Flex $/s | First render (cold + DL) | Warm render (8s clip) |
|---|---|---|---|
| RTX 5090 32GB | $0.00040 | ~3 min | ~$0.020 |
| L40S 48GB | $0.00053 | ~3 min | ~$0.022 |
| A100 80GB | $0.00076 | ~3 min | ~$0.018 |

Image is ~55 GB (LTX fp8 + Gemma + ComfyUI) — first build is
~25-40 min (one-time per commit).
