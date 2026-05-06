# RunPod LTX-2.3 Worker — AdsXFlow Reference

Drop-in **RunPod Serverless** worker that AdsXFlow's
`ltx_video_provider` talks to. Renders **a single MP4 with
synchronized audio** (LTX-2.3 is the first DiT-based audio-video
foundation model — no separate VO step needed).

> ⚙️  **Serverless, not Pods.** Workers spin up on the first /run,
> idle out after 30s, and scale back to zero. **You only pay for
> seconds of inference** — there's no idle storage or always-on
> compute bill. RunPod handles the lifecycle.

## Cost shape

| GPU         | Flex $/s     | Time / 8s clip | Cost / 24s reel | vs Veo 3.1 Fast |
|-------------|--------------|----------------|-----------------|-----------------|
| L40S 48GB   | $0.00053     | ~30-40s        | **~$0.055**     | **~65× cheaper** |
| A100 80GB   | $0.00076     | ~20-25s        | ~$0.045         | ~80× cheaper    |
| H100 80GB   | $0.00116     | ~12-15s        | ~$0.045         | ~80× cheaper    |
| (Veo Fast)  | n/a          | n/a            | $3.60           | —               |

(Numbers assume LTX-2.3 distilled-1.1 at 8 inference steps. The
**dev** checkpoint at 40 steps is ~5× slower and proportionally
more expensive.)

## Deploy in 5 minutes

1. Push this directory to a Git repo (private works too if you
   connect RunPod's GitHub app).
2. RunPod console → **Serverless** → **New Endpoint** → **Deploy
   from GitHub** → point at the repo + this directory.
3. Worker config:
   - **GPU**: **L40S 48GB minimum** (LTX-2.3 is 22B in bf16 ≈ 44GB).
     A100 80GB recommended for headroom + faster renders.
   - **Min workers**: `0` (true scale-to-zero).
   - **Max workers**: `5` (covers ~190 reels/hr peak; tune to traffic).
   - **Idle timeout**: `30s`.
   - **Execution timeout**: `300s`.
   - **FlashBoot**: `ON`.
4. Build env vars (set in the endpoint's "Container env" panel):
   - `HF_TOKEN` — your HuggingFace access token. Required for the
     gated Gemma model. Without this the build will fail at the
     `snapshot_download` step.
5. Wait for the first build (~25-40 min — pulls ~50GB of weights into
   the image cache).
6. Copy the **Endpoint ID** from the dashboard.

## Wire it to AdsXFlow

```bash
# backend/.env
RUNPOD_API_KEY=your-account-key                      # console → Settings → API Keys
RUNPOD_LTX_ENDPOINT_ID=your-endpoint-id-from-step-6
```

Restart the backend. The bento model picker in Studio surfaces
**LTX 2.3** as a render target. The "OPEN-SOURCE" tile goes from
dimmed (Configure) → live the moment both env vars are set.

## What's different from LTX 1.x

LTX 1.x (LTX-Video 13B) was the previous generation. LTX-2.3 brings:

- **Native synchronized audio + lip-sync**. A talking-head reel
  comes back as a single MP4 with the dialog already voiced — no
  separate Chirp-TTS / mux step. (The provider's `supports_audio`
  + `supports_lip_sync` registry flags both flip to `true`.)
- **Up to 1080p portrait** (LTX 1.x topped out at 768×1280).
- **Better prompt adherence** — the gated attention text connector
  follows multi-clause prompts much more reliably.
- **Bigger model** — 22B vs 13B → needs L40S 48GB minimum.

## Test locally before deploying

```bash
pip install -r requirements.txt
# Also clone + install the LTX-2 repo packages:
git clone https://github.com/Lightricks/LTX-2.git /tmp/LTX-2
pip install -e /tmp/LTX-2/packages/ltx-core
pip install -e /tmp/LTX-2/packages/ltx-pipelines
# Set checkpoint paths to wherever you've staged the weights:
export LTX_CHECKPOINT_PATH=/path/to/ltx-2.3-22b-distilled-1.1.safetensors
export LTX_SPATIAL_UPSAMPLER_PATH=/path/to/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
export LTX_GEMMA_ROOT=/path/to/gemma
python handler.py     # uses test_input.json
```

## I/O contract

This worker implements the schema documented at
`backend/app/services/ltx_video_provider.py`. If you replace this
with a different LTX-2.3 worker (community templates from Civitai
etc.), make sure its `event["input"]` accepts the same keys and its
return value matches the documented output shape.

## Storage / billing notes

- **No persistent network volume.** Weights are baked into the
  Docker image at build time, so the runtime cost is purely Flex
  inference seconds. No idle storage bill.
- **Cancel-on-timeout** is enforced server-side by AdsXFlow's
  `runpod_client.wait_for_completion` — if our wall-clock budget
  elapses, we POST `/cancel/{job_id}` so the worker stops billing
  immediately rather than running to completion.
- **Cold starts** are FlashBoot-bounded to ~2-5s when the worker
  pool has been warm in the last hour. From fully cold, expect
  ~60-90s for the first render (model load + first inference). The
  pool stays warm for 30s after each job by default.
