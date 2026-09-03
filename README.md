# face-gpu-worker

Stateless InsightFace inference for the ephemeral RunPod GPU burst tier.

**Contract:** a presigned URL goes in, bounding boxes and 512-dim L2-normalized embeddings come out. Nothing else.

Part of the system described in `runpod_burst_implementation_plan.md` (India repository). This service is Tier 2; India is Tier 1 (control) and Tier 3 (finalization).

---

## What this service must never do

These are invariants, not guidelines. Each is enforced structurally rather than by policy:

| Prohibition | How it is enforced |
|---|---|
| Connect to MSSQL or PostgreSQL | The drivers are not installed. CI fails the build if `requirements.txt` declares one. |
| Read or write a FAISS index | `faiss` is not installed. |
| Generate production face crops | India downloads the full-resolution image and does this, because crops need the original pixels. |
| Write `.bin` embedding files | Embeddings are returned over the wire; India writes them. |
| Set `isInsightFace` | It has no database connection to set it with. |
| Fetch anything except the URLs it was given | Host allowlist, HTTPS-only, no redirects, private-address guard. An empty allowlist rejects everything. |

Separate repository from the India codebase on purpose: it cannot `import config.env_loader` or `database.connection` even by accident.

---

## The thing that makes this correct rather than merely fast

**Numerical parity with India's CPU path.** An embedding produced from a differently-resized image, or by a different model, is a valid-looking unit vector that silently fails to match an 11.3M-face FAISS index. Nothing downstream notices.

Two files are deliberate mirrors of India's code:

| This repo | Mirrors | Guarded by |
|---|---|---|
| `app/preprocess.py` | `core/image_processor.py` | `tests/test_preprocess.py` — interpolation mode, `<=` boundary, `int()` truncation |
| `app/quality.py` | `core/face_detector.py::_calculate_quality_score` | `tests/test_quality.py` — weights and 4dp rounding |

Three specific traps, all pinned by tests because each is easy to "clean up" into a silent bug:

- **`cv2.INTER_AREA`, not `INTER_LINEAR`.** Visually similar, numerically different.
- **`face.embedding` + manual L2 norm, not `face.normed_embedding`.** Nearly identical, which is the danger.
- **bboxes in ORIGINAL image coordinates**, divided by the detection scale. India crops from the full-resolution image.

`quality_score` matters more than it looks: `faiss_updater` picks which faces enter the searchable index with `ORDER BY quality_score DESC`, capped at 5 per person. A drifted score changes which faces are findable without changing any embedding.

---

## ⚠️ Before the first build: resolve `MODEL_NAME`

`model_manifest.json` ships **empty**, and the build warns loudly until it is populated.

The India repository disagrees with itself:

| Source | Says |
|---|---|
| `config/env_loader.py:79` | `buffalo_l` |
| `.env.example` | `buffalo_s` |
| `NEXT_STEPS.md` | the 2.1M-image bulk run used `buffalo_s` |

`buffalo_l` (w600k_r50) and `buffalo_s` (w600k_mbf) are **different 512-dim embedding spaces**. Both look valid. Only one matches the production index.

Read the live `.env` on the India server, hash its `.onnx` files, and populate `model_manifest.json` before building anything intended for production. See plan Appendix A / Q1.

---

## API

Base URL: `https://{pod-id}-8000.proxy.runpod.net`, discovered per Pod, never configured.

All endpoints require `Authorization: Bearer <GPU_WORKER_API_KEY>` — including `/v1/health`. The endpoint is publicly reachable; the Pod id is obscurity, not security.

| Endpoint | Purpose |
|---|---|
| `GET /v1/health` | Readiness and full provenance. India's gate asserts every field. Returns 503 with the same body until ready, so one URL covers the whole boot. |
| `POST /v1/process` | Batch inference. Idempotent by `batch_id`. |
| `GET /v1/status` | Operational counters. |

### The timeout ladder

```
worker deadline (75s)  <  India's client timeout (90s)  <  proxy ceiling (100s)
```

The RunPod HTTP proxy is Cloudflare-fronted and returns an opaque 524 past 100 seconds. Sitting below it means India always sees its own timeout, which it knows how to handle. On deadline the worker returns whatever finished and marks the rest `error_code: "deadline"` — a partial batch India can act on beats a 524 it cannot.

### Idempotency

A repeat `POST` with the same `batch_id` returns the cached response with `cached: true` — no re-download, no re-inference. This turns the most common failure (a completed batch whose response was lost on the way back) from *pay twice* into *free*.

A `batch_id` reused with a **different** set of pictures returns `409`. That cannot happen in correct operation, and serving the cached answer would mark the wrong pictures complete.

### Error model

Per-image failures live inside a `200` response as `success: false` with an `error_code`. One bad URL never costs the other 31 pictures their inference. Only whole-request problems produce a non-200.

`error_code` values: `download_failed`, `download_timeout`, `url_rejected`, `too_large`, `decode_failed`, `inference_failed`, `oom`, `deadline`, `internal`.

---

## Build

```bash
docker build \
  --build-arg IMAGE_VERSION=1.0.0 \
  --build-arg GIT_SHA=$(git rev-parse HEAD) \
  --build-arg MODEL_NAME=buffalo_l \
  -t ghcr.io/<org>/face-gpu-worker:1.0.0 .
```

Models are **baked in at build time**. On a disposable Pod, a runtime download happens on every burst — minutes added to `GPU_READY_TIME`, with a third-party CDN on the provisioning critical path. Baking also makes the model immutable per image tag, which is exactly the guarantee FAISS compatibility needs, and lets `/health` publish a SHA-256 India can assert.

Consequence: **no Network Volume is needed.**

Deploy **by digest**, not by tag. A tag is policy; a digest is a guarantee that the model which produced an embedding is identifiable.

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `GPU_WORKER_API_KEY` | — | **Required.** Generated per cycle by India; dies with the Pod. |
| `MODEL_NAME` | — | **Required.** Must match India exactly. |
| `ALLOWED_URL_HOSTS` | — | **Required.** Empty rejects every URL. |
| `DET_SIZE` | `640,640` | Parity-critical |
| `DET_THRESH` | `0.6` | Parity-critical |
| `MAX_DETECTION_SIZE` | `640` | Parity-critical |
| `ONNX_PROVIDER` | `CUDAExecutionProvider` | fp32 only in Phase 1 |
| `CTX_ID` | `0` | First GPU. India's CPU path uses `-1`; the one intentional difference. |
| `REQUEST_DEADLINE_SECONDS` | `75` | Must stay below 100 |
| `MAX_BATCH_IMAGES` | `128` | |
| `MAX_REQUEST_BYTES` | `1048576` | |
| `DOWNLOAD_CONCURRENCY` | `16` | Likely the real throughput lever |
| `RESULT_CACHE_SIZE` | `64` | Batches retained for idempotent replay |

The worker receives exactly **one** secret. No database credentials, no Backblaze keys, no storage encryption key.

---

## Tests

```bash
pip install pytest pytest-asyncio respx numpy opencv-python-headless pydantic httpx fastapi
pytest tests -v
```

104 tests, no GPU required. They cover the parity mirrors, the SSRF guard, the idempotency cache, schema validation and config refusals.

**What CI cannot prove:** that `CUDAExecutionProvider` is actually used, real inference numerics, §24 parity, or §25 throughput. Hosted runners have no GPU. Those are manual gates on a real RTX 4090 Pod — plan Phases 6, 7 and 7b — and must be re-run for every image version.

---

## Benchmark

```bash
python scripts/benchmark.py \
  --endpoint https://<pod-id>-8000.proxy.runpod.net \
  --api-key <key> --urls urls.txt \
  --batch-sizes 8,16,32,64,128 --repetitions 3 \
  --out ../benchmarks/gpu_1.0.0.json
```

Refuses to run on the CPU provider — numbers from a silent fallback would look plausible and mean nothing.

It reports a recommended `GPU_BATCH_SIZE` chosen so p99 stays within 80% of the worker deadline, and names the dominant stage. The plan's hypothesis is that **fetch and decode dominate, not inference** — in which case the levers are `DOWNLOAD_CONCURRENCY` and the Pod's vCPU allocation, not a bigger batch.

This measures the GPU ceiling only. Sustainable end-to-end throughput is `min(gpu_rate, India finalization rate)`, and India must be measured separately.

---

## Local development

```bash
export GPU_WORKER_API_KEY=$(openssl rand -hex 32)
export ALLOWED_URL_HOSTS=s3.us-west-004.backblazeb2.com
docker compose up --build            # needs the NVIDIA container toolkit
docker compose --profile cpu-parity up worker-cpu   # API surface only, no GPU
```

The CPU profile exercises auth, validation, limits and the cache. It is **not** valid for parity or benchmark work: the provider differs, so the numerics are not production's.

---

## Layout

```
app/
  main.py                 FastAPI app, auth, rate limit, error envelope
  config.py               env config + fail-fast validation
  schemas.py              request/response contract
  preprocess.py           decode + resize          ← parity mirror
  quality.py              quality score            ← parity mirror
  insightface_service.py  model load, provider verification, inference
  fetcher.py              SSRF-hardened presigned-URL fetching
  worker.py               batch orchestration, deadline, failure isolation
  result_cache.py         batch_id idempotency
  health.py               readiness + provenance reporting
  logging_setup.py        structured JSON logs, credential redaction
scripts/
  download_models.py      build-time model fetch + hash verification
  benchmark.py            §25.1 harness
tests/                    104 tests, no GPU required
Dockerfile                multi-stage, models baked, non-root
model_manifest.json       the parity contract (ships empty — see above)
```
