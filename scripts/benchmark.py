#!/usr/bin/env python3
"""
GPU benchmark harness (plan §25.1, Phase 7).

Run this AGAINST A REAL RTX 4090 POD, from a machine with network access to it.

WHAT THIS ANSWERS, and why each matters:

  * The batch size at which images/sec plateaus.
  * The batch size at which p99 batch duration approaches the worker's deadline.
    THIS IS THE HARD CAP, regardless of throughput: past it, requests die to the
    RunPod proxy's 100-second Cloudflare ceiling as an opaque 524.
  * Whether the bottleneck is fetch, decode, or inference. The plan's working
    hypothesis (§25.1) is that at a 640px detection input, buffalo inference on a
    4090 is fast and the binding constraint is image download plus full-resolution
    JPEG decode on the Pod's CPU allocation. If that holds, the lever is
    DOWNLOAD_CONCURRENCY and decode threads, not batch size — and a larger vCPU
    flavour may matter more than a faster GPU.

WHAT THIS DOES NOT ANSWER: end-to-end throughput. India's finalization (crops,
.bin files, DB rows) must be measured separately (plan §25.2), and the sustainable
rate is min(gpu_rate, finalizer_rate). Reporting the GPU number alone would be
misleading.

USAGE
    python benchmark.py \
        --endpoint https://<pod-id>-8000.proxy.runpod.net \
        --api-key <per-cycle key> \
        --urls urls.txt \
        --batch-sizes 8,16,32,64,128 \
        --repetitions 3 \
        --out ../benchmarks/gpu_1.0.0.json

    urls.txt: one presigned URL per line, from production objects. Synthetic
    images would not exercise realistic decode cost or face counts.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx


def load_urls(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        urls = [line.strip() for line in handle if line.strip()]
    if not urls:
        raise SystemExit(f"{path} contains no URLs")
    return urls


def fetch_health(client: httpx.Client, endpoint: str) -> Dict[str, Any]:
    response = client.get(f"{endpoint}/v1/health")
    response.raise_for_status()
    return response.json()


def fetch_status(client: httpx.Client, endpoint: str) -> Dict[str, Any]:
    try:
        response = client.get(f"{endpoint}/v1/status")
        return response.json() if response.status_code == 200 else {}
    except Exception:
        return {}


def run_batch(
    client: httpx.Client, endpoint: str, urls: List[str], start_id: int
) -> Dict[str, Any]:
    """Send one batch and record what happened."""
    batch_id = str(uuid.uuid4())
    payload = {
        "batch_id": batch_id,
        "images": [
            {"picture_id": start_id + index, "image_url": url}
            for index, url in enumerate(urls)
        ],
    }

    started = time.perf_counter()
    try:
        response = client.post(f"{endpoint}/v1/process", json=payload)
    except httpx.TimeoutException:
        return {
            "ok": False,
            "error": "client timeout",
            "wall_ms": int((time.perf_counter() - started) * 1000),
            "images": len(urls),
        }
    wall_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code != 200:
        return {
            "ok": False,
            "error": f"HTTP {response.status_code}: {response.text[:200]}",
            "http_status": response.status_code,
            "wall_ms": wall_ms,
            "images": len(urls),
            # A 524 is the proxy ceiling, and it is the number that caps batch
            # size no matter what throughput says.
            "proxy_timeout": response.status_code == 524,
        }

    body = response.json()
    timings = body.get("timings", {})
    return {
        "ok": True,
        "wall_ms": wall_ms,
        "server_ms": body.get("duration_ms"),
        "images": body.get("processed_count", len(urls)),
        "success": body.get("success_count", 0),
        "failed": body.get("failed_count", 0),
        "faces": body.get("total_faces", 0),
        "fetch_ms": timings.get("fetch_ms"),
        "decode_ms": timings.get("decode_ms"),
        "inference_ms": timings.get("inference_ms"),
        "cached": body.get("cached", False),
    }


def summarize(runs: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
    ok_runs = [run for run in runs if run.get("ok")]
    if not ok_runs:
        return {
            "batch_size": batch_size,
            "batches_attempted": len(runs),
            "batches_ok": 0,
            "errors": [run.get("error") for run in runs][:5],
            "proxy_timeouts": sum(1 for r in runs if r.get("proxy_timeout")),
        }

    walls = sorted(run["wall_ms"] for run in ok_runs)
    total_images = sum(run["images"] for run in ok_runs)
    total_faces = sum(run["faces"] for run in ok_runs)
    total_seconds = sum(walls) / 1000.0

    def percentile(values: List[int], fraction: float) -> int:
        if not values:
            return 0
        index = min(len(values) - 1, int(round(fraction * (len(values) - 1))))
        return values[index]

    def mean_of(key: str) -> Optional[float]:
        present = [run[key] for run in ok_runs if run.get(key) is not None]
        return round(statistics.mean(present), 1) if present else None

    return {
        "batch_size": batch_size,
        "batches_attempted": len(runs),
        "batches_ok": len(ok_runs),
        "proxy_timeouts": sum(1 for r in runs if r.get("proxy_timeout")),
        "images_total": total_images,
        "faces_total": total_faces,
        "images_per_second": (
            round(total_images / total_seconds, 2) if total_seconds else 0
        ),
        "faces_per_second": (
            round(total_faces / total_seconds, 2) if total_seconds else 0
        ),
        "faces_per_image": round(total_faces / total_images, 2) if total_images else 0,
        "wall_ms_p50": percentile(walls, 0.50),
        "wall_ms_p95": percentile(walls, 0.95),
        "wall_ms_p99": percentile(walls, 0.99),
        "wall_ms_max": walls[-1],
        "mean_fetch_ms": mean_of("fetch_ms"),
        "mean_decode_ms": mean_of("decode_ms"),
        "mean_inference_ms": mean_of("inference_ms"),
        "images_failed": sum(run["failed"] for run in ok_runs),
    }


def identify_bottleneck(summary: Dict[str, Any]) -> str:
    """Name the dominant stage, from the server-side timing breakdown."""
    stages = {
        "fetch": summary.get("mean_fetch_ms") or 0,
        "decode": summary.get("mean_decode_ms") or 0,
        "inference": summary.get("mean_inference_ms") or 0,
    }
    total = sum(stages.values())
    if total <= 0:
        return "unknown (no server timings)"
    dominant = max(stages, key=stages.get)
    share = stages[dominant] / total * 100
    return f"{dominant} ({share:.0f}% of measured time)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the GPU worker")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key", default=os.environ.get("GPU_WORKER_API_KEY"))
    parser.add_argument("--urls", required=True, help="File of presigned URLs")
    parser.add_argument("--batch-sizes", default="8,16,32,64,128")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", help="Write the JSON report here")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("--api-key or GPU_WORKER_API_KEY is required")

    endpoint = args.endpoint.rstrip("/")
    urls = load_urls(args.urls)
    batch_sizes = [int(v) for v in args.batch_sizes.split(",") if v.strip()]

    client = httpx.Client(
        timeout=httpx.Timeout(args.timeout, connect=15),
        headers={"Authorization": f"Bearer {args.api_key}"},
    )

    print(f"Benchmarking {endpoint}")
    health = fetch_health(client, endpoint)
    print(
        f"  gpu={health.get('gpu_name')} provider={health.get('provider')} "
        f"cuda={health.get('cuda_version')} model={health.get('model_name')} "
        f"image={health.get('image_version')}"
    )

    # Guard the whole run: a benchmark on the CPU provider would produce numbers
    # that look plausible and mean nothing.
    if health.get("provider") != "CUDAExecutionProvider":
        raise SystemExit(
            f"REFUSING TO BENCHMARK: provider is {health.get('provider')!r}, not "
            f"CUDAExecutionProvider. onnxruntime-gpu falls back to CPU on a "
            f"CUDA/cuDNN mismatch; these numbers would be meaningless."
        )
    if health.get("status") != "ready":
        raise SystemExit(f"worker is not ready (status={health.get('status')})")

    report: Dict[str, Any] = {
        "endpoint": endpoint,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "health": health,
        "url_pool_size": len(urls),
        "repetitions": args.repetitions,
        "results": [],
    }

    next_id = 1
    for batch_size in batch_sizes:
        if batch_size > len(urls):
            print(f"  skipping batch size {batch_size}: only {len(urls)} URLs")
            continue

        print(f"\n  batch size {batch_size} x{args.repetitions}...", flush=True)
        runs: List[Dict[str, Any]] = []
        for repetition in range(args.repetitions):
            # A fresh window of URLs each time, so the worker's result cache and
            # any origin-side caching cannot flatter the numbers.
            offset = (repetition * batch_size) % max(1, len(urls) - batch_size + 1)
            window = urls[offset : offset + batch_size]
            if len(window) < batch_size:
                window = (urls * 2)[offset : offset + batch_size]

            run = run_batch(client, endpoint, window, next_id)
            next_id += batch_size
            runs.append(run)

            if run.get("ok"):
                print(
                    f"    rep {repetition + 1}: {run['wall_ms']}ms wall, "
                    f"{run['server_ms']}ms server, {run['faces']} faces "
                    f"(fetch {run['fetch_ms']}ms, decode {run['decode_ms']}ms, "
                    f"inference {run['inference_ms']}ms)"
                )
            else:
                print(f"    rep {repetition + 1}: FAILED — {run.get('error')}")

        summary = summarize(runs, batch_size)
        summary["bottleneck"] = identify_bottleneck(summary)
        summary["status_after"] = fetch_status(client, endpoint)
        report["results"].append(summary)

        print(
            f"    -> {summary.get('images_per_second', 0)} img/s, "
            f"p99 {summary.get('wall_ms_p99', 0)}ms, "
            f"bottleneck: {summary['bottleneck']}"
        )

    _print_recommendation(report, health)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nWrote {args.out}")

    client.close()
    return 0


def _print_recommendation(report: Dict[str, Any], health: Dict[str, Any]) -> None:
    """Turn the numbers into the one decision Phase 7 has to make."""
    usable = [
        r
        for r in report["results"]
        if r.get("batches_ok") and not r.get("proxy_timeouts")
    ]
    print("\n" + "=" * 72)
    print("RECOMMENDATION")
    print("=" * 72)

    if not usable:
        print("  No batch size completed cleanly. Investigate before proceeding.")
        return

    # The deadline is the constraint, so prefer the fastest size whose p99 leaves
    # comfortable headroom rather than the fastest size overall.
    deadline_ms = 75_000
    safe_ceiling = deadline_ms * 0.8
    safe = [r for r in usable if r["wall_ms_p99"] <= safe_ceiling]
    pool = safe or usable
    best = max(pool, key=lambda r: r["images_per_second"])

    print(f"  GPU_BATCH_SIZE = {best['batch_size']}")
    print(f"    {best['images_per_second']} img/s, {best['faces_per_second']} faces/s")
    print(
        f"    p50 {best['wall_ms_p50']}ms / p99 {best['wall_ms_p99']}ms "
        f"(worker deadline {deadline_ms}ms)"
    )
    print(f"    bottleneck: {best['bottleneck']}")

    if not safe:
        print(
            "\n  WARNING: no batch size kept p99 within 80% of the worker "
            "deadline. Reduce batch size further or raise DOWNLOAD_CONCURRENCY; "
            "a p99 near the deadline means real batches will return partial "
            "results under load."
        )

    if "fetch" in best["bottleneck"]:
        print(
            "\n  Fetch dominates, which matches the plan's hypothesis (§25.1). "
            "The lever is DOWNLOAD_CONCURRENCY and the Pod's network, NOT a "
            "bigger batch or a faster GPU."
        )
    elif "decode" in best["bottleneck"]:
        print(
            "\n  Decode dominates. The lever is the Pod's vCPU allocation and "
            "CV2_THREADS, NOT the GPU."
        )
    else:
        print(
            "\n  Inference dominates, so the GPU is genuinely the constraint — "
            "the one case where batch size and GPU choice are the right levers."
        )

    print(
        "\n  REMEMBER: this is the GPU ceiling only. Sustainable end-to-end "
        "throughput is min(this, India finalization rate). Measure India "
        "separately (plan §25.2) before quoting a system number."
    )


if __name__ == "__main__":
    sys.exit(main())
