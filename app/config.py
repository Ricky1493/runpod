"""
GPU worker configuration.

Read once at import from the environment, validated eagerly. A misconfigured
worker must refuse to start rather than serve wrong results: the whole point of
this service is numerical parity with India's CPU path, and every value here
participates in that.

GPU_WORKER_API_KEY HAS NO DEFAULT, on purpose: the /process endpoint is publicly
reachable through the RunPod proxy, so starting without auth is not a degraded
mode, it is an open door.

MODEL_NAME IS PINNED to buffalo_l and any other value is refused. See
PINNED_MODEL_NAME below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: The RunPod HTTP proxy (Cloudflare) closes the connection with a 524 after this
#: long. Every deadline in the worker is derived from it.
PROXY_TIMEOUT_CEILING_SECONDS = 100

#: The only embedding dimension India supports. The .bin format is a raw
#: float32[512] dump (2048 bytes) and faiss_updater rejects anything else.
EMBEDDING_DIM = 512

#: THE PINNED INSIGHTFACE MODEL. A constraint, not a default.
#:
#: buffalo_l (w600k_r50) and buffalo_s (w600k_mbf) are different 512-dim
#: embedding spaces. Both produce valid-looking unit vectors; only buffalo_l
#: matches India's existing FAISS index, and a mismatch fails SILENTLY.
#:
#: The worker REFUSES TO START on any other value. That is deliberate belt and
#: braces: the model is baked into the image at build time, so a runtime
#: MODEL_NAME that disagrees with the baked pack means the image was built wrong
#: — and a Pod that cannot prove which model it is running must not serve.
PINNED_MODEL_NAME = "buffalo_l"


class ConfigError(RuntimeError):
    """Configuration is missing or inconsistent."""


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be an integer, got {raw!r}")


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a number, got {raw!r}")


def _env_list(key: str) -> List[str]:
    raw = _env(key)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_pair(key: str, default: Tuple[int, int]) -> Tuple[int, int]:
    raw = _env(key)
    if raw is None:
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ConfigError(f"{key} must be 'W,H', got {raw!r}")
    return (int(parts[0]), int(parts[1]))


@dataclass(frozen=True)
class WorkerConfig:
    # ---- identity and auth ----
    api_key: str
    image_version: str = "unknown"
    image_digest: Optional[str] = None
    burst_cycle_id: Optional[str] = None

    # ---- model / parity (must match India exactly) ----
    model_name: str = PINNED_MODEL_NAME
    """PINNED to buffalo_l. validate() rejects anything else."""
    det_size: Tuple[int, int] = (640, 640)
    det_thresh: float = 0.6
    max_detection_size: int = 640
    """Longest-side target for the pre-detection resize.

    Detection runs on the RESIZED image and bboxes are scaled back to original
    coordinates. India uses MAX_IMAGE_SIZE=640; a different value here means
    different pixels reach the model, which means different embeddings.
    """
    onnx_provider: str = "CUDAExecutionProvider"
    embedding_dim: int = EMBEDDING_DIM
    ctx_id: int = 0
    """0 = first GPU. India's CPU path uses -1. This is the ONE intentional
    difference between the two."""

    # ---- limits ----
    port: int = 8000
    max_batch_images: int = 128
    max_request_bytes: int = 1_048_576
    max_image_bytes: int = 33_554_432
    max_image_pixels: int = 178_956_970
    """Decompression-bomb guard. Matches Pillow's default ceiling."""
    download_concurrency: int = 16
    download_timeout_seconds: int = 20
    request_deadline_seconds: int = 75
    result_cache_size: int = 64
    rate_limit_rps: float = 20.0
    allowed_url_hosts: List[str] = field(default_factory=list)
    """Host allowlist for image fetches. EMPTY MEANS REJECT EVERYTHING — the safe
    direction. Without this, /process is an SSRF primitive for anyone who gets
    past auth."""

    # ---- runtime ----
    log_level: str = "INFO"
    model_root: str = "/home/worker/.insightface"
    cv2_threads: int = 0
    """0 = let OpenCV decide. Decode is likely the real bottleneck, so this is
    worth tuning against the Pod's vCPU allocation during benchmarking."""

    def validate(self) -> None:
        problems: List[str] = []

        if not self.api_key:
            problems.append(
                "GPU_WORKER_API_KEY is not set. The /process endpoint is publicly "
                "reachable through the RunPod proxy; refusing to start without "
                "authentication."
            )
        elif len(self.api_key) < 16:
            problems.append("GPU_WORKER_API_KEY is implausibly short (<16 chars).")

        if self.model_name != PINNED_MODEL_NAME:
            problems.append(
                f"MODEL_NAME is {self.model_name!r} but this worker is PINNED to "
                f"{PINNED_MODEL_NAME!r}. buffalo_l (w600k_r50) and buffalo_s "
                f"(w600k_mbf) are different 512-dim embedding spaces; only "
                f"buffalo_l matches India's FAISS index, and a mismatch fails "
                f"silently. The model is baked into the image at build time, so "
                f"this disagreeing means the image was built with the wrong "
                f"--build-arg MODEL_NAME."
            )

        if self.embedding_dim != EMBEDDING_DIM:
            problems.append(
                f"embedding_dim must be {EMBEDDING_DIM}; got {self.embedding_dim}."
            )

        if not 0 < self.det_thresh < 1:
            problems.append("DET_THRESH must be between 0 and 1.")

        if self.max_detection_size < 1:
            problems.append("MAX_DETECTION_SIZE must be positive.")

        if self.request_deadline_seconds >= PROXY_TIMEOUT_CEILING_SECONDS:
            problems.append(
                f"REQUEST_DEADLINE_SECONDS ({self.request_deadline_seconds}) must "
                f"be below the RunPod proxy's {PROXY_TIMEOUT_CEILING_SECONDS}s "
                f"ceiling, or responses are lost to a Cloudflare 524 that the "
                f"client cannot interpret."
            )

        if self.download_timeout_seconds >= self.request_deadline_seconds:
            problems.append(
                "DOWNLOAD_TIMEOUT_SECONDS must be below REQUEST_DEADLINE_SECONDS."
            )

        if self.max_batch_images < 1:
            problems.append("MAX_BATCH_IMAGES must be >= 1.")

        if not self.allowed_url_hosts:
            problems.append(
                "ALLOWED_URL_HOSTS is empty, so every image URL will be rejected. "
                "Set it to the Backblaze endpoint host. (Failing closed is "
                "intentional: an unrestricted fetcher is an SSRF primitive.)"
            )

        if problems:
            raise ConfigError(
                "GPU worker configuration is invalid; refusing to start:\n  - "
                + "\n  - ".join(problems)
            )

    def redacted(self) -> Dict[str, object]:
        """Config safe to log at startup."""
        return {
            "image_version": self.image_version,
            "image_digest": self.image_digest,
            "burst_cycle_id": self.burst_cycle_id,
            "model_name": self.model_name,
            "det_size": list(self.det_size),
            "det_thresh": self.det_thresh,
            "max_detection_size": self.max_detection_size,
            "onnx_provider": self.onnx_provider,
            "embedding_dim": self.embedding_dim,
            "ctx_id": self.ctx_id,
            "port": self.port,
            "max_batch_images": self.max_batch_images,
            "max_request_bytes": self.max_request_bytes,
            "download_concurrency": self.download_concurrency,
            "request_deadline_seconds": self.request_deadline_seconds,
            "result_cache_size": self.result_cache_size,
            "rate_limit_rps": self.rate_limit_rps,
            "allowed_url_hosts": self.allowed_url_hosts,
            "api_key": (
                f"<set:{len(self.api_key)} chars>" if self.api_key else "<unset>"
            ),
        }


def load_config() -> WorkerConfig:
    return WorkerConfig(
        api_key=_env("GPU_WORKER_API_KEY", "") or "",
        image_version=_env("IMAGE_VERSION", "unknown"),
        image_digest=_env("IMAGE_DIGEST"),
        burst_cycle_id=_env("BURST_CYCLE_ID"),
        # Read so a stray override is DETECTED and reported, not silently ignored.
        model_name=_env("MODEL_NAME", PINNED_MODEL_NAME) or PINNED_MODEL_NAME,
        det_size=_env_pair("DET_SIZE", (640, 640)),
        det_thresh=_env_float("DET_THRESH", 0.6),
        max_detection_size=_env_int("MAX_DETECTION_SIZE", 640),
        onnx_provider=_env("ONNX_PROVIDER", "CUDAExecutionProvider"),
        embedding_dim=_env_int("EMBEDDING_DIM", EMBEDDING_DIM),
        ctx_id=_env_int("CTX_ID", 0),
        port=_env_int("GPU_WORKER_PORT", 8000),
        max_batch_images=_env_int("MAX_BATCH_IMAGES", 128),
        max_request_bytes=_env_int("MAX_REQUEST_BYTES", 1_048_576),
        max_image_bytes=_env_int("MAX_IMAGE_BYTES", 33_554_432),
        max_image_pixels=_env_int("MAX_IMAGE_PIXELS", 178_956_970),
        download_concurrency=_env_int("DOWNLOAD_CONCURRENCY", 16),
        download_timeout_seconds=_env_int("DOWNLOAD_TIMEOUT_SECONDS", 20),
        request_deadline_seconds=_env_int("REQUEST_DEADLINE_SECONDS", 75),
        result_cache_size=_env_int("RESULT_CACHE_SIZE", 64),
        rate_limit_rps=_env_float("WORKER_RATE_LIMIT_RPS", 20.0),
        allowed_url_hosts=_env_list("ALLOWED_URL_HOSTS"),
        log_level=_env("LOG_LEVEL", "INFO"),
        model_root=_env("INSIGHTFACE_HOME", "/home/worker/.insightface"),
        cv2_threads=_env_int("CV2_THREADS", 0),
    )


_config: Optional[WorkerConfig] = None


def get_config(reload: bool = False) -> WorkerConfig:
    global _config
    if _config is None or reload:
        _config = load_config()
    return _config
