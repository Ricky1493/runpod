"""
FastAPI application for the ephemeral GPU worker.

Endpoints (plan §31):
    GET  /v1/health    readiness + full provenance. Authenticated.
    POST /v1/process   batch inference. Authenticated, idempotent by batch_id.
    GET  /v1/status    operational counters. Authenticated.

WHAT THIS SERVICE IS NOT ALLOWED TO DO, and does not:
    * connect to MSSQL or PostgreSQL (the drivers are not even installed)
    * read or write FAISS
    * generate production face crops
    * write .bin embedding files
    * set isInsightFace
    * make any outbound request other than to the presigned URLs it was given

AUTHENTICATION IS ON EVERY ENDPOINT, INCLUDING /health. The service is publicly
reachable through the RunPod proxy — the Pod id is obscurity, not security — and
an unauthenticated health endpoint would leak the model identity and GPU details
to anyone who found it.

Bind address is 0.0.0.0, not localhost. Binding to 127.0.0.1 makes the container
unreachable through the proxy and presents as a permanent 502.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, Header, Request, Response, status
from fastapi.responses import JSONResponse

from . import logging_setup
from .config import ConfigError, WorkerConfig, get_config
from .health import HealthReporter
from .insightface_service import (
    InsightFaceService,
    ModelLoadError,
    ProviderMismatchError,
)
from .result_cache import BatchIdConflict, ResultCache
from .schemas import (
    ErrorResponse,
    HealthResponse,
    ProcessRequest,
    StatusResponse,
)
from .worker import BatchProcessor

logger = logging.getLogger("app.main")


class AppState:
    """Process-wide singletons, assembled during startup."""

    config: WorkerConfig
    service: InsightFaceService
    processor: BatchProcessor
    cache: ResultCache
    reporter: HealthReporter
    rate_limiter: "RateLimiter"
    startup_error: Optional[str] = None


state = AppState()


class RateLimiter:
    """Token bucket per process.

    Not a performance feature — a guard against a dispatch-loop bug or a flood
    hammering a Pod we are paying for by the second.
    """

    def __init__(self, rate_per_second: float, burst: int = 10):
        self.rate = max(0.1, rate_per_second)
        self.burst = max(1, burst)
        self._tokens = float(self.burst)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self.burst, self._tokens + (now - self._updated) * self.rate
            )
            self._updated = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and the model BEFORE readiness flips.

    A configuration or model problem is fatal on purpose: a worker that cannot
    prove which model it is running must not serve. It stays up long enough to
    report the failure through /health (503 with a reason), which is how India's
    provisioning loop learns to terminate it rather than waiting out the timeout.
    """
    try:
        config = get_config()
        logging_setup.configure(
            config.log_level,
            image_version=config.image_version,
            burst_cycle_id=config.burst_cycle_id,
        )
        config.validate()
    except ConfigError as exc:
        logging_setup.configure("INFO")
        logger.critical("Configuration is invalid, refusing to serve: %s", exc)
        state.startup_error = str(exc)
        state.config = get_config()
        state.service = InsightFaceService(state.config)
        state.cache = ResultCache(1)
        state.processor = BatchProcessor(state.config, state.service)
        state.reporter = HealthReporter(state.config, state.service, state.processor)
        state.rate_limiter = RateLimiter(1.0)
        yield
        return

    logger.info("Worker starting with config: %s", config.redacted())

    if config.cv2_threads > 0:
        try:
            import cv2

            cv2.setNumThreads(config.cv2_threads)
        except Exception as exc:
            logger.warning("Could not set OpenCV thread count: %s", exc)

    state.config = config
    state.service = InsightFaceService(config)
    state.cache = ResultCache(config.result_cache_size)
    state.processor = BatchProcessor(config, state.service)
    state.reporter = HealthReporter(config, state.service, state.processor)
    state.rate_limiter = RateLimiter(config.rate_limit_rps)

    try:
        # Synchronous and blocking on purpose: readiness must not flip until the
        # model is loaded, the provider is verified and a warm-up inference has
        # succeeded.
        state.service.load()
    except (ModelLoadError, ProviderMismatchError) as exc:
        logger.critical("Model load failed, worker will report unhealthy: %s", exc)
        state.startup_error = str(exc)
    except Exception as exc:
        logger.critical("Unexpected model load failure: %s", exc, exc_info=True)
        state.startup_error = str(exc)

    _install_sigterm_handler()

    yield

    logger.info(
        "Worker shutting down after %d batches", state.processor.stats.batches_processed
    )


def _install_sigterm_handler() -> None:
    """Drain on SIGTERM so a terminate arriving mid-batch is clean."""

    def handler(_signum, _frame):
        logger.info("SIGTERM received: entering drain, finishing in-flight work")
        state.reporter.begin_drain()

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):  # pragma: no cover - not main thread
        pass


app = FastAPI(
    title="Face GPU Worker",
    description=(
        "Stateless InsightFace inference for the ephemeral RunPod GPU burst tier. "
        "Presigned URL in, bounding boxes and 512-dim embeddings out. No database, "
        "no FAISS, no crops, no completion flags."
    ),
    version=os.environ.get("IMAGE_VERSION", "unknown"),
    lifespan=lifespan,
    # Docs off: this is a machine-to-machine endpoint on the public internet.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# =============================================================================
# Auth and errors
# =============================================================================


def _error(
    status_code: int,
    error_code: str,
    message: str,
    *,
    batch_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error_code=error_code,
            message=message,
            batch_id=batch_id,
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).model_dump(),
    )


async def require_auth(
    request: Request, authorization: Optional[str] = Header(None)
) -> None:
    """Bearer-token auth with a timing-safe comparison."""
    import hmac

    expected = state.config.api_key
    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    if not expected or not hmac.compare_digest(provided, expected):
        # Deliberately generic: never reveal whether the key was absent,
        # malformed or simply wrong. Source IP is logged for flood detection.
        client = request.client.host if request.client else "unknown"
        logger.warning(
            "Rejected unauthenticated request from %s to %s", client, request.url.path
        )
        raise _AuthError()


class _AuthError(Exception):
    pass


@app.exception_handler(_AuthError)
async def _auth_error_handler(_request: Request, _exc: _AuthError) -> JSONResponse:
    return _error(
        status.HTTP_401_UNAUTHORIZED, "unauthorized", "authentication required"
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return _error(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal",
        "an internal error occurred",
        request_id=request_id,
    )


@app.middleware("http")
async def correlate_and_limit(request: Request, call_next):
    """Attach a request id, enforce the body-size cap and the rate limit."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id
    logging_setup.set_context(
        request_id=request_id, burst_cycle_id=request.headers.get("X-Burst-Cycle-Id")
    )

    try:
        # 413 before reading the body: enforcing the cap from Content-Length
        # avoids buffering a hostile payload just to reject it.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit():
            if int(declared) > state.config.max_request_bytes:
                return _error(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    f"request body exceeds {state.config.max_request_bytes} bytes",
                    request_id=request_id,
                )

        if request.url.path.startswith("/v1/process"):
            if not await state.rate_limiter.allow():
                return _error(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "rate_limited",
                    "too many requests",
                    request_id=request_id,
                )

        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        logging_setup.clear_context()


# =============================================================================
# Endpoints
# =============================================================================


@app.get(
    "/v1/health", response_model=HealthResponse, dependencies=[Depends(require_auth)]
)
async def health(response: Response) -> HealthResponse:
    """Readiness and provenance.

    Returns 503 with the SAME body until ready, so India can poll one endpoint
    from container start through model load to ready.
    """
    payload = state.reporter.health()
    if not state.reporter.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return payload


@app.get(
    "/v1/status", response_model=StatusResponse, dependencies=[Depends(require_auth)]
)
async def worker_status() -> StatusResponse:
    return state.reporter.status_payload(cache_entries=len(state.cache))


@app.post("/v1/process", dependencies=[Depends(require_auth)])
async def process(request: ProcessRequest, http_request: Request):
    """Batch inference.

    Idempotent by ``batch_id``: an identical repeat returns the cached response
    with ``cached: true``, no re-download and no re-inference. That is what makes
    India's timeout retry free.

    Per-image failures are reported inside a 200 response; only whole-request
    problems produce a non-200.
    """
    request_id = getattr(http_request.state, "request_id", None)
    batch_id = str(request.batch_id)
    logging_setup.set_context(batch_id=batch_id)

    if state.startup_error is not None:
        return _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "not_ready",
            f"worker failed to start: {state.startup_error}",
            batch_id=batch_id,
            request_id=request_id,
        )

    if not state.reporter.is_ready:
        return _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "not_ready",
            f"worker status is {state.reporter.status!r}",
            batch_id=batch_id,
            request_id=request_id,
        )

    if len(request.images) > state.config.max_batch_images:
        return _error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "batch_too_large",
            f"images must contain between 1 and "
            f"{state.config.max_batch_images} entries",
            batch_id=batch_id,
            request_id=request_id,
        )

    signature = request.content_signature()
    try:
        cached = state.cache.get(batch_id, signature)
    except BatchIdConflict as exc:
        return _error(
            status.HTTP_409_CONFLICT,
            "batch_id_conflict",
            str(exc),
            batch_id=batch_id,
            request_id=request_id,
        )
    if cached is not None:
        return JSONResponse(status_code=200, content=cached)

    result = await state.processor.process(request)
    payload = result.model_dump()
    state.cache.put(batch_id, signature, payload)
    return JSONResponse(status_code=200, content=payload)


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Unauthenticated, and deliberately uninformative.

    Enough for a human to know what they found; nothing about the model, the GPU
    or the cycle.
    """
    return JSONResponse(
        {"service": "face-gpu-worker", "endpoints": ["/v1/health", "/v1/process"]}
    )
