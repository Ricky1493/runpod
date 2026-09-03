"""
Readiness reporting.

``/v1/health`` is the gate India uses to decide whether a Pod may receive
production work. It is not a liveness ping — every field is asserted, and a single
mismatch means the Pod is terminated without processing anything.

That is the right severity. A Pod running the wrong model produces embeddings that
are numerically plausible and silently incompatible with the existing 11.3M-face
FAISS index; nothing downstream would notice. So this endpoint's job is to make
the model's identity, the execution provider actually in use, and every
preprocessing parameter externally verifiable.

The endpoint returns 503 (with the same body) until ready, so the caller can watch
image pull and model load progress through one URL rather than guessing.
"""

from __future__ import annotations

import time
from typing import Optional

from .config import WorkerConfig
from .insightface_service import InsightFaceService
from .schemas import HealthResponse, StatusResponse
from .worker import BatchProcessor


class HealthReporter:
    """Builds the /v1/health and /v1/status payloads."""

    def __init__(
        self,
        config: WorkerConfig,
        service: InsightFaceService,
        processor: Optional[BatchProcessor] = None,
    ):
        self.config = config
        self.service = service
        self.processor = processor
        self.started_at = time.time()
        self._draining = False

    def begin_drain(self) -> None:
        """Report 'draining' so a terminate arriving mid-batch is graceful."""
        self._draining = True

    @property
    def uptime_seconds(self) -> int:
        return int(time.time() - self.started_at)

    @property
    def status(self) -> str:
        if self._draining:
            return "draining"
        return self.service.status

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    def health(self) -> HealthResponse:
        info = self.service.info
        return HealthResponse(
            status=self.status,
            image_version=self.config.image_version,
            image_digest=self.config.image_digest,
            # A GPU is only claimed present if the live session is genuinely on
            # the CUDA provider. Reporting True merely because nvidia-smi found a
            # card would let a silent CPU fallback pass the gate.
            gpu_available=bool(
                info.provider == self.config.onnx_provider and info.gpu_name
            ),
            gpu_name=info.gpu_name,
            gpu_count=info.gpu_count,
            gpu_memory_total_mb=info.gpu_memory_total_mb,
            gpu_memory_used_mb=self.service.gpu_memory_used_mb(),
            cuda_version=info.cuda_version,
            provider=info.provider,
            providers_available=info.providers_available,
            model_loaded=info.loaded,
            model_name=self.config.model_name,
            model_pack_sha256=info.model_pack_sha256,
            det_size=list(self.config.det_size),
            det_thresh=self.config.det_thresh,
            max_detection_size=self.config.max_detection_size,
            embedding_dim=self.config.embedding_dim,
            embedding_dtype="float32",
            normalized=True,
            warmup_inference_ms=info.warmup_inference_ms,
            uptime_seconds=self.uptime_seconds,
            batches_processed=(
                self.processor.stats.batches_processed if self.processor else 0
            ),
        )

    def status_payload(self, cache_entries: int = 0) -> StatusResponse:
        stats = self.processor.stats if self.processor else None
        return StatusResponse(
            uptime_seconds=self.uptime_seconds,
            status=self.status,
            batches_processed=stats.batches_processed if stats else 0,
            images_processed=stats.images_processed if stats else 0,
            faces_detected=stats.faces_detected if stats else 0,
            images_failed=stats.images_failed if stats else 0,
            in_flight_batches=self.processor.in_flight if self.processor else 0,
            avg_batch_duration_ms=stats.avg_batch_duration_ms if stats else None,
            avg_images_per_second=stats.avg_images_per_second if stats else None,
            gpu_utilization_percent=_gpu_utilization(),
            gpu_memory_used_mb=self.service.gpu_memory_used_mb(),
            cache_entries=cache_entries,
            last_error=stats.last_error if stats else None,
            last_error_at=stats.last_error_at if stats else None,
        )


def _gpu_utilization() -> Optional[int]:
    """Instantaneous GPU utilization, for the benchmark and dashboards."""
    try:
        import subprocess

        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return int(float(output.stdout.strip().splitlines()[0]))
    except Exception:
        return None
