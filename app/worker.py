"""
Batch orchestration: fetch, decode, infer, assemble.

Two properties define this file:

  PER-IMAGE FAILURE ISOLATION. One bad URL, one corrupt JPEG or one OOM never
  fails the batch. Each picture gets its own ``success`` and ``error_code``, so a
  single broken object cannot cost the other 31 pictures their inference. The
  batch-level HTTP status stays 200; per-image failure is the normal error
  channel, not an exception.

  A HARD DEADLINE, BELOW THE PROXY CEILING. The RunPod proxy returns an opaque
  Cloudflare 524 past 100 seconds, which the client cannot interpret. So the
  worker stops at ``REQUEST_DEADLINE_SECONDS`` (75 by default) and returns
  whatever finished, marking the rest ``error_code: "deadline"``. A partial batch
  India can act on is strictly better than a 524 it cannot.

Concurrency shape: fetching is async and bounded; decode runs in a thread pool
because it is CPU-bound and releases the GIL inside OpenCV; inference is
serialized by the service's lock because there is one GPU and one model.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import WorkerConfig
from .fetcher import FetchResult, ImageFetcher
from .insightface_service import InsightFaceService
from .preprocess import (
    DecodeError,
    ImageTooLargeError,
    PreparedImage,
    prepare,
)
from .schemas import FaceResponse, ImageResult, ProcessRequest, ProcessResponse

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    batches_processed: int = 0
    images_processed: int = 0
    images_failed: int = 0
    faces_detected: int = 0
    total_batch_ms: int = 0
    deadline_truncations: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[str] = None

    @property
    def avg_batch_duration_ms(self) -> Optional[float]:
        if not self.batches_processed:
            return None
        return round(self.total_batch_ms / self.batches_processed, 1)

    @property
    def avg_images_per_second(self) -> Optional[float]:
        if not self.total_batch_ms:
            return None
        return round(self.images_processed / (self.total_batch_ms / 1000.0), 2)


class BatchProcessor:
    """Runs one /v1/process request end to end."""

    def __init__(self, config: WorkerConfig, service: InsightFaceService):
        self.config = config
        self.service = service
        self.stats = WorkerStats()
        self._in_flight = 0
        self._in_flight_lock = asyncio.Lock()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def process(self, request: ProcessRequest) -> ProcessResponse:
        """Fetch, decode and infer a whole batch."""
        loop = asyncio.get_event_loop()
        started = loop.time()
        deadline = started + self.config.request_deadline_seconds
        batch_id = str(request.batch_id)

        async with self._in_flight_lock:
            self._in_flight += 1
        try:
            logger.info(
                "Batch %s: %d images (deadline %ds, cycle=%s)",
                batch_id[:8], len(request.images),
                self.config.request_deadline_seconds, request.burst_cycle_id,
            )

            # --- 1. fetch ------------------------------------------------
            fetch_started = loop.time()
            async with ImageFetcher(
                allowed_hosts=self.config.allowed_url_hosts,
                concurrency=self.config.download_concurrency,
                timeout_seconds=self.config.download_timeout_seconds,
                max_bytes=self.config.max_image_bytes,
            ) as fetcher:
                fetched = await fetcher.fetch_many(request.pairs, deadline=deadline)
            fetch_ms = int((loop.time() - fetch_started) * 1000)

            # --- 2. decode (thread pool: CPU-bound, releases the GIL) -----
            decode_started = loop.time()
            prepared: Dict[int, PreparedImage] = {}
            decode_errors: Dict[int, Tuple[str, str]] = {}

            decodable = [
                (picture_id, result)
                for picture_id, result in fetched.items()
                if result.ok
            ]
            if decodable:
                outcomes = await asyncio.gather(
                    *(
                        loop.run_in_executor(
                            None, self._decode_one, picture_id, result.content
                        )
                        for picture_id, result in decodable
                    )
                )
                for picture_id, image, error in outcomes:
                    if image is not None:
                        prepared[picture_id] = image
                    elif error is not None:
                        decode_errors[picture_id] = error
            decode_ms = int((loop.time() - decode_started) * 1000)

            # --- 3. inference (serialized on the GPU) ---------------------
            inference_started = loop.time()
            detections: Dict[int, List] = {}
            inference_errors: Dict[int, Tuple[str, str]] = {}
            timed_out: List[int] = []

            for picture_id, image in prepared.items():
                if loop.time() > deadline:
                    timed_out.append(picture_id)
                    continue
                try:
                    faces = await loop.run_in_executor(
                        None, self.service.detect, image
                    )
                    detections[picture_id] = faces
                except Exception as exc:
                    error_code = (
                        "oom"
                        if "out of memory" in str(exc).lower()
                        else "inference_failed"
                    )
                    logger.error(
                        "Inference failed for picture %d: %s", picture_id, exc
                    )
                    inference_errors[picture_id] = (error_code, str(exc)[:500])
            inference_ms = int((loop.time() - inference_started) * 1000)

            if timed_out:
                self.stats.deadline_truncations += 1
                logger.warning(
                    "Batch %s hit the %ds deadline with %d images unprocessed; "
                    "returning a partial result rather than risking a proxy 524",
                    batch_id[:8], self.config.request_deadline_seconds,
                    len(timed_out),
                )

            # --- 4. assemble ---------------------------------------------
            results: List[ImageResult] = []
            for image_request in request.images:
                picture_id = image_request.picture_id
                results.append(
                    self._build_result(
                        picture_id,
                        fetched.get(picture_id),
                        prepared.get(picture_id),
                        detections.get(picture_id),
                        decode_errors.get(picture_id),
                        inference_errors.get(picture_id),
                        picture_id in timed_out,
                    )
                )

            duration_ms = int((loop.time() - started) * 1000)
            success_count = sum(1 for r in results if r.success)
            total_faces = sum(len(r.faces) for r in results)

            self.stats.batches_processed += 1
            self.stats.images_processed += len(results)
            self.stats.images_failed += len(results) - success_count
            self.stats.faces_detected += total_faces
            self.stats.total_batch_ms += duration_ms

            logger.info(
                "Batch %s done in %dms: %d/%d ok, %d faces "
                "(fetch %dms, decode %dms, inference %dms)",
                batch_id[:8], duration_ms, success_count, len(results),
                total_faces, fetch_ms, decode_ms, inference_ms,
            )

            return ProcessResponse(
                batch_id=batch_id,
                image_version=self.config.image_version,
                model_name=self.config.model_name,
                embedding_dim=self.config.embedding_dim,
                cached=False,
                processed_count=len(results),
                success_count=success_count,
                failed_count=len(results) - success_count,
                total_faces=total_faces,
                duration_ms=duration_ms,
                timings={
                    "fetch_ms": fetch_ms,
                    "decode_ms": decode_ms,
                    "inference_ms": inference_ms,
                },
                results=results,
            )
        finally:
            async with self._in_flight_lock:
                self._in_flight -= 1

    def _decode_one(
        self, picture_id: int, content: bytes
    ) -> Tuple[int, Optional[PreparedImage], Optional[Tuple[str, str]]]:
        """Decode and resize one image. Runs in a worker thread."""
        try:
            return (
                picture_id,
                prepare(
                    content,
                    self.config.max_detection_size,
                    max_pixels=self.config.max_image_pixels,
                ),
                None,
            )
        except ImageTooLargeError as exc:
            return picture_id, None, ("too_large", str(exc))
        except DecodeError as exc:
            return picture_id, None, ("decode_failed", str(exc))
        except Exception as exc:
            logger.error("Unexpected decode failure for picture %d: %s",
                         picture_id, exc)
            return picture_id, None, ("decode_failed", str(exc)[:500])

    def _build_result(
        self,
        picture_id: int,
        fetch: Optional[FetchResult],
        image: Optional[PreparedImage],
        faces: Optional[List],
        decode_error: Optional[Tuple[str, str]],
        inference_error: Optional[Tuple[str, str]],
        deadline_hit: bool,
    ) -> ImageResult:
        """Turn per-stage outcomes into one picture's verdict.

        Ordered so the FIRST failure encountered is the one reported: a fetch
        failure explains a missing decode, and reporting the downstream symptom
        instead would send India chasing the wrong cause.
        """
        timings: Dict[str, int] = {}
        if fetch is not None:
            timings["fetch_ms"] = fetch.duration_ms

        if fetch is None:
            return ImageResult(
                picture_id=picture_id, success=False, error_code="internal",
                error="no fetch outcome was recorded for this picture",
                timings=timings,
            )

        if not fetch.ok:
            return ImageResult(
                picture_id=picture_id, success=False,
                error_code=fetch.error_code or "download_failed",
                error=fetch.error, timings=timings,
            )

        if decode_error is not None:
            code, message = decode_error
            return ImageResult(
                picture_id=picture_id, success=False, error_code=code,
                error=message, timings=timings,
            )

        if deadline_hit:
            return ImageResult(
                picture_id=picture_id, success=False, error_code="deadline",
                error="the batch deadline elapsed before this image was inferred",
                image_width=image.width if image else None,
                image_height=image.height if image else None,
                timings=timings,
            )

        if inference_error is not None:
            code, message = inference_error
            return ImageResult(
                picture_id=picture_id, success=False, error_code=code,
                error=message,
                image_width=image.width if image else None,
                image_height=image.height if image else None,
                timings=timings,
            )

        if image is None or faces is None:
            return ImageResult(
                picture_id=picture_id, success=False, error_code="internal",
                error="inference produced no outcome for this picture",
                timings=timings,
            )

        return ImageResult(
            picture_id=picture_id,
            success=True,
            image_width=image.width,
            image_height=image.height,
            detection_scale=image.scale,
            faces=[
                FaceResponse(
                    bbox=face.bbox,
                    det_score=face.det_score,
                    quality_score=face.quality_score,
                    quality_params=face.quality_params,
                    gender=face.gender,
                    age=face.age,
                    embedding=face.embedding,
                    emotion=face.emotion,
                )
                for face in faces
            ],
            timings=timings,
        )
