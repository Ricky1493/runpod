"""
Request and response models for the GPU worker API (plan §31).

The contract is validated at the edge so a malformed request never reaches the
GPU. Limits are enforced here rather than deeper in, because rejecting early is
what keeps a hostile or buggy caller from consuming inference time.

Every rejection maps to a specific status (plan §31.2):
    422  schema or limit violation
    413  body over MAX_REQUEST_BYTES
    409  batch_id reused with different contents
    401  missing or invalid bearer token
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class ImageRequest(BaseModel):
    """One picture to process."""

    picture_id: int = Field(..., gt=0, description="tab_picturesv2.id")
    image_url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Short-lived presigned HTTPS URL. Treated as a credential: "
                    "never logged, never echoed back.",
    )

    @field_validator("image_url")
    @classmethod
    def _https_only(cls, value: str) -> str:
        # Cheap early check; fetcher.validate_url does the full host allowlist,
        # redirect and private-address checks at fetch time.
        if not value.startswith("https://"):
            raise ValueError("image_url must be an https URL")
        return value


class ProcessRequest(BaseModel):
    """POST /v1/process body."""

    batch_id: UUID = Field(
        ...,
        description="Stable across retries of the same work. The result cache "
                    "keys on it, which makes a retry after a lost response free.",
    )
    burst_cycle_id: Optional[int] = Field(
        None, description="India's cycle id, for log correlation only."
    )
    images: List[ImageRequest] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_picture_ids(self) -> "ProcessRequest":
        seen = set()
        for image in self.images:
            if image.picture_id in seen:
                raise ValueError(
                    f"duplicate picture_id {image.picture_id} in the batch"
                )
            seen.add(image.picture_id)
        return self

    @property
    def picture_ids(self) -> List[int]:
        return [image.picture_id for image in self.images]

    @property
    def pairs(self) -> List[tuple]:
        return [(image.picture_id, image.image_url) for image in self.images]

    def content_signature(self) -> str:
        """Fingerprint of the batch's contents, ignoring URL signatures.

        Used to detect a ``batch_id`` reused with a DIFFERENT set of pictures,
        which indicates a client bug and must not be silently served from cache.
        Picture ids alone are the right basis: a legitimate retry re-mints the
        presigned URLs, so comparing URLs would produce false conflicts.
        """
        import hashlib

        joined = ",".join(str(pid) for pid in sorted(self.picture_ids))
        return hashlib.sha256(joined.encode()).hexdigest()


class FaceResponse(BaseModel):
    """One detected face."""

    bbox: List[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] in ORIGINAL image pixel coordinates, "
                    "already divided by the detection scale.",
    )
    det_score: float
    quality_score: float
    quality_params: Dict[str, Any]
    gender: Optional[str] = None
    age: Optional[int] = None
    embedding: List[float] = Field(
        ...,
        description="512 float32 values, L2-normalized to |v| = 1.0. Dimension "
                    "verified against India's implementation, not assumed.",
    )
    emotion: Optional[str] = None


class ImageResult(BaseModel):
    """The verdict on one picture. A failure here does NOT fail the batch."""

    picture_id: int
    success: bool
    error_code: Optional[str] = None
    error: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    detection_scale: Optional[float] = Field(
        None,
        description="resized/original. Returned for verification only; bboxes "
                    "are already in original coordinates.",
    )
    faces: List[FaceResponse] = Field(default_factory=list)
    timings: Dict[str, int] = Field(default_factory=dict)


class ProcessResponse(BaseModel):
    """POST /v1/process response."""

    batch_id: str
    image_version: str
    model_name: str
    embedding_dim: int
    cached: bool = Field(
        False,
        description="True when this is an idempotent replay of a known batch_id: "
                    "no re-download, no re-inference.",
    )
    processed_count: int
    success_count: int
    failed_count: int
    total_faces: int
    duration_ms: int
    timings: Dict[str, int] = Field(default_factory=dict)
    results: List[ImageResult]


class HealthResponse(BaseModel):
    """GET /v1/health.

    Every field here exists because India's health gate asserts it. The gate
    refuses to dispatch a single batch unless they all match, because a model or
    preprocessing mismatch is a data-integrity incident rather than a transient
    fault.
    """

    status: str = Field(
        ..., description="starting | loading_model | ready | degraded | draining"
    )
    image_version: str
    image_digest: Optional[str] = None
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    gpu_memory_total_mb: Optional[int] = None
    gpu_memory_used_mb: Optional[int] = None
    cuda_version: Optional[str] = None
    provider: Optional[str] = Field(
        None,
        description="The provider the LIVE session is using, not merely one that "
                    "is available. This is what catches a silent CPU fallback.",
    )
    providers_available: List[str] = Field(default_factory=list)
    model_loaded: bool
    model_name: str
    model_pack_sha256: Dict[str, str] = Field(
        default_factory=dict,
        description="SHA-256 of each .onnx file on disk. The decisive parity "
                    "check: it cannot be satisfied by a similar-looking model.",
    )
    det_size: List[int]
    det_thresh: float
    max_detection_size: int
    embedding_dim: int
    embedding_dtype: str = "float32"
    normalized: bool = True
    warmup_inference_ms: Optional[int] = None
    uptime_seconds: int
    batches_processed: int = 0


class StatusResponse(BaseModel):
    """GET /v1/status — operational counters."""

    uptime_seconds: int
    status: str
    batches_processed: int
    images_processed: int
    faces_detected: int
    images_failed: int
    in_flight_batches: int
    avg_batch_duration_ms: Optional[float] = None
    avg_images_per_second: Optional[float] = None
    gpu_utilization_percent: Optional[int] = None
    gpu_memory_used_mb: Optional[int] = None
    cache_entries: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[str] = None


class ErrorResponse(BaseModel):
    """Single error envelope for every 4xx/5xx."""

    error_code: str
    message: str
    batch_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: str
