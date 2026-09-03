"""
InsightFace model loading and inference.

Three responsibilities, in order of importance:

  1. NUMERICAL PARITY with India's CPU path. The embedding is taken from
     ``face.embedding`` and normalized manually — NOT ``face.normed_embedding``.
     Those two are nearly identical, which is precisely the danger: using the
     wrong one would hide a real difference if one ever appeared.

  2. PROVING WHICH MODEL RAN. ``onnxruntime-gpu`` silently falls back to
     ``CPUExecutionProvider`` when the host CUDA/cuDNN does not match the build.
     That would pass a naive "is a GPU present?" check while destroying throughput
     and changing numerics. So we record the provider the LIVE SESSION reports,
     not merely what is available, and we hash the model files on disk.

  3. SERIALIZING GPU ACCESS. One GPU, one loaded model, one lock. Concurrent
     sessions would contend for the device and risk VRAM thrash for no throughput
     gain.

REFERENCE — core/face_detector.py::detect_faces():

    faces = self.app.get(image)
    for face in faces:
        if face.embedding is None:
            continue
        embedding = face.embedding.astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        bbox = face.bbox.tolist()
        if scale != 1.0:
            bbox = [int(x / scale) for x in bbox]
        else:
            bbox = [int(x) for x in bbox]
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import WorkerConfig
from .preprocess import PreparedImage, clamp_bbox, scale_bbox_to_original
from .quality import age_label, calculate_quality, gender_label

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """The model could not be loaded, or is not the model we were told to load."""


class ProviderMismatchError(RuntimeError):
    """The session is not using the required execution provider."""


@dataclass
class DetectedFace:
    """One face, in the shape the API returns."""

    bbox: List[int]
    embedding: List[float]
    det_score: float
    quality_score: float
    quality_params: Dict[str, Any]
    gender: Optional[str] = None
    age: Optional[int] = None
    emotion: Optional[str] = None
    """Always None. InsightFace does not produce it in this pipeline and India's
    CPU path writes NULL; kept for exact column parity."""


@dataclass
class ModelInfo:
    """Everything needed to prove which model is running."""

    model_name: str = ""
    provider: Optional[str] = None
    providers_available: List[str] = field(default_factory=list)
    model_pack_sha256: Dict[str, str] = field(default_factory=dict)
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    gpu_memory_total_mb: Optional[int] = None
    cuda_version: Optional[str] = None
    warmup_inference_ms: Optional[int] = None
    loaded: bool = False


class InsightFaceService:
    """Wraps InsightFace's FaceAnalysis with parity and provenance guarantees."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self._app = None
        self._lock = threading.Lock()
        self.info = ModelInfo(model_name=config.model_name)
        self._status = "starting"

    @property
    def status(self) -> str:
        return self._status

    @property
    def ready(self) -> bool:
        return self._status == "ready"

    # =====================================================================
    # Loading
    # =====================================================================

    def load(self) -> None:
        """Load the model, verify the provider, and warm up.

        Called once during app startup, before readiness flips. Raises on any
        problem: a worker that cannot prove what it is running must not serve.
        """
        self._status = "loading_model"
        started = time.time()

        # Hash the model files BEFORE loading, so the manifest reflects what is
        # actually on disk rather than anything the library might report.
        self.info.model_pack_sha256 = self._hash_model_pack()
        if not self.info.model_pack_sha256:
            raise ModelLoadError(
                f"No .onnx model files found under {self.config.model_root!r} for "
                f"model {self.config.model_name!r}. Models are baked into the "
                f"image at build time; this means the build is wrong, and the "
                f"worker must not fall back to downloading them at runtime."
            )

        self._collect_gpu_info()

        from insightface.app import FaceAnalysis  # noqa: PLC0415

        logger.info(
            "Loading InsightFace model=%s providers=[%s] ctx_id=%d det_size=%s "
            "from %s",
            self.config.model_name,
            self.config.onnx_provider,
            self.config.ctx_id,
            self.config.det_size,
            self.config.model_root,
        )

        try:
            app = FaceAnalysis(
                name=self.config.model_name,
                root=self.config.model_root,
                providers=[self.config.onnx_provider, "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=self.config.ctx_id, det_size=self.config.det_size)
            # Set after prepare(), exactly as India does — prepare() overwrites it.
            app.det_thresh = self.config.det_thresh
        except Exception as exc:
            raise ModelLoadError(f"FaceAnalysis failed to load: {exc}") from exc

        self._app = app
        self._verify_provider()
        self.info.loaded = True

        warmup_ms = self._warmup()
        self.info.warmup_inference_ms = warmup_ms
        self._status = "ready"

        logger.info(
            "Model ready in %.1fs (warmup %dms): provider=%s gpu=%s cuda=%s "
            "files=%s",
            time.time() - started,
            warmup_ms,
            self.info.provider,
            self.info.gpu_name,
            self.info.cuda_version,
            sorted(self.info.model_pack_sha256),
        )

    def _hash_model_pack(self) -> Dict[str, str]:
        """SHA-256 every .onnx file in the model directory.

        The decisive parity check: it is the only assertion that cannot be
        satisfied by a coincidentally-similar model. India compares this against
        the hashes of its own model files.
        """
        hashes: Dict[str, str] = {}
        search_roots = [
            os.path.join(self.config.model_root, "models", self.config.model_name),
            os.path.join(self.config.model_root, self.config.model_name),
            self.config.model_root,
        ]

        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in sorted(filenames):
                    if not filename.endswith(".onnx"):
                        continue
                    path = os.path.join(dirpath, filename)
                    digest = hashlib.sha256()
                    with open(path, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                            digest.update(chunk)
                    hashes[filename] = digest.hexdigest()
            if hashes:
                break
        return hashes

    def _collect_gpu_info(self) -> None:
        """Record GPU identity. Best-effort: absence is reported, not fatal here.

        The hard GPU requirement is enforced by _verify_provider and by India's
        health gate, which is the right place for it — a policy decision belongs
        with the caller, not buried in a probe.
        """
        try:
            import onnxruntime as ort  # noqa: PLC0415

            self.info.providers_available = list(ort.get_available_providers())
        except Exception as exc:
            logger.warning("Could not list ONNX providers: %s", exc)

        try:
            import subprocess  # noqa: PLC0415

            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            lines = [ln.strip() for ln in output.stdout.splitlines() if ln.strip()]
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                self.info.gpu_name = parts[0]
                self.info.gpu_count = len(lines)
                if len(parts) > 1:
                    self.info.gpu_memory_total_mb = int(float(parts[1]))
        except Exception as exc:
            logger.warning("nvidia-smi probe failed: %s", exc)

        try:
            import onnxruntime as ort  # noqa: PLC0415

            self.info.cuda_version = (
                ort.get_build_info().split("CUDA Version ")[-1].split()[0]
                if "CUDA Version" in ort.get_build_info()
                else None
            )
        except Exception:
            self.info.cuda_version = None

    def _verify_provider(self) -> None:
        """Assert the LIVE session uses the required provider.

        This is the check that catches a silent CPU fallback. Availability in
        ``get_available_providers()`` is not evidence: the session picks at
        creation time, and a CUDA/cuDNN mismatch makes it pick CPU without
        raising.
        """
        providers = self._session_providers()
        if not providers:
            raise ProviderMismatchError(
                "Could not determine the execution provider of the loaded "
                "session; refusing to serve, because a silent CPU fallback would "
                "mean paying GPU rates for CPU speed with different numerics."
            )

        self.info.provider = providers[0]
        if self.config.onnx_provider not in providers:
            raise ProviderMismatchError(
                f"Session is using {providers} but "
                f"{self.config.onnx_provider} is required. onnxruntime-gpu falls "
                f"back to CPU when the host CUDA/cuDNN does not match the build. "
                f"Available: {self.info.providers_available}"
            )

    def _session_providers(self) -> List[str]:
        """Providers reported by the recognition model's live ORT session."""
        app = self._app
        if app is None:
            return []
        for attr in ("models", "model_zoo"):
            container = getattr(app, attr, None)
            if not isinstance(container, dict):
                continue
            for model in container.values():
                session = getattr(model, "session", None)
                if session is not None and hasattr(session, "get_providers"):
                    try:
                        return list(session.get_providers())
                    except Exception:
                        continue
        return []

    def _warmup(self) -> int:
        """Run one inference on a synthetic image.

        Two purposes: it pays the one-off CUDA kernel-compilation cost before the
        first real batch, and it proves the whole path works before readiness
        flips. A worker that reports ready without this could accept a batch and
        then fail on the first image.
        """
        canvas = np.zeros(
            (self.config.det_size[1], self.config.det_size[0], 3), dtype=np.uint8
        )
        # A mid-grey rectangle is enough to exercise the graph; we do not care
        # whether a face is found.
        canvas[:] = 114

        # Explicit guard rather than relying on call order. _app is Optional, and
        # without this a warm-up before load() surfaces as
        # "warm-up inference failed: 'NoneType' object has no attribute 'get'" —
        # which sends the reader looking for a broken model instead of a broken
        # sequence.
        if self._app is None:
            raise ModelLoadError(
                "warm-up was called before the model was loaded; load() must "
                "complete first"
            )

        started = time.time()
        try:
            with self._lock:
                self._app.get(canvas)
        except Exception as exc:
            raise ModelLoadError(f"warm-up inference failed: {exc}") from exc
        return int((time.time() - started) * 1000)

    # =====================================================================
    # Inference
    # =====================================================================

    def detect(self, prepared: PreparedImage) -> List[DetectedFace]:
        """Detect faces and extract embeddings.

        Reproduces ``FaceDetector.detect_faces`` step for step. Returns bboxes in
        ORIGINAL image coordinates.
        """
        if self._app is None:
            raise ModelLoadError("model is not loaded")

        # One GPU, one model, one inference at a time.
        with self._lock:
            faces = self._app.get(prepared.resized)

        detected: List[DetectedFace] = []
        for face in faces:
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                # India skips these too: a detection without a recognition vector
                # is useless downstream.
                continue

            # float32 then MANUAL L2 normalization, from face.embedding —
            # deliberately not face.normed_embedding.
            vector = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm <= 0 or not np.isfinite(norm):
                logger.warning("Skipping a face with a degenerate embedding norm")
                continue
            vector = vector / norm

            bbox = scale_bbox_to_original(face.bbox.tolist(), prepared.scale)
            bbox = clamp_bbox(bbox, prepared.width, prepared.height)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                logger.warning(
                    "Skipping a face whose bbox collapsed after " "clamping: %s", bbox
                )
                continue

            det_score = float(getattr(face, "det_score", 0.5))
            quality_score, quality_params = calculate_quality(det_score, bbox)

            detected.append(
                DetectedFace(
                    bbox=bbox,
                    embedding=[float(v) for v in vector],
                    det_score=det_score,
                    quality_score=quality_score,
                    quality_params=quality_params,
                    gender=gender_label(getattr(face, "gender", None)),
                    age=age_label(getattr(face, "age", None)),
                    emotion=None,
                )
            )

        return detected

    def gpu_memory_used_mb(self) -> Optional[int]:
        try:
            import subprocess  # noqa: PLC0415

            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            line = output.stdout.strip().splitlines()[0]
            return int(float(line.strip()))
        except Exception:
            return None
