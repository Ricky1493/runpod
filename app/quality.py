"""
Quality scoring.

THIS FILE MUST MIRROR core/face_detector.py::_calculate_quality_score EXACTLY.

Why an apparently cosmetic score is parity-critical: ``faiss_updater`` selects
which faces enter the searchable index by
``ORDER BY quality_score DESC`` with a cap of MAX_EMBEDDINGS_PER_PERSON (5) and a
diversity threshold. So the quality score does not just decorate a row — it
decides WHICH faces become findable. A drifted score changes search results
without changing any embedding.

REFERENCE — core/face_detector.py:

    det_score = float(face.det_score) if hasattr(face, 'det_score') else 0.5

    width  = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    size   = width * height
    size_score = min(1.0, size / (256 * 256))

    aspect_ratio = width / max(height, 1)
    aspect_score = 1.0 - abs(1.0 - aspect_ratio) * 0.5

    quality = (det_score * 0.5 + size_score * 0.3 + aspect_score * 0.2)

    quality_params = {
        'det_score':    round(det_score, 4),
        'size_score':   round(size_score, 4),
        'aspect_score': round(aspect_score, 4),
        'face_width':   width,
        'face_height':  height,
        'face_area':    size,
        'aspect_ratio': round(aspect_ratio, 4),
        'weights': {'det': 0.5, 'size': 0.3, 'aspect': 0.2},
    }

    return round(quality, 4), quality_params

Two details that are easy to lose and both matter:

  * The bbox used is the ORIGINAL-COORDINATE one, computed after dividing by the
    detection scale. Using detection-space dimensions would shrink face_area by
    the square of the scale factor and change size_score for nearly every face.
  * ``max(height, 1)`` guards a degenerate box; keep it rather than "improving"
    it to a zero check, so behaviour on odd input matches too.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

#: The weights, named so a future change is a visible, deliberate act.
WEIGHT_DET = 0.5
WEIGHT_SIZE = 0.3
WEIGHT_ASPECT = 0.2

#: Reference face area for size_score normalization: a 256x256 face scores 1.0.
SIZE_REFERENCE_AREA = 256 * 256


def calculate_quality(
    det_score: float, bbox: List[int]
) -> Tuple[float, Dict[str, Any]]:
    """Compute (quality_score, quality_params) for one face.

    Args:
        det_score: the detector's confidence.
        bbox: [x1, y1, x2, y2] in ORIGINAL image coordinates.

    Returns:
        (score rounded to 4dp, params dict matching India's key-for-key).
    """
    det_score = float(det_score)

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    size = width * height

    size_score = min(1.0, size / SIZE_REFERENCE_AREA)

    aspect_ratio = width / max(height, 1)
    aspect_score = 1.0 - abs(1.0 - aspect_ratio) * 0.5

    quality = (
        det_score * WEIGHT_DET + size_score * WEIGHT_SIZE + aspect_score * WEIGHT_ASPECT
    )

    params: Dict[str, Any] = {
        "det_score": round(det_score, 4),
        "size_score": round(size_score, 4),
        "aspect_score": round(aspect_score, 4),
        "face_width": width,
        "face_height": height,
        "face_area": size,
        "aspect_ratio": round(aspect_ratio, 4),
        "weights": {
            "det": WEIGHT_DET,
            "size": WEIGHT_SIZE,
            "aspect": WEIGHT_ASPECT,
        },
    }

    return round(quality, 4), params


def gender_label(gender_value: Any) -> str | None:
    """Map InsightFace's numeric gender to India's label.

    REFERENCE — core/face_detector.py::_get_gender:
        return 'M' if face.gender == 1 else 'F'

    Note the exact semantics: anything that is not 1 becomes 'F', including 0.
    Reproduced rather than "fixed", because India's stored data follows this rule.
    """
    if gender_value is None:
        return None
    return "M" if int(gender_value) == 1 else "F"


def age_label(age_value: Any) -> int | None:
    """REFERENCE — core/face_detector.py::_get_age: ``int(face.age)``."""
    if age_value is None:
        return None
    return int(age_value)
