"""
Image decode and pre-detection resize.

THIS FILE MUST MIRROR core/image_processor.py FROM THE INDIA REPOSITORY, EXACTLY.

It is the one deliberate duplication in the whole design (plan §29.3): the GPU
worker lives in a separate repository so it cannot import India's modules, but
this transform decides which pixels reach the model. A different interpolation
mode, a different rounding, or a different size threshold produces a different
embedding — one that is numerically plausible and silently incompatible with the
11.3M-face FAISS index.

The duplication is guarded three ways:
  * the reference implementation is quoted below, line by line;
  * tests/fixtures/ holds arrays produced by India's actual code, and
    tests/test_preprocess.py asserts byte-equality against them;
  * the parity harness (scripts/gpu_parity_check.py, India side) re-checks it
    against production images for every image version.

REFERENCE — core/image_processor.py, resize_for_detection():

    h, w = image.shape[:2]
    if max(h, w) <= self.max_detection_size:
        return image, 1.0
    scale = self.max_detection_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

REFERENCE — core/image_processor.py, load_image_from_bytes():

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

Note IMREAD_COLOR: 3-channel BGR, alpha dropped, 8-bit. Not IMREAD_UNCHANGED.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DecodeError(ValueError):
    """The bytes could not be decoded as an image."""


class ImageTooLargeError(ValueError):
    """The decoded image exceeds the pixel budget (decompression-bomb guard)."""


@dataclass
class PreparedImage:
    """An image ready for detection, plus what is needed to map results back."""

    resized: "np.ndarray"
    """The array actually fed to InsightFace."""

    scale: float
    """resized / original. 1.0 when no resize happened.

    bbox_original = bbox_detection / scale
    """

    width: int
    """ORIGINAL width, before the resize."""

    height: int
    """ORIGINAL height, before the resize."""

    @property
    def resized_shape(self) -> Tuple[int, int]:
        return (self.resized.shape[1], self.resized.shape[0])


def decode(image_bytes: bytes, max_pixels: int = 178_956_970) -> "np.ndarray":
    """Decode bytes to a BGR uint8 array.

    Mirrors ``ImageProcessor.load_image_from_bytes``: ``cv2.imdecode`` with
    ``IMREAD_COLOR``, which forces 3-channel BGR and drops any alpha channel.
    """
    if not image_bytes:
        raise DecodeError("empty image payload")

    buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise DecodeError("cv2.imdecode returned None (not a decodable image)")

    height, width = image.shape[:2]
    if height * width > max_pixels:
        # Guard against a deliberately tiny file that decodes to gigabytes.
        raise ImageTooLargeError(
            f"decoded image is {width}x{height} = {height * width} pixels, "
            f"over the {max_pixels} limit"
        )
    return image


def resize_for_detection(
    image: "np.ndarray", max_detection_size: int
) -> Tuple["np.ndarray", float]:
    """Resize so the longest side is at most `max_detection_size`.

    Byte-for-byte identical to ``ImageProcessor.resize_for_detection``:

      * no-op (scale 1.0) when the image is already small enough — note ``<=``,
        so an image exactly at the limit is NOT resized;
      * ``int()`` truncation of the new dimensions, not rounding;
      * ``cv2.INTER_AREA``, which is the choice that matters most — INTER_LINEAR
        would produce visibly similar images with measurably different embeddings.
    """
    height, width = image.shape[:2]
    if max(height, width) <= max_detection_size:
        return image, 1.0

    scale = max_detection_size / max(height, width)
    new_width, new_height = int(width * scale), int(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def prepare(
    image_bytes: bytes,
    max_detection_size: int,
    max_pixels: int = 178_956_970,
) -> PreparedImage:
    """Decode and resize in one step."""
    original = decode(image_bytes, max_pixels=max_pixels)
    height, width = original.shape[:2]
    resized, scale = resize_for_detection(original, max_detection_size)
    return PreparedImage(resized=resized, scale=scale, width=width, height=height)


def scale_bbox_to_original(bbox, scale: float) -> list:
    """Map a detection-space bbox back to ORIGINAL image pixels.

    Mirrors ``FaceDetector.detect_faces``:

        bbox = face.bbox.tolist()
        if scale != 1.0:
            bbox = [int(x / scale) for x in bbox]
        else:
            bbox = [int(x) for x in bbox]

    The branch is not redundant. Dividing by exactly 1.0 is a float operation
    whose truncation can differ from a direct ``int()`` at the last bit, and
    India takes the second branch when no resize happened. Reproducing the branch
    keeps the two byte-identical.

    This is also why the API returns original-space boxes: India crops from the
    full-resolution image, so a detection-space box would produce crops wrong by
    the resize factor, and crops are what users see.
    """
    values = list(bbox)
    if scale != 1.0:
        return [int(v / scale) for v in values]
    return [int(v) for v in values]


def clamp_bbox(bbox: list, width: int, height: int) -> list:
    """Clamp a bbox into the image.

    A detector can return coordinates slightly outside the frame for a face at
    the edge. India's ``crop_face`` clamps before slicing, so clamping here keeps
    the returned geometry consistent with what will actually be cropped — and
    keeps it inside the bounds India's validator checks.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return [x1, y1, x2, y2]
