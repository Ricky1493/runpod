"""
Preprocessing parity tests.

app/preprocess.py is a hand-copy of core/image_processor.py from the India
repository. This is the guard on that copy, and it is the most important test file
in this service.

The reason: detection runs on the RESIZED image. If the resize differs from
India's by so much as an interpolation mode, a different set of pixels reaches the
model, the embedding changes, and it stops matching an 11.3M-face FAISS index —
while still looking like a perfectly valid unit vector. Nothing downstream would
notice.

Three specific things these tests pin down, each of which is easy to "clean up"
into a bug:
  * ``<=`` in the no-resize branch, so an image exactly at the limit is untouched;
  * ``int()`` truncation of the new dimensions, not ``round()``;
  * ``cv2.INTER_AREA``, which produces visibly similar but numerically different
    output from INTER_LINEAR.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.preprocess import (
    DecodeError,
    ImageTooLargeError,
    clamp_bbox,
    decode,
    prepare,
    resize_for_detection,
    scale_bbox_to_original,
)

MAX_DETECTION_SIZE = 640


def make_image(width: int, height: int) -> np.ndarray:
    """A deterministic non-uniform BGR image.

    A gradient rather than flat colour on purpose: a flat image resizes
    identically under any interpolation mode, so it could not detect an
    INTER_AREA -> INTER_LINEAR change, which is the thing these tests exist to
    catch.

    Built in int32 and cast at the end. Doing the modulo in uint8 raises under
    numpy 2.x, and would silently wrap under numpy 1.x.
    """
    rows = np.arange(height, dtype=np.int32).reshape(-1, 1)
    cols = np.arange(width, dtype=np.int32).reshape(1, -1)
    blue = (rows + cols) % 256
    green = (rows * 2 + cols) % 256
    red = (rows + cols * 3) % 256
    stacked = np.stack(np.broadcast_arrays(blue, green, red), axis=-1)
    return stacked.astype(np.uint8)


def encode_jpeg(image: np.ndarray, quality: int = 95) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buffer.tobytes()


# =============================================================================
# resize_for_detection
# =============================================================================


class TestResizeForDetection:
    def test_small_image_is_untouched_and_scale_is_exactly_one(self):
        image = make_image(400, 300)
        resized, scale = resize_for_detection(image, MAX_DETECTION_SIZE)
        assert scale == 1.0
        # Identity, not a copy-with-same-values: India returns the same object.
        assert np.array_equal(resized, image)

    def test_image_exactly_at_the_limit_is_not_resized(self):
        """The reference uses `<=`. An off-by-one here would resize a 640px image
        that India leaves alone, changing its embedding."""
        image = make_image(640, 480)
        resized, scale = resize_for_detection(image, MAX_DETECTION_SIZE)
        assert scale == 1.0
        assert resized.shape == image.shape

    def test_one_pixel_over_the_limit_is_resized(self):
        image = make_image(641, 480)
        _resized, scale = resize_for_detection(image, MAX_DETECTION_SIZE)
        assert scale != 1.0

    def test_landscape_scales_on_width(self):
        image = make_image(4032, 3024)
        resized, scale = resize_for_detection(image, MAX_DETECTION_SIZE)
        assert scale == pytest.approx(640 / 4032)
        assert resized.shape[1] == int(4032 * scale)
        assert resized.shape[0] == int(3024 * scale)
        assert max(resized.shape[:2]) == 640

    def test_portrait_scales_on_height(self):
        image = make_image(3024, 4032)
        resized, scale = resize_for_detection(image, MAX_DETECTION_SIZE)
        assert scale == pytest.approx(640 / 4032)
        assert max(resized.shape[:2]) == 640

    def test_dimensions_are_truncated_not_rounded(self):
        """int() truncation, exactly as the reference does.

        1000x777 at scale 0.64: 777*0.64 = 497.28, which truncates to 497 and
        would round to 497 as well — so use a width where they differ.
        1001 * (640/1001) is exactly 640, but the other side is the test:
        333 * 0.6393... = 212.9 -> 212 truncated, 213 rounded.
        """
        image = make_image(1001, 333)
        resized, scale = resize_for_detection(image, MAX_DETECTION_SIZE)
        expected_height = int(333 * scale)
        assert resized.shape[0] == expected_height
        assert expected_height == 212, (
            f"expected truncation to 212, got {expected_height}; if this is 213 "
            f"the implementation is rounding instead of truncating"
        )

    def test_uses_inter_area_not_inter_linear(self):
        """The interpolation mode is part of the parity contract.

        INTER_LINEAR output is visibly similar and numerically different, which is
        precisely the kind of change that would pass casual review and silently
        degrade matching.
        """
        image = make_image(2000, 1500)
        resized, scale = resize_for_detection(image, MAX_DETECTION_SIZE)

        new_size = (int(2000 * scale), int(1500 * scale))
        expected_area = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        alternative = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        assert np.array_equal(resized, expected_area)
        assert not np.array_equal(resized, alternative), (
            "INTER_AREA and INTER_LINEAR produced identical output for this "
            "fixture, so this test cannot detect the difference — pick a "
            "different image"
        )


# =============================================================================
# decode
# =============================================================================


class TestDecode:
    def test_round_trips_a_jpeg_to_bgr_uint8(self):
        image = make_image(320, 240)
        decoded = decode(encode_jpeg(image))
        assert decoded.dtype == np.uint8
        assert decoded.shape == (240, 320, 3)

    def test_drops_alpha_because_imread_color_is_used(self):
        """IMREAD_COLOR forces 3 channels. IMREAD_UNCHANGED would keep 4 and
        change what the model sees."""
        rgba = np.dstack([make_image(64, 64), np.full((64, 64), 128, np.uint8)])
        ok, buffer = cv2.imencode(".png", rgba)
        assert ok
        decoded = decode(buffer.tobytes())
        assert decoded.shape == (64, 64, 3)

    def test_empty_payload_is_a_decode_error(self):
        with pytest.raises(DecodeError):
            decode(b"")

    def test_garbage_is_a_decode_error_not_a_crash(self):
        with pytest.raises(DecodeError):
            decode(b"this is definitely not an image")

    def test_pixel_budget_is_enforced(self):
        """Decompression-bomb guard: a small file can decode to a huge array."""
        image = make_image(2000, 2000)
        with pytest.raises(ImageTooLargeError):
            decode(encode_jpeg(image), max_pixels=1_000_000)


# =============================================================================
# prepare
# =============================================================================


class TestPrepare:
    def test_reports_original_dimensions_not_resized_ones(self):
        """India needs the ORIGINAL size: bboxes are in original coordinates and
        crops come from the full-resolution image."""
        image = make_image(4032, 3024)
        prepared = prepare(encode_jpeg(image), MAX_DETECTION_SIZE)
        assert prepared.width == 4032
        assert prepared.height == 3024
        assert max(prepared.resized.shape[:2]) == 640
        assert prepared.scale == pytest.approx(640 / 4032)

    def test_small_image_keeps_scale_one(self):
        prepared = prepare(encode_jpeg(make_image(320, 240)), MAX_DETECTION_SIZE)
        assert prepared.scale == 1.0
        assert prepared.width == 320


# =============================================================================
# bbox mapping
# =============================================================================


class TestScaleBboxToOriginal:
    def test_maps_detection_space_back_to_original(self):
        scale = 0.15873015873015872  # 640/4032
        bbox = scale_bbox_to_original([191.0, 128.9, 238.4, 190.2], scale)
        assert bbox == [
            int(191.0 / scale),
            int(128.9 / scale),
            int(238.4 / scale),
            int(190.2 / scale),
        ]

    def test_scale_one_takes_the_int_branch_not_the_divide_branch(self):
        """The reference has two branches and this reproduces both.

        Not redundant: dividing by exactly 1.0 is a float operation whose
        truncation can differ from a direct int() at the last bit, and India takes
        the second branch when no resize happened.
        """
        raw = [10.9, 20.9, 30.9, 40.9]
        assert scale_bbox_to_original(raw, 1.0) == [10, 20, 30, 40]

    def test_all_values_are_integers(self):
        bbox = scale_bbox_to_original([1.5, 2.5, 3.5, 4.5], 0.5)
        assert all(isinstance(v, int) for v in bbox)


class TestClampBbox:
    def test_leaves_an_interior_box_alone(self):
        assert clamp_bbox([10, 20, 100, 200], 640, 480) == [10, 20, 100, 200]

    def test_clamps_negative_origin(self):
        assert clamp_bbox([-5, -10, 100, 200], 640, 480) == [0, 0, 100, 200]

    def test_clamps_overflow_to_the_image_bounds(self):
        """A detector can return coordinates outside the frame for an edge face.
        India's crop_face clamps before slicing, so the returned geometry must
        match what will actually be cropped."""
        assert clamp_bbox([600, 400, 700, 500], 640, 480) == [600, 400, 640, 480]
