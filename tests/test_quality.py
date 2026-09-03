"""
Quality-score parity tests.

app/quality.py is a hand-copy of core/face_detector.py::_calculate_quality_score
from the India repository. These tests are the guard on that copy.

Why it matters more than it looks: faiss_updater selects which faces enter the
searchable index with ``ORDER BY quality_score DESC``, capped at 5 embeddings per
person. So a drifted score changes WHICH faces are findable, without changing a
single embedding — a silent search-quality regression.

The expected values below were computed from the reference formula
    quality = det_score*0.5 + size_score*0.3 + aspect_score*0.2
with size_score = min(1.0, w*h/65536) and aspect_score = 1 - |1-w/max(h,1)|*0.5,
all rounded to 4dp.
"""
from __future__ import annotations

import pytest

from app.quality import (
    SIZE_REFERENCE_AREA,
    WEIGHT_ASPECT,
    WEIGHT_DET,
    WEIGHT_SIZE,
    age_label,
    calculate_quality,
    gender_label,
)


def test_weights_match_the_reference():
    """The weights are the contract. A change here changes search results."""
    assert WEIGHT_DET == 0.5
    assert WEIGHT_SIZE == 0.3
    assert WEIGHT_ASPECT == 0.2
    assert WEIGHT_DET + WEIGHT_SIZE + WEIGHT_ASPECT == 1.0
    assert SIZE_REFERENCE_AREA == 256 * 256


def test_perfect_square_face_at_reference_size():
    """A 256x256 face at det_score 1.0 scores 1.0 exactly."""
    score, params = calculate_quality(1.0, [0, 0, 256, 256])
    assert score == 1.0
    assert params["size_score"] == 1.0
    assert params["aspect_score"] == 1.0
    assert params["face_area"] == 65536


def test_size_score_is_capped_at_one():
    """A face larger than the reference area does not score above 1.0."""
    _score, params = calculate_quality(1.0, [0, 0, 1000, 1000])
    assert params["size_score"] == 1.0


def test_small_face_scales_by_area_not_by_side():
    """size_score is area-based: a 128x128 face is a QUARTER, not a half."""
    _score, params = calculate_quality(1.0, [0, 0, 128, 128])
    assert params["face_area"] == 16384
    assert params["size_score"] == pytest.approx(0.25)


def test_tall_face_is_penalised_on_aspect():
    _score, params = calculate_quality(0.9, [0, 0, 100, 200])
    assert params["aspect_ratio"] == 0.5
    # 1 - |1 - 0.5| * 0.5
    assert params["aspect_score"] == pytest.approx(0.75)


def test_wide_face_is_penalised_symmetrically():
    _score, params = calculate_quality(0.9, [0, 0, 200, 100])
    assert params["aspect_ratio"] == 2.0
    assert params["aspect_score"] == pytest.approx(0.5)


def test_known_reference_value():
    """A worked example, computed by hand from the reference formula.

    bbox 298x386 (a realistic portrait face), det_score 0.8912:
        size_score  = min(1, 115028/65536) = 1.0
        aspect      = 298/386             = 0.7720 (4dp)
        aspect_score= 1 - |1-0.772|*0.5   = 0.8860
        quality     = 0.8912*0.5 + 1.0*0.3 + 0.8860*0.2 = 0.9228
    """
    score, params = calculate_quality(0.8912, [1204, 812, 1502, 1198])
    assert params["face_width"] == 298
    assert params["face_height"] == 386
    assert params["face_area"] == 115028
    assert params["size_score"] == 1.0
    assert params["aspect_ratio"] == 0.772
    assert params["aspect_score"] == pytest.approx(0.886, abs=1e-4)
    assert score == pytest.approx(0.9228, abs=1e-4)


def test_degenerate_zero_height_uses_the_max_guard():
    """max(height, 1) must be preserved, not 'improved' to a zero check.

    India's stored data follows this rule, so reproducing it matters more than
    tidiness.
    """
    score, params = calculate_quality(0.5, [0, 0, 10, 0])
    assert params["face_height"] == 0
    assert params["aspect_ratio"] == 10.0
    assert score == pytest.approx(
        0.5 * 0.5 + 0.0 * 0.3 + (1.0 - 9.0 * 0.5) * 0.2, abs=1e-4
    )


def test_all_params_are_rounded_to_four_places():
    """India rounds to 4dp; the parity criterion is a 0.001 tolerance."""
    _score, params = calculate_quality(0.123456789, [0, 0, 137, 211])
    for key in ("det_score", "size_score", "aspect_score", "aspect_ratio"):
        rendered = f"{params[key]!r}"
        decimals = rendered.split(".")[-1] if "." in rendered else ""
        assert len(decimals) <= 4, f"{key} is not rounded to 4dp: {params[key]}"


def test_quality_params_key_set_matches_india_exactly():
    """A missing or extra key would break the stored quality_params JSON."""
    _score, params = calculate_quality(0.7, [0, 0, 100, 100])
    assert set(params) == {
        "det_score", "size_score", "aspect_score", "face_width", "face_height",
        "face_area", "aspect_ratio", "weights",
    }
    assert params["weights"] == {"det": 0.5, "size": 0.3, "aspect": 0.2}


class TestGenderLabel:
    """REFERENCE: return 'M' if face.gender == 1 else 'F'"""

    def test_one_is_male(self):
        assert gender_label(1) == "M"

    def test_zero_is_female(self):
        assert gender_label(0) == "F"

    def test_anything_else_is_female_not_none(self):
        """Faithful to India's semantics, which are not obviously intentional.

        Reproduced rather than 'fixed' because India's existing rows follow it.
        """
        assert gender_label(2) == "F"

    def test_none_stays_none(self):
        assert gender_label(None) is None


class TestAgeLabel:
    def test_float_is_truncated_to_int(self):
        assert age_label(27.8) == 27

    def test_none_stays_none(self):
        assert age_label(None) is None
