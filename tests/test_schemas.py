"""
Request/response contract tests (plan §31.2, §34.2).

Validation happens at the edge so a malformed request never reaches the GPU.
Rejecting early is what stops a buggy or hostile caller from consuming inference
time we are paying for by the second.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    FaceResponse,
    ImageRequest,
    ImageResult,
    ProcessRequest,
    ProcessResponse,
)

VALID_UUID = "3f2a91c4-7d5e-4b1a-9f8c-2e6d0a1b3c4d"
VALID_URL = (
    "https://s3.us-west-004.backblazeb2.com/bucket/large/ab/cd.jpg?X-Amz-Signature=x"
)


def image(picture_id: int = 5142338, url: str = VALID_URL) -> dict:
    return {"picture_id": picture_id, "image_url": url}


class TestProcessRequest:
    def test_accepts_a_well_formed_batch(self):
        request = ProcessRequest(
            batch_id=VALID_UUID, burst_cycle_id=1042, images=[image()]
        )
        assert request.picture_ids == [5142338]
        assert request.pairs == [(5142338, VALID_URL)]

    def test_batch_id_must_be_a_uuid(self):
        """The batch id is the idempotency key. A non-UUID suggests the client is
        generating them some other way, which would break replay."""
        with pytest.raises(ValidationError):
            ProcessRequest(batch_id="not-a-uuid", images=[image()])

    def test_empty_image_list_is_rejected(self):
        with pytest.raises(ValidationError):
            ProcessRequest(batch_id=VALID_UUID, images=[])

    def test_duplicate_picture_ids_are_rejected(self):
        """A duplicate would produce two results for one picture, and India's
        envelope validation treats that as fatal — so reject it at the source."""
        with pytest.raises(ValidationError, match="duplicate picture_id"):
            ProcessRequest(batch_id=VALID_UUID, images=[image(1), image(1)])

    def test_burst_cycle_id_is_optional(self):
        request = ProcessRequest(batch_id=VALID_UUID, images=[image()])
        assert request.burst_cycle_id is None


class TestImageRequest:
    def test_rejects_non_https(self):
        with pytest.raises(ValidationError):
            ImageRequest(picture_id=1, image_url="http://example.com/x.jpg")

    def test_rejects_non_positive_picture_id(self):
        with pytest.raises(ValidationError):
            ImageRequest(picture_id=0, image_url=VALID_URL)
        with pytest.raises(ValidationError):
            ImageRequest(picture_id=-5, image_url=VALID_URL)

    def test_rejects_an_overlong_url(self):
        with pytest.raises(ValidationError):
            ImageRequest(picture_id=1, image_url="https://x.com/" + "a" * 3000)


class TestContentSignature:
    def test_same_pictures_produce_the_same_signature(self):
        first = ProcessRequest(batch_id=VALID_UUID, images=[image(1), image(2)])
        second = ProcessRequest(batch_id=VALID_UUID, images=[image(2), image(1)])
        assert (
            first.content_signature() == second.content_signature()
        ), "signature must not depend on ordering"

    def test_signature_ignores_the_url(self):
        """A legitimate retry re-mints presigned URLs, so comparing URLs would
        produce false batch_id conflicts on every retry."""
        fresh_url = VALID_URL.replace("Signature=x", "Signature=totally-different")
        first = ProcessRequest(batch_id=VALID_UUID, images=[image(1)])
        second = ProcessRequest(batch_id=VALID_UUID, images=[image(1, fresh_url)])
        assert first.content_signature() == second.content_signature()

    def test_different_pictures_produce_a_different_signature(self):
        first = ProcessRequest(batch_id=VALID_UUID, images=[image(1)])
        second = ProcessRequest(batch_id=VALID_UUID, images=[image(2)])
        assert first.content_signature() != second.content_signature()


class TestFaceResponse:
    def test_bbox_must_have_exactly_four_values(self):
        base = dict(
            det_score=0.9,
            quality_score=0.7,
            quality_params={},
            embedding=[0.0] * 512,
        )
        with pytest.raises(ValidationError):
            FaceResponse(bbox=[1, 2, 3], **base)
        with pytest.raises(ValidationError):
            FaceResponse(bbox=[1, 2, 3, 4, 5], **base)

    def test_accepts_a_512_dimensional_embedding(self):
        face = FaceResponse(
            bbox=[10, 20, 110, 140],
            det_score=0.9,
            quality_score=0.7,
            quality_params={"det_score": 0.9},
            embedding=[0.1] * 512,
        )
        assert len(face.embedding) == 512


class TestProcessResponse:
    def test_carries_the_provenance_fields_india_asserts(self):
        """model_name / embedding_dim / image_version in the envelope let India
        detect a Pod running an unexpected image WITHOUT re-reading /health."""
        response = ProcessResponse(
            batch_id=VALID_UUID,
            image_version="1.0.0",
            model_name="buffalo_l",
            embedding_dim=512,
            processed_count=1,
            success_count=1,
            failed_count=0,
            total_faces=2,
            duration_ms=1800,
            results=[ImageResult(picture_id=1, success=True)],
        )
        assert response.model_name == "buffalo_l"
        assert response.embedding_dim == 512
        assert response.image_version == "1.0.0"
        assert response.cached is False

    def test_per_image_failure_is_representable_inside_a_success_response(self):
        """Per-image failure is the NORMAL error channel: a 200 carrying
        success=False, not an exception."""
        result = ImageResult(
            picture_id=42,
            success=False,
            error_code="download_failed",
            error="HTTP 403 fetching object (presigned URL may have expired)",
        )
        assert result.success is False
        assert result.faces == []
        assert result.error_code == "download_failed"
