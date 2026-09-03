"""
Result-cache idempotency tests (plan §31.2, §34.2).

The cache turns the system's most common failure — a completed batch whose
response was lost on the way back — from "pay for it twice" into "free replay".
These tests pin that behaviour, and pin the one case where a replay must be
REFUSED rather than served.
"""

from __future__ import annotations

import pytest

from app.result_cache import BatchIdConflict, ResultCache

BATCH = "3f2a91c4-7d5e-4b1a-9f8c-2e6d0a1b3c4d"
SIG_A = "a" * 64
SIG_B = "b" * 64


def payload(faces: int = 3) -> dict:
    return {"batch_id": BATCH, "total_faces": faces, "results": [], "cached": False}


class TestReplay:
    def test_miss_returns_none(self):
        cache = ResultCache(4)
        assert cache.get(BATCH, SIG_A) is None
        assert cache.stats()["misses"] == 1

    def test_identical_replay_returns_the_stored_payload(self):
        cache = ResultCache(4)
        cache.put(BATCH, SIG_A, payload())
        replay = cache.get(BATCH, SIG_A)
        assert replay is not None
        assert replay["total_faces"] == 3

    def test_replay_is_flagged_cached_true(self):
        """India uses this flag to distinguish a real run from a replay, which is
        how the metrics separate paid work from free work."""
        cache = ResultCache(4)
        cache.put(BATCH, SIG_A, payload())
        assert cache.get(BATCH, SIG_A)["cached"] is True

    def test_first_store_is_flagged_cached_false(self):
        cache = ResultCache(4)
        cache.put(BATCH, SIG_A, payload())
        # Inspect the stored entry rather than a replay.
        assert cache._entries[BATCH].payload["cached"] is False  # noqa: SLF001

    def test_replay_does_not_mutate_the_cached_entry(self):
        cache = ResultCache(4)
        cache.put(BATCH, SIG_A, payload())
        first = cache.get(BATCH, SIG_A)
        first["total_faces"] = 999
        second = cache.get(BATCH, SIG_A)
        assert (
            second["total_faces"] == 3
        ), "a caller mutated the cached payload; get() must return a copy"


class TestConflict:
    def test_reused_id_with_different_contents_raises(self):
        """This combination cannot occur in correct operation.

        Serving the cached answer would mark the WRONG pictures complete, so a
        409 is the only safe response.
        """
        cache = ResultCache(4)
        cache.put(BATCH, SIG_A, payload())
        with pytest.raises(BatchIdConflict):
            cache.get(BATCH, SIG_B)

    def test_conflicts_are_counted(self):
        cache = ResultCache(4)
        cache.put(BATCH, SIG_A, payload())
        with pytest.raises(BatchIdConflict):
            cache.get(BATCH, SIG_B)
        assert cache.stats()["conflicts"] == 1

    def test_conflict_message_does_not_leak_the_payload(self):
        cache = ResultCache(4)
        cache.put(BATCH, SIG_A, payload())
        try:
            cache.get(BATCH, SIG_B)
        except BatchIdConflict as exc:
            assert BATCH in str(exc)
            assert "results" not in str(exc)


class TestEviction:
    def test_respects_the_size_limit(self):
        cache = ResultCache(3)
        for index in range(5):
            cache.put(f"batch-{index}", SIG_A, payload(index))
        assert len(cache) == 3

    def test_evicts_least_recently_used(self):
        cache = ResultCache(3)
        for index in range(3):
            cache.put(f"batch-{index}", SIG_A, payload(index))

        # Touch batch-0 so batch-1 becomes the LRU.
        assert cache.get("batch-0", SIG_A) is not None
        cache.put("batch-3", SIG_A, payload(3))

        assert cache.get("batch-0", SIG_A) is not None
        assert cache.get("batch-1", SIG_A) is None
        assert cache.get("batch-3", SIG_A) is not None

    def test_zero_max_entries_is_clamped_to_one(self):
        """A cache of size 0 would silently disable idempotency, so the floor is
        1 rather than an error the caller might not notice."""
        cache = ResultCache(0)
        cache.put(BATCH, SIG_A, payload())
        assert cache.get(BATCH, SIG_A) is not None


class TestStats:
    def test_tracks_hits_and_misses(self):
        cache = ResultCache(4)
        cache.get(BATCH, SIG_A)
        cache.put(BATCH, SIG_A, payload())
        cache.get(BATCH, SIG_A)
        cache.get(BATCH, SIG_A)

        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 2
        assert stats["entries"] == 1
