"""
Idempotent batch-result cache.

The problem it solves is the most common failure in this system: India POSTs a
batch, the worker completes it, and the response is lost on the way back. India
cannot tell that from "the worker never received it", so it must retry — and
without a cache, that retry re-downloads 32 images and re-runs inference, paying
twice for work already done.

With the cache, a retry carrying the SAME ``batch_id`` gets the stored response
back immediately with ``cached: true``. That turns the expensive-and-ambiguous
case into a cheap-and-certain one, and is why the client is written to reuse the
batch id rather than mint a new one on retry.

TWO DESIGN CHOICES WORTH NOTING:

  * A reused ``batch_id`` with a DIFFERENT set of pictures returns 409 rather
    than serving the cached answer. That combination cannot occur in correct
    operation, so silently serving stale results would hide an India-side bug and
    could mark the wrong pictures complete.

  * The cache is in memory only, sized by count. The Pod is disposable, so
    persistence would be pointless; and the authoritative copy of every result is
    already in India's ``insightface_job.gpu_result``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class BatchIdConflict(ValueError):
    """A known batch_id was reused with a different set of pictures."""

    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        super().__init__(
            f"batch_id {batch_id} was already processed with a different set of "
            f"pictures. This indicates a client bug; refusing to serve the "
            f"cached response, because it describes different work."
        )


@dataclass
class CacheEntry:
    signature: str
    payload: dict
    stored_at: float
    hits: int = 0


class ResultCache:
    """Thread-safe bounded LRU of batch responses."""

    def __init__(self, max_entries: int = 64):
        self.max_entries = max(1, max_entries)
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.conflicts = 0

    def get(self, batch_id: str, signature: str) -> Optional[dict]:
        """Return the cached payload for an identical batch.

        Raises:
            BatchIdConflict: the id is known but the contents differ.
        """
        with self._lock:
            entry = self._entries.get(batch_id)
            if entry is None:
                self.misses += 1
                return None

            if entry.signature != signature:
                self.conflicts += 1
                logger.error(
                    "batch_id %s reused with different contents (stored "
                    "signature %s..., requested %s...)",
                    batch_id[:8],
                    entry.signature[:8],
                    signature[:8],
                )
                raise BatchIdConflict(batch_id)

            # Refresh LRU position and hand back a copy so a caller cannot
            # mutate the cached payload.
            self._entries.move_to_end(batch_id)
            entry.hits += 1
            self.hits += 1
            age = time.time() - entry.stored_at
            logger.info(
                "Idempotent replay of batch %s (age %.1fs, hit #%d): returning "
                "the cached response with no re-download and no re-inference",
                batch_id[:8],
                age,
                entry.hits,
            )
            payload = dict(entry.payload)
            payload["cached"] = True
            return payload

    def put(self, batch_id: str, signature: str, payload: dict) -> None:
        with self._lock:
            self._entries[batch_id] = CacheEntry(
                signature=signature,
                # Store with cached=False so the first (real) response is
                # correctly labelled and only replays are marked.
                payload={**payload, "cached": False},
                stored_at=time.time(),
            )
            self._entries.move_to_end(batch_id)
            while len(self._entries) > self.max_entries:
                evicted, _ = self._entries.popitem(last=False)
                logger.debug("Evicted batch %s from the result cache", evicted[:8])

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "conflicts": self.conflicts,
            }
