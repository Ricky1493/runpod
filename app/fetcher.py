"""
Presigned-URL fetching, hardened against SSRF.

The threat is concrete. ``/v1/process`` accepts URLs and fetches them from inside
a Pod on someone else's network. Without restriction, anyone who got past the
bearer token could use this worker as a proxy to reach cloud metadata endpoints,
internal services, or arbitrary hosts. So:

  * HOST ALLOWLIST, and an EMPTY ALLOWLIST REJECTS EVERYTHING. Failing closed is
    the point: a misconfigured worker that fetches nothing is a visible outage,
    whereas one that fetches anything is an invisible hole.
  * HTTPS only.
  * NO REDIRECTS FOLLOWED. A redirect is the classic allowlist bypass — the first
    hop passes the check and the second goes wherever it likes.
  * NO CREDENTIALS IN THE URL.
  * SIZE CAP while streaming, so a hostile or broken origin cannot exhaust memory
    by sending an endless body.
  * PER-REQUEST TIMEOUT, below the batch deadline.

URLS ARE NEVER LOGGED. A presigned URL is a bearer credential for the object; its
query string contains the signature.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class UrlRejected(ValueError):
    """The URL failed validation and was never fetched."""


class DownloadFailed(RuntimeError):
    """The fetch was attempted and failed."""

    def __init__(self, message: str, *, error_code: str = "download_failed"):
        self.error_code = error_code
        super().__init__(message)


class DownloadTooLarge(DownloadFailed):
    def __init__(self, message: str):
        super().__init__(message, error_code="too_large")


@dataclass
class FetchResult:
    picture_id: int
    content: Optional[bytes] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.content is not None


def redact(url: str) -> str:
    """Render a URL safe to log: scheme, host, path, no query string."""
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?<redacted>"
    except Exception:
        return "<unparseable url>"


def validate_url(url: str, allowed_hosts: Sequence[str]) -> str:
    """Validate a URL before any network call. Returns the host.

    Raises UrlRejected with a reason. The reason is safe to return to the caller
    (it names the host, never the signature).
    """
    if not url or len(url) > 2048:
        raise UrlRejected("url is empty or longer than 2048 characters")

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise UrlRejected(f"url could not be parsed: {exc}") from exc

    if parsed.scheme != "https":
        raise UrlRejected(f"scheme {parsed.scheme!r} is not https")

    if parsed.username or parsed.password:
        raise UrlRejected("url must not contain embedded credentials")

    host = (parsed.hostname or "").lower()
    if not host:
        raise UrlRejected("url has no host")

    if not allowed_hosts:
        raise UrlRejected(
            "no allowed hosts are configured, so every URL is rejected "
            "(ALLOWED_URL_HOSTS is unset)"
        )

    if not _host_allowed(host, allowed_hosts):
        raise UrlRejected(f"host {host!r} is not in the allowlist")

    _reject_private_address(host)
    return host


def _host_allowed(host: str, allowed_hosts: Sequence[str]) -> bool:
    """Exact match, or a subdomain of an allowlisted host.

    Subdomain matching is anchored on a leading dot so that allowlisting
    ``example.com`` cannot be satisfied by ``notexample.com``.
    """
    for allowed in allowed_hosts:
        candidate = allowed.strip().lower()
        if not candidate:
            continue
        if host == candidate or host.endswith(f".{candidate}"):
            return True
    return False


def _reject_private_address(host: str) -> None:
    """Refuse hosts that resolve to a private, loopback, or link-local address.

    Defence in depth behind the allowlist: it also catches an allowlisted name
    that has been repointed at an internal address (DNS rebinding).
    """
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlRejected(f"host {host!r} could not be resolved: {exc}") from exc

    for info in infos:
        address = info[4][0]
        try:
            parsed_ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if (
            parsed_ip.is_private
            or parsed_ip.is_loopback
            or parsed_ip.is_link_local
            or parsed_ip.is_reserved
            or parsed_ip.is_multicast
        ):
            raise UrlRejected(
                f"host {host!r} resolves to the non-public address {address}"
            )


class ImageFetcher:
    """Bounded-concurrency HTTPS fetcher for presigned object URLs."""

    def __init__(
        self,
        *,
        allowed_hosts: Sequence[str],
        concurrency: int = 16,
        timeout_seconds: int = 20,
        max_bytes: int = 33_554_432,
    ):
        self.allowed_hosts = list(allowed_hosts)
        self.concurrency = max(1, concurrency)
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "ImageFetcher":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                self.timeout_seconds, connect=min(10, self.timeout_seconds)
            ),
            # NOT negotiable: following redirects would defeat the allowlist.
            follow_redirects=False,
            limits=httpx.Limits(
                max_connections=self.concurrency,
                max_keepalive_connections=self.concurrency,
            ),
            headers={"User-Agent": "face-gpu-worker/1.0"},
        )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_many(
        self, items: Sequence[Tuple[int, str]], deadline: Optional[float] = None
    ) -> Dict[int, FetchResult]:
        """Fetch a batch concurrently.

        Args:
            items: (picture_id, url) pairs.
            deadline: monotonic time after which remaining fetches are abandoned.

        Returns:
            picture_id -> FetchResult. Every requested id is present; a failure is
            a FetchResult with an error_code, never a missing entry, so the caller
            can always account for every picture.
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(picture_id: int, url: str) -> FetchResult:
            async with semaphore:
                if deadline is not None and asyncio.get_event_loop().time() > deadline:
                    return FetchResult(
                        picture_id=picture_id,
                        error="batch deadline elapsed before this image was fetched",
                        error_code="deadline",
                    )
                return await self.fetch_one(picture_id, url)

        results = await asyncio.gather(
            *(one(picture_id, url) for picture_id, url in items),
            return_exceptions=True,
        )

        out: Dict[int, FetchResult] = {}
        for (picture_id, _url), result in zip(items, results):
            if isinstance(result, BaseException):
                out[picture_id] = FetchResult(
                    picture_id=picture_id,
                    error=f"unexpected fetch error: {result}",
                    error_code="internal",
                )
            else:
                out[picture_id] = result
        return out

    async def fetch_one(self, picture_id: int, url: str) -> FetchResult:
        """Fetch one object, streaming with a hard size cap."""
        loop = asyncio.get_event_loop()
        started = loop.time()

        try:
            validate_url(url, self.allowed_hosts)
        except UrlRejected as exc:
            logger.warning(
                "Rejected URL for picture %d: %s (%s)",
                picture_id,
                exc,
                redact(url),
            )
            return FetchResult(
                picture_id=picture_id,
                error=str(exc),
                error_code="url_rejected",
                duration_ms=int((loop.time() - started) * 1000),
            )

        if self._client is None:
            raise RuntimeError("ImageFetcher must be used as an async context manager")

        try:
            chunks: List[bytes] = []
            total = 0
            async with self._client.stream("GET", url) as response:
                if response.status_code >= 400:
                    detail = (
                        "presigned URL may have expired"
                        if response.status_code in (401, 403)
                        else (
                            "object may not exist"
                            if response.status_code == 404
                            else "origin error"
                        )
                    )
                    return FetchResult(
                        picture_id=picture_id,
                        error=f"HTTP {response.status_code} fetching object "
                        f"({detail})",
                        error_code="download_failed",
                        duration_ms=int((loop.time() - started) * 1000),
                    )

                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > self.max_bytes:
                    return FetchResult(
                        picture_id=picture_id,
                        error=f"object is {declared} bytes, over the "
                        f"{self.max_bytes} limit",
                        error_code="too_large",
                        duration_ms=int((loop.time() - started) * 1000),
                    )

                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self.max_bytes:
                        # Abandon mid-stream: a declared length can lie.
                        return FetchResult(
                            picture_id=picture_id,
                            error=f"object exceeded the {self.max_bytes} byte "
                            f"limit while streaming",
                            error_code="too_large",
                            duration_ms=int((loop.time() - started) * 1000),
                        )
                    chunks.append(chunk)

            return FetchResult(
                picture_id=picture_id,
                content=b"".join(chunks),
                duration_ms=int((loop.time() - started) * 1000),
            )

        except httpx.TimeoutException:
            return FetchResult(
                picture_id=picture_id,
                error=f"timed out after {self.timeout_seconds}s",
                error_code="download_timeout",
                duration_ms=int((loop.time() - started) * 1000),
            )
        except httpx.HTTPError as exc:
            return FetchResult(
                picture_id=picture_id,
                error=f"transport error: {type(exc).__name__}: {exc}",
                error_code="download_failed",
                duration_ms=int((loop.time() - started) * 1000),
            )
