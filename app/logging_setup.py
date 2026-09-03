"""
Structured JSON logging with correlation ids (plan §27).

Every line carries whichever of these are known:

    burst_cycle_id -> pod_id -> batch_id -> picture_id

That chain is what makes a production failure traceable: India records the same
ids in ``gpu_burst_cycle`` / ``gpu_pod`` / ``gpu_batch`` / ``insightface_job``, so
a Pod's container log (retrievable via ``GET /v2/pods/{id}/logs``) joins to the
database records by foreign key rather than by grepping timestamps.

Output goes to stdout only. Pods are disposable — writing log files inside one
would discard them on termination.

REDACTION IS ENFORCED HERE, not left to call sites. Presigned URLs are bearer
credentials for the object, and the worker API key is a credential; a formatter
that strips them means a careless log statement somewhere else cannot leak one.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

#: Per-request correlation context, set by the API middleware.
_cycle_id: ContextVar[Optional[str]] = ContextVar("burst_cycle_id", default=None)
_batch_id: ContextVar[Optional[str]] = ContextVar("batch_id", default=None)
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

#: Query strings on the storage host carry the SigV4 signature.
_SIGNED_URL = re.compile(r"(https://[^\s\"']+)\?[^\s\"']*")
#: AWS SigV4 query parameters, in case a URL is assembled in pieces.
_SIG_PARAM = re.compile(
    r"(X-Amz-(?:Signature|Credential|Security-Token)=)[^&\s\"']+", re.IGNORECASE
)
#: Bearer tokens.
_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)


def set_context(
    *,
    burst_cycle_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    if burst_cycle_id is not None:
        _cycle_id.set(burst_cycle_id)
    if batch_id is not None:
        _batch_id.set(batch_id)
    if request_id is not None:
        _request_id.set(request_id)


def clear_context() -> None:
    _cycle_id.set(None)
    _batch_id.set(None)
    _request_id.set(None)


def redact(text: str) -> str:
    """Strip credentials from a log message."""
    text = _SIGNED_URL.sub(r"\1?<redacted>", text)
    text = _SIG_PARAM.sub(r"\1<redacted>", text)
    text = _BEARER.sub(r"\1<redacted>", text)
    return text


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the correlation chain attached."""

    def __init__(self, static_fields: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.static_fields = static_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "component": record.name,
            "event": redact(record.getMessage()),
        }
        payload.update(self.static_fields)

        for key, var in (
            ("burst_cycle_id", _cycle_id),
            ("batch_id", _batch_id),
            ("request_id", _request_id),
        ):
            value = var.get()
            if value is not None:
                payload[key] = value

        # Allow call sites to attach structured fields via
        # logger.info(..., extra={"picture_id": 123}).
        for key, value in record.__dict__.items():
            if key in (
                "picture_id",
                "image_count",
                "duration_ms",
                "faces",
                "http_status",
                "error_code",
            ):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))

        return json.dumps(payload, default=str, separators=(",", ":"))


def configure(
    level: str = "INFO",
    *,
    image_version: str = "unknown",
    burst_cycle_id: Optional[str] = None,
) -> None:
    """Install the JSON formatter on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(static_fields={"image_version": image_version}))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # uvicorn installs its own handlers; route them through ours so every line
    # in the container log has the same shape.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        target = logging.getLogger(name)
        target.handlers.clear()
        target.propagate = True

    # InsightFace is chatty at INFO during model load.
    logging.getLogger("insightface").setLevel(logging.WARNING)

    if burst_cycle_id:
        set_context(burst_cycle_id=burst_cycle_id)
