"""Safe structured request-completion logging for the gateway."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)

_STATE_KEY = "request_logging"
_MAX_REQUEST_ID_LENGTH = 128
_SAFE_FIELDS = {
    "cache_status",
    "circuit_outcome",
    "error_category",
    "fallback",
    "model_tier",
    "provider_outcome",
    "retry_outcome",
    "selected_model",
    "stream",
    "used_model",
}


def request_id(request: Request) -> str:
    """Return a bounded, printable request ID from the client or generate one."""
    candidate = request.headers.get("x-request-id", "").strip()
    if (
        candidate
        and len(candidate) <= _MAX_REQUEST_ID_LENGTH
        and all(32 <= ord(character) < 127 for character in candidate)
    ):
        return candidate
    return str(uuid4())


def initialize_request_logging(request: Request) -> None:
    """Attach safe, mutable request metadata to the request state."""
    request.state.request_id = request_id(request)
    setattr(
        request.state,
        _STATE_KEY,
        {
            "cache_status": "NOT_LOOKED_UP",
            "circuit_outcome": None,
            "error_category": None,
            "error_categories": [],
            "fallback": False,
            "model_tier": None,
            "provider_outcome": "not_attempted",
            "retry_outcome": None,
            "selected_model": None,
            "stream": False,
            "used_model": None,
        },
    )


def set_request_metadata(request: Request, **values: Any) -> None:
    """Set allow-listed metadata without ever accepting request contents."""
    metadata = getattr(request.state, _STATE_KEY, None)
    if metadata is None:
        initialize_request_logging(request)
        metadata = getattr(request.state, _STATE_KEY)
    for key, value in values.items():
        if key in _SAFE_FIELDS:
            metadata[key] = value


def add_error_category(request: Request, category: str) -> None:
    """Record a stable error category, never an exception or provider message."""
    metadata = getattr(request.state, _STATE_KEY, None)
    if metadata is None:
        initialize_request_logging(request)
        metadata = getattr(request.state, _STATE_KEY)
    if category not in metadata["error_categories"]:
        metadata["error_categories"].append(category)
    if metadata["error_category"] is None:
        metadata["error_category"] = category


def _metadata(request: Request) -> dict[str, Any]:
    metadata = getattr(request.state, _STATE_KEY, {})
    categories = metadata.get("error_categories", [])
    return {
        "event": "request_completed",
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "path": request.url.path,
        "status_code": 500,
        "duration_ms": 0.0,
        "cache_status": metadata.get("cache_status", "UNKNOWN"),
        "model_tier": metadata.get("model_tier"),
        "selected_model": metadata.get("selected_model"),
        "used_model": metadata.get("used_model"),
        "fallback": bool(metadata.get("fallback", False)),
        "stream": bool(metadata.get("stream", False)),
        "provider_outcome": metadata.get("provider_outcome", "not_attempted"),
        "error_category": metadata.get("error_category"),
        "error_categories": categories or None,
        "retry_outcome": metadata.get("retry_outcome"),
        "circuit_outcome": metadata.get("circuit_outcome"),
    }


def log_request_completion(request: Request, status_code: int, started: float) -> None:
    """Emit one JSON completion record containing only safe request metadata."""
    payload = _metadata(request)
    payload["status_code"] = status_code
    payload["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    logger.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def _status_error_category(status_code: int) -> str | None:
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "http_server_error"
    if status_code >= 400:
        return "http_client_error"
    return None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Add request IDs and log completion after normal or streaming responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        initialize_request_logging(request)
        started = time.perf_counter()
        emitted = False

        def emit(status_code: int) -> None:
            nonlocal emitted
            if emitted:
                return
            emitted = True
            metadata = getattr(request.state, _STATE_KEY)
            if metadata["error_category"] is None:
                category = _status_error_category(status_code)
                if category is not None:
                    add_error_category(request, category)
            log_request_completion(request, status_code, started)

        try:
            response = await call_next(request)
        except asyncio.CancelledError:
            set_request_metadata(request, provider_outcome="cancelled")
            add_error_category(request, "client_cancelled")
            emit(499)
            raise
        except Exception:
            add_error_category(request, "unhandled_error")
            emit(500)
            raise

        response.headers["X-Request-ID"] = request.state.request_id
        if not hasattr(response, "body_iterator"):
            emit(response.status_code)
            return response

        body_iterator = response.body_iterator

        async def iter_body() -> AsyncIterator[bytes]:
            try:
                async for chunk in body_iterator:
                    yield chunk
            except asyncio.CancelledError:
                set_request_metadata(request, provider_outcome="cancelled")
                add_error_category(request, "client_cancelled")
                emit(499)
                raise
            except Exception:
                add_error_category(request, "response_stream_error")
                emit(500)
                raise
            else:
                emit(response.status_code)

        response.body_iterator = iter_body()
        return response
