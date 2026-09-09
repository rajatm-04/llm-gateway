"""Per-client token-bucket rate limiting."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse


@dataclass
class _Bucket:
    """Mutable state for one rate-limit identity."""

    tokens: float
    updated_at: float


class RateLimiter:
    """Limit requests per identity with a continuously refilling token bucket.

    The bucket capacity is the configured requests-per-minute value. Tokens
    refill continuously, so a client can burst up to one full bucket and then
    gradually recover rather than waiting for a fixed one-minute reset.
    """

    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than zero")

        self.capacity = float(requests_per_minute)
        self.refill_rate = self.capacity / 60.0
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, identity: str) -> tuple[bool, int]:
        """Consume one token and return ``(allowed, retry_after_seconds)``."""
        now = self._clock()
        bucket = self._buckets.get(identity)

        if bucket is None:
            bucket = _Bucket(tokens=self.capacity, updated_at=now)
            self._buckets[identity] = bucket
        else:
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(
                self.capacity,
                bucket.tokens + elapsed * self.refill_rate,
            )
            bucket.updated_at = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0

        retry_after = math.ceil((1.0 - bucket.tokens) / self.refill_rate)
        return False, max(1, retry_after)


def rate_limit_identity(request: Request) -> str:
    """Return a stable client identity from auth header or peer address."""
    authorization = request.headers.get("authorization")
    if authorization:
        return f"authorization:{authorization}"

    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def rate_limit_response(retry_after: int) -> JSONResponse:
    """Build the standard response for an exhausted rate-limit bucket."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"},
        headers={"Retry-After": str(retry_after)},
    )
