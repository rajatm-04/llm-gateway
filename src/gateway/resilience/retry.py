"""Bounded retries for transient provider failures."""

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from time import time

import httpx
from openai import APIConnectionError, APITimeoutError

TransientStatusCodes = frozenset({429, 500, 502, 503})


@dataclass(frozen=True)
class RetryConfig:
    """Retry timing and attempt limits."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 8.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if self.base_delay < 0:
            raise ValueError("base_delay cannot be negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be at least base_delay")
        if self.jitter < 0:
            raise ValueError("jitter cannot be negative")


def response_status_code(error: Exception) -> int | None:
    """Extract an HTTP status from common HTTP client exceptions."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def is_retryable(error: Exception) -> bool:
    """Return whether an exception represents a transient provider failure."""
    status_code = response_status_code(error)
    if status_code is not None:
        return status_code in TransientStatusCodes

    return isinstance(
        error,
        (
            APIConnectionError,
            APITimeoutError,
            httpx.TimeoutException,
            httpx.NetworkError,
        ),
    )


def retry_after_seconds(error: Exception) -> float | None:
    """Read a provider Retry-After value, supporting seconds or HTTP dates."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    if not value:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            target = parsedate_to_datetime(value).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0.0, target - time())


def retry_delay(
    retry_number: int,
    config: RetryConfig,
    *,
    random_value: Callable[[], float] = random.random,
) -> float:
    """Calculate bounded exponential backoff with positive jitter."""
    exponential = min(
        config.max_delay,
        config.base_delay * (2 ** (retry_number - 1)),
    )
    return min(
        config.max_delay,
        exponential + random_value() * config.jitter,
    )


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    config: RetryConfig,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Retry an async operation when it fails transiently."""
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await operation()
        except Exception as error:
            if attempt == config.max_attempts or not is_retryable(error):
                raise

            delay = retry_after_seconds(error)
            if delay is None:
                delay = retry_delay(attempt, config)
            await sleep(delay)

    raise RuntimeError("Retry loop exited unexpectedly")


async def retry_stream[T](
    operation: Callable[[], AsyncIterator[T]],
    config: RetryConfig,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[T]:
    """Retry a stream only when it fails before yielding its first item."""
    for attempt in range(1, config.max_attempts + 1):
        yielded = False
        try:
            async with aclosing(operation()) as stream:
                async for item in stream:
                    yielded = True
                    yield item
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if (
                yielded
                or attempt == config.max_attempts
                or not is_retryable(error)
            ):
                raise

            delay = retry_after_seconds(error)
            if delay is None:
                delay = retry_delay(attempt, config)
            await sleep(delay)
        else:
            return

    raise RuntimeError("Retry loop exited unexpectedly")
