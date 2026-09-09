"""Circuit breaker for protecting unhealthy providers."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from enum import StrEnum
from typing import TypeVar


class CircuitState(StrEnum):
    """States in the circuit-breaker lifecycle."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected by an open circuit."""


T = TypeVar("T")


class CircuitBreaker:
    """Stop calls to a provider after repeated failures.

    The breaker is safe for concurrent async requests. Calls are performed
    outside the lock; the lock only protects state transitions and counters.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be greater than zero")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be greater than zero")
        if half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be greater than zero")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Return the consecutive failure count."""
        return self._failure_count

    async def execute(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run an operation if permitted, recording success or failure."""
        is_probe = await self._before_call()

        try:
            result = await operation()
        except Exception:
            await self._record_failure()
            raise
        else:
            await self._record_success(is_probe)
            return result

    async def execute_stream(
        self,
        operation: Callable[[], AsyncIterator[T]],
    ) -> AsyncIterator[T]:
        """Run and protect an async streaming operation."""
        is_probe = await self._before_call()

        try:
            async with aclosing(operation()) as stream:
                async for item in stream:
                    yield item
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._record_failure()
            raise
        else:
            await self._record_success(is_probe)

    async def _before_call(self) -> bool:
        """Reserve a normal call or a half-open probe."""
        async with self._lock:
            now = self._clock()

            if self._state is CircuitState.OPEN:
                if self._opened_at is None or (
                    now - self._opened_at < self.recovery_timeout
                ):
                    raise CircuitOpenError("Circuit is open")

                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0

            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError("Circuit is half-open")

                self._half_open_calls += 1
                return True

            return False

    async def _record_success(self, is_probe: bool) -> None:
        """Close a recovered circuit or clear a normal failure count."""
        async with self._lock:
            if is_probe or self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._opened_at = None
                self._half_open_calls = 0
            else:
                self._failure_count = 0

    async def _record_failure(self) -> None:
        """Count a failure and open the circuit when the threshold is reached."""
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._half_open_calls = 0
                return

            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
