import pytest

from gateway.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


async def failing_operation() -> None:
    raise RuntimeError("provider unavailable")


async def successful_operation() -> str:
    return "ok"


async def streaming_operation():
    for item in ("a", "b"):
        yield item


@pytest.mark.asyncio
async def test_breaker_opens_after_failure_threshold():
    breaker = CircuitBreaker(failure_threshold=2)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.execute(failing_operation)

    assert breaker.state is CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        await breaker.execute(successful_operation)


@pytest.mark.asyncio
async def test_breaker_recovers_after_timeout():
    now = 0.0
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=10,
        clock=lambda: now,
    )

    with pytest.raises(RuntimeError):
        await breaker.execute(failing_operation)

    now = 10
    assert await breaker.execute(successful_operation) == "ok"
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0


@pytest.mark.asyncio
async def test_failed_half_open_probe_reopens_breaker():
    now = 0.0
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=10,
        clock=lambda: now,
    )

    with pytest.raises(RuntimeError):
        await breaker.execute(failing_operation)

    now = 10
    with pytest.raises(RuntimeError):
        await breaker.execute(failing_operation)

    assert breaker.state is CircuitState.OPEN


@pytest.mark.asyncio
async def test_success_resets_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=2)

    with pytest.raises(RuntimeError):
        await breaker.execute(failing_operation)

    assert await breaker.execute(successful_operation) == "ok"
    assert breaker.failure_count == 0
    assert breaker.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_streaming_operation_is_protected():
    breaker = CircuitBreaker(failure_threshold=1)
    items = [
        item
        async for item in breaker.execute_stream(streaming_operation)
    ]

    assert items == ["a", "b"]
    assert breaker.state is CircuitState.CLOSED


def test_breaker_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(recovery_timeout=0)
    with pytest.raises(ValueError):
        CircuitBreaker(half_open_max_calls=0)
