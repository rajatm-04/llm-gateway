import pytest

from gateway.resilience.retry import RetryConfig, retry_async, retry_delay


class ProviderError(Exception):
    def __init__(self, status_code: int):
        super().__init__(str(status_code))
        self.status_code = status_code


@pytest.mark.asyncio
async def test_retry_eventually_succeeds_after_transient_failures():
    attempts = 0
    delays = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderError(503)
        return "ok"

    async def sleep(delay):
        delays.append(delay)

    result = await retry_async(
        operation,
        RetryConfig(max_attempts=3, base_delay=1, jitter=0),
        sleep=sleep,
    )

    assert result == "ok"
    assert attempts == 3
    assert delays == [1, 2]


@pytest.mark.asyncio
async def test_retry_does_not_retry_permanent_failure():
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise ProviderError(400)

    with pytest.raises(ProviderError):
        await retry_async(operation, RetryConfig(max_attempts=3))

    assert attempts == 1


@pytest.mark.asyncio
async def test_retry_stops_after_max_attempts():
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise ProviderError(503)

    with pytest.raises(ProviderError):
        await retry_async(
            operation,
            RetryConfig(max_attempts=2),
            sleep=lambda _: _completed(),
        )

    assert attempts == 2


async def _completed():
    return None


def test_retry_delay_is_bounded_and_includes_jitter():
    config = RetryConfig(
        max_attempts=3,
        base_delay=1,
        max_delay=2,
        jitter=0.5,
    )

    assert retry_delay(1, config, random_value=lambda: 1) == 1.5
    assert retry_delay(3, config, random_value=lambda: 1) == 2


def test_retry_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        RetryConfig(max_attempts=0)
    with pytest.raises(ValueError):
        RetryConfig(max_delay=0, base_delay=1)
