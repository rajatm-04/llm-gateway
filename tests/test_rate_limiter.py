import pytest

from gateway.resilience.rate_limiter import RateLimiter


def test_rate_limiter_allows_capacity_then_returns_retry_after():
    now = 0.0
    limiter = RateLimiter(2, clock=lambda: now)

    assert limiter.allow("client")[0] is True
    assert limiter.allow("client")[0] is True

    allowed, retry_after = limiter.allow("client")

    assert allowed is False
    assert retry_after == 30


def test_rate_limiter_refills_tokens_over_time():
    now = 0.0
    limiter = RateLimiter(2, clock=lambda: now)

    limiter.allow("client")
    limiter.allow("client")
    now = 30.0

    assert limiter.allow("client")[0] is True


def test_rate_limiter_keeps_clients_in_separate_buckets():
    limiter = RateLimiter(1, clock=lambda: 0.0)

    assert limiter.allow("first")[0] is True
    assert limiter.allow("first")[0] is False
    assert limiter.allow("second")[0] is True


def test_rate_limiter_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="greater than zero"):
        RateLimiter(0)
