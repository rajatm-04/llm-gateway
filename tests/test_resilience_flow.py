from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from gateway.api.v1 import chat
from gateway.main import RateLimitMiddleware
from gateway.models.schemas import ChatCompletionRequest, ChatMessage
from gateway.providers.base import LLMResponse
from gateway.resilience.circuit_breaker import CircuitBreaker, CircuitState
from gateway.resilience.rate_limiter import RateLimiter
from gateway.resilience.retry import RetryConfig


class ProviderError(Exception):
    def __init__(self, status_code=503):
        super().__init__(str(status_code))
        self.status_code = status_code


class FakeCache:
    def __init__(self, payload=None):
        self.payload = payload
        self.lookup_count = 0
        self.store_count = 0

    async def lookup(self, query):
        self.lookup_count += 1
        return self.payload

    async def store(self, **kwargs):
        self.store_count += 1


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def generate(self, messages, *, model=None, temperature=None, max_tokens=None):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    async def generate_stream(
        self, messages, *, model=None, temperature=None, max_tokens=None
    ):
        yield "streamed"


def request_for(cache, headers=None):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(cache=cache)),
        headers=headers or {},
    )


def chat_request(model="gpt-test", stream=False):
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content="hello")],
        stream=stream,
    )


@pytest.mark.asyncio
async def test_transient_failures_retry_before_breaker_records_failure(monkeypatch):
    provider = FakeProvider(
        [
            ProviderError(),
            LLMResponse(content="ok", model="gpt-test"),
        ]
    )
    breaker = CircuitBreaker(failure_threshold=1)
    cache = FakeCache()
    monkeypatch.setattr(chat, "get_provider", lambda _: (provider, breaker))
    monkeypatch.setattr(
        chat,
        "retry_config",
        RetryConfig(max_attempts=2, base_delay=0, jitter=0),
    )

    response = await chat.chat_completions(
        chat_request(),
        request_for(cache),
    )

    assert response.status_code == 200
    assert provider.calls == 2
    assert breaker.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_exhausted_retries_open_breaker_and_return_503(monkeypatch):
    provider = FakeProvider(
        [        ProviderError(), ProviderError()]
    )
    breaker = CircuitBreaker(failure_threshold=1)
    cache = FakeCache()
    monkeypatch.setattr(chat, "get_provider", lambda _: (provider, breaker))
    monkeypatch.setattr(
        chat,
        "retry_config",
        RetryConfig(max_attempts=2, base_delay=0, jitter=0),
    )

    with pytest.raises(ProviderError):
        await chat.chat_completions(chat_request(model="ollama"), request_for(cache))

    assert provider.calls == 2
    assert breaker.state is CircuitState.OPEN

    with pytest.raises(HTTPException) as error:
        await chat.chat_completions(
            chat_request(model="ollama"),
            request_for(cache),
        )
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_premium_circuit_requires_confirmation_then_falls_back(monkeypatch):
    premium = FakeProvider([ProviderError()])
    local = FakeProvider([LLMResponse(content="local", model="phi4-mini")])
    premium_breaker = CircuitBreaker(failure_threshold=1)
    local_breaker = CircuitBreaker()
    cache = FakeCache()
    monkeypatch.setattr(
        chat,
        "get_provider",
        lambda _: (premium, premium_breaker),
    )
    monkeypatch.setattr(chat, "ollama_provider", local)
    monkeypatch.setattr(chat, "ollama_breaker", local_breaker)
    monkeypatch.setattr(
        chat,
        "retry_config",
        RetryConfig(max_attempts=1, base_delay=0, jitter=0),
    )

    with pytest.raises(ProviderError):
        await chat.chat_completions(chat_request(), request_for(cache))
    assert premium_breaker.state is CircuitState.OPEN

    denied = await chat.chat_completions(
        chat_request(),
        request_for(cache),
    )
    assert denied.status_code == 503
    assert denied.body.find(b"requires_confirmation") >= 0

    allowed = await chat.chat_completions(
        chat_request(),
        request_for(cache, {"x-allow-fallback": "true"}),
    )
    assert allowed.headers["X-Fallback"] == "true"
    assert allowed.headers["X-Model-Used"] == "phi4-mini"
    assert local.calls == 1


@pytest.mark.asyncio
async def test_cache_hit_bypasses_provider_and_does_not_store():
    provider = FakeProvider([LLMResponse(content="unexpected", model="gpt-test")])
    cache = FakeCache(
        {
            "response": "cached",
            "model": "gpt-test",
            "finish_reason": "stop",
        }
    )
    response = await chat.chat_completions(
        chat_request(),
        request_for(cache),
    )

    assert response.status_code == 200
    assert provider.calls == 0
    assert cache.lookup_count == 1
    assert cache.store_count == 0


def test_rate_limit_middleware_returns_429_after_exhaustion(monkeypatch):
    from gateway import main

    monkeypatch.setattr(main, "rate_limiter", RateLimiter(2))
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/test")
    async def test_route():
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/test").status_code == 200
        assert client.get("/test").status_code == 200
        limited = client.get("/test")

    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1
