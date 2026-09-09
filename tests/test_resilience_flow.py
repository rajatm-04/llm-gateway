import pytest
from conftest import FakeCache, FakeRegistry
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.main import RateLimitMiddleware, create_app
from gateway.models.schemas import ChatCompletionRequest, ChatMessage
from gateway.providers.base import LLMResponse
from gateway.resilience.circuit_breaker import CircuitState
from gateway.resilience.rate_limiter import RateLimiter


class ProviderError(Exception):
    def __init__(self, status_code=503):
        super().__init__(str(status_code))
        self.status_code = status_code


class SequenceProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def generate(self, messages, *, model=None, temperature=None, max_tokens=None):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def chat_request(model=None):
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content="hello")],
    )


def resilience_config(config, **updates):
    return config.model_copy(
        update={
            "retry_base_delay": 0,
            "retry_jitter": 0,
            **updates,
        }
    )


def test_transient_failures_retry_before_breaker_records_failure(config):
    provider = SequenceProvider(
        [ProviderError(), LLMResponse(content="ok", model="phi4-mini")]
    )
    providers = FakeRegistry()
    providers.local = provider
    config = resilience_config(
        config,
        retry_max_attempts=2,
        circuit_failure_threshold=1,
    )
    app = create_app(config=config, cache=FakeCache(), providers=providers)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=chat_request().model_dump(),
            headers={"X-Model-Tier": "local"},
        )

    assert response.status_code == 200
    assert provider.calls == 2
    assert app.state.breakers["local"].state is CircuitState.CLOSED


def test_exhausted_retries_open_breaker_and_return_503(config):
    provider = SequenceProvider([ProviderError(), ProviderError()])
    providers = FakeRegistry()
    providers.local = provider
    config = resilience_config(
        config,
        retry_max_attempts=2,
        circuit_failure_threshold=1,
    )
    app = create_app(config=config, cache=FakeCache(), providers=providers)

    with TestClient(app, raise_server_exceptions=True) as client:
        with pytest.raises(ProviderError):
            client.post(
                "/v1/chat/completions",
                json=chat_request().model_dump(),
                headers={"X-Model-Tier": "local"},
            )

        assert provider.calls == 2
        assert app.state.breakers["local"].state is CircuitState.OPEN

        response = client.post(
            "/v1/chat/completions",
            json=chat_request().model_dump(),
            headers={"X-Model-Tier": "local"},
        )

    assert response.status_code == 503


def test_premium_circuit_requires_confirmation_then_falls_back(config):
    premium = SequenceProvider([ProviderError()])
    local = SequenceProvider([LLMResponse(content="local", model="phi4-mini")])
    providers = FakeRegistry()
    providers.premium = premium
    providers.local = local
    config = resilience_config(
        config,
        retry_max_attempts=1,
        circuit_failure_threshold=1,
    )
    app = create_app(config=config, cache=FakeCache(), providers=providers)

    with TestClient(app, raise_server_exceptions=True) as client:
        with pytest.raises(ProviderError):
            client.post(
                "/v1/chat/completions",
                json=chat_request().model_dump(),
                headers={"X-Model-Tier": "premium"},
            )
        assert app.state.breakers["premium"].state is CircuitState.OPEN

        denied = client.post(
            "/v1/chat/completions",
            json=chat_request().model_dump(),
            headers={"X-Model-Tier": "premium"},
        )
        allowed = client.post(
            "/v1/chat/completions",
            json=chat_request().model_dump(),
            headers={"X-Model-Tier": "premium", "X-Allow-Fallback": "true"},
        )

    assert denied.status_code == 503
    assert denied.json()["error"]["requires_confirmation"] is True
    assert allowed.headers["X-Fallback"] == "true"
    assert allowed.headers["X-Model-Used"] == "phi4-mini"
    assert local.calls == 1


def test_cache_hit_bypasses_provider_and_does_not_store(config):
    provider = SequenceProvider([LLMResponse(content="cached", model="phi4-mini")])
    providers = FakeRegistry()
    providers.local = provider
    cache = FakeCache()
    app = create_app(config=config, cache=cache, providers=providers)

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            json=chat_request().model_dump(),
            headers={"X-Model-Tier": "local"},
        )
        response = client.post(
            "/v1/chat/completions",
            json=chat_request().model_dump(),
            headers={"X-Model-Tier": "local"},
        )

    assert first.status_code == 200
    assert response.status_code == 200
    assert provider.calls == 1
    assert len(cache.lookups) == 2
    assert len(cache.writes) == 1


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
