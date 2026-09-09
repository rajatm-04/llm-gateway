import asyncio

import pytest
from conftest import FakeCache, FakeRegistry, request_for
from fastapi.testclient import TestClient

from gateway.api.v1.chat import stream_and_cache
from gateway.cache.scope import build_cache_key
from gateway.main import create_app
from gateway.providers.base import StreamChunk
from gateway.providers.registry import ProviderRegistry
from gateway.router.model_router import ModelRouter
from gateway.streaming.sse_handler import format_done, format_error, format_sse


@pytest.fixture
def resources(config):
    cache, providers = FakeCache(), FakeRegistry()
    app = create_app(config=config, cache=cache, providers=providers)
    with TestClient(app) as client:
        yield client, cache, providers


def test_local_miss_hit_and_headers(resources):
    client, cache, providers = resources
    body = request_for().model_dump()
    first = client.post("/v1/chat/completions", json=body)
    assert first.status_code == 200
    assert first.headers["x-model-tier"] == "local"
    assert first.headers["x-routing-profile"] == "experimental"
    assert first.headers["x-cache"] == "MISS"
    assert providers.local.calls[0]["model"] == "phi4-mini"
    second = client.post("/v1/chat/completions", json=body)
    assert second.headers["x-cache"] == "HIT"
    assert second.json()["usage"]["total_tokens"] == 0
    assert len(providers.local.calls) == 1
    assert len(cache.writes) == 1


def test_premium_override_cannot_hit_local_cache(resources):
    client, _, providers = resources
    body = request_for().model_dump()
    client.post("/v1/chat/completions", json=body)
    result = client.post("/v1/chat/completions", json=body, headers={"X-Model-Tier": "premium"})
    assert result.headers["x-cache"] == "MISS"
    assert result.headers["x-model-used"] == "gpt-5.6-sol"
    assert len(providers.premium.calls) == 1


def test_validation_precedes_cache(resources):
    client, cache, _ = resources
    response = client.post("/v1/chat/completions", json=request_for(model="phi4-mini").model_dump(),
                           headers={"X-Model-Tier": "premium"})
    assert response.status_code == 422
    assert not cache.lookups


def test_stream_resolved_model_finish_reason_and_cache(resources):
    client, cache, providers = resources
    body = request_for("Prove this theorem.", stream=True).model_dump()
    first = client.post("/v1/chat/completions", json=body)
    assert first.status_code == 200
    assert first.headers["x-model-used"] == "gpt-5.6-sol"
    assert first.headers["x-model-used-source"] == "selected"
    assert '"finish_reason": "length"' in first.text
    assert "data: [DONE]" in first.text
    assert cache.writes[0][2]["model"] == "gpt-5.6-sol"
    assert providers.premium.calls[0]["model"] == "gpt-5.6-sol"
    second = client.post("/v1/chat/completions", json=body)
    assert second.headers["x-cache"] == "HIT"
    assert '"finish_reason": "length"' in second.text


@pytest.mark.parametrize("flag", ["fail_stream", "omit_terminal"])
def test_failed_or_incomplete_stream_not_cached(resources, flag):
    client, cache, providers = resources
    setattr(providers.local, flag, True)
    response = client.post("/v1/chat/completions", json=request_for(stream=True).model_dump())
    assert "upstream_stream_error" in response.text
    assert response.text.endswith(format_error() + format_done())
    assert not cache.writes


@pytest.mark.asyncio
async def test_cancelled_stream_not_cached(config):
    cache, providers = FakeCache(), FakeRegistry()
    providers.local.cancel_stream = True
    request = request_for(stream=True)
    decision = ModelRouter(config).select(request)
    with pytest.raises(asyncio.CancelledError):
        async for _ in stream_and_cache(request=request, provider=providers.local, cache=cache,
                                        key=build_cache_key(request, decision, config), decision=decision):
            pass
    assert not cache.writes


@pytest.mark.asyncio
async def test_cancelled_stream_propagates_and_closes_provider(config):
    class CancelledProvider:
        closed = False

        async def generate_stream(self, **kwargs):
            try:
                yield StreamChunk(content="partial")
                raise asyncio.CancelledError()
            finally:
                self.closed = True

    cache = FakeCache()
    provider = CancelledProvider()
    request = request_for(stream=True)
    decision = ModelRouter(config).select(request)
    events = []

    with pytest.raises(asyncio.CancelledError):
        async for event in stream_and_cache(
            request=request,
            provider=provider,
            cache=cache,
            key=build_cache_key(request, decision, config),
            decision=decision,
        ):
            events.append(event)

    assert events == [format_sse(content="partial")]
    assert provider.closed
    assert not cache.writes


def test_successful_stream_ends_with_terminal_and_done_events(resources):
    client, _, _ = resources
    response = client.post(
        "/v1/chat/completions",
        json=request_for(stream=True).model_dump(),
    )

    assert response.text.endswith(format_sse(finish_reason="length") + format_done())


def test_cache_failure_does_not_discard_answer(resources):
    client, cache, _ = resources
    cache.fail_lookup = cache.fail_store = True
    response = client.post("/v1/chat/completions", json=request_for().model_dump())
    assert response.status_code == 200
    assert response.headers["x-cache"] == "BYPASS"


def test_missing_premium_key_returns_503_without_network(config):
    app = create_app(config=config, cache=FakeCache(), providers=ProviderRegistry(config))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_for("An unknown task.").model_dump())
        assert response.status_code == 503
