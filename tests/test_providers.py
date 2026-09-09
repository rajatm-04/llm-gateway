import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.providers.ollama_provider import OllamaProvider
from gateway.providers.openai_provider import OpenAIProvider


def test_ollama_payload_uses_resolved_model():
    provider = OllamaProvider()
    payload = provider._payload([], model="chosen", temperature=0.0, max_tokens=32, stream=True)
    assert payload["model"] == "chosen"
    assert payload["options"] == {"temperature": 0.0, "num_predict": 32}


@pytest.mark.asyncio
async def test_openai_stream_metadata_empty_choices_and_cleanup():
    class FakeStream:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

        def __aiter__(self):
            return self.events()

        async def events(self):
            yield SimpleNamespace(choices=[], model="actual")
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"), finish_reason=None)], model="actual")
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="length")], model="actual")

    stream = FakeStream()
    create = AsyncMock(return_value=stream)
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.default_model = "default"
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    chunks = [chunk async for chunk in provider.generate_stream([], model="selected", max_tokens=32)]
    assert chunks[0].content == "Hi"
    assert chunks[-1].finish_reason == "length"
    assert chunks[-1].model == "actual"
    assert create.call_args.kwargs["model"] == "selected"
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_openai_stream_closes_on_failure_or_cancellation(failure):
    class FailingStream:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

        def __aiter__(self):
            return self.events()

        async def events(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="partial"),
                        finish_reason=None,
                    )
                ],
                model="actual",
            )
            raise failure("upstream failure")

    stream = FailingStream()
    create = AsyncMock(return_value=stream)
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.default_model = "default"
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(failure):
        _ = [chunk async for chunk in provider.generate_stream([], model="selected")]

    assert stream.closed


@pytest.mark.asyncio
async def test_ollama_stream_with_mock_transport(monkeypatch):
    real_client = httpx.AsyncClient

    def handler(request):
        return httpx.Response(200, content=(
            '{"message":{"content":"Hi"},"done":false}\n'
            '{"message":{"content":""},"done":true,"done_reason":"stop","model":"actual"}\n'
        ))

    monkeypatch.setattr("gateway.providers.ollama_provider.httpx.AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    provider = OllamaProvider(host="http://test.invalid")
    chunks = [chunk async for chunk in provider.generate_stream([], model="chosen")]
    assert chunks[0].content == "Hi"
    assert chunks[-1].model == "actual"
    assert chunks[-1].finish_reason == "stop"
