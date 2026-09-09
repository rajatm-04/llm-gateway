"""Offline fixtures: no embedding downloads, network calls, or real credentials."""

from pathlib import Path

import pytest

from gateway.config import Settings
from gateway.models.schemas import ChatCompletionRequest
from gateway.providers.base import LLMResponse, StreamChunk


@pytest.fixture
def config():
    return Settings(
        _env_file=None, openai_api_key="", ollama_model="phi4-mini",
        openai_model="gpt-5.6-sol", routing_mode="experimental",
        routing_profile_path=Path(__file__).resolve().parents[1] / "src/gateway/router/profiles/phi4-mini.json",
        ollama_host="http://local.invalid", openai_base_url="https://premium.invalid/v1",
        local_max_input_bytes=12000, premium_max_input_bytes=120000,
        local_max_output_tokens=2048, premium_max_output_tokens=4096,
    )


def request_for(text="Make this email more polite: Send the report today.", **kwargs):
    return ChatCompletionRequest(messages=[{"role": "user", "content": text}], **kwargs)


class FakeCache:
    def __init__(self):
        self.entries = {}
        self.lookups = []
        self.writes = []
        self.fail_lookup = False
        self.fail_store = False

    async def initialize(self):
        pass

    async def close(self):
        pass

    async def lookup(self, key):
        self.lookups.append(key)
        if self.fail_lookup:
            raise RuntimeError("fake cache unavailable")
        return self.entries.get((key.scope_id, key.request_id))

    async def store(self, key, response, metadata):
        if self.fail_store:
            raise RuntimeError("fake write failure")
        self.writes.append((key, response, metadata))
        self.entries[(key.scope_id, key.request_id)] = {"response": response, **metadata}


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.fail_stream = False
        self.cancel_stream = False
        self.omit_terminal = False

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse("A complete answer.", kwargs["model"], 10, 4)

    async def generate_stream(self, **kwargs):
        import asyncio
        self.calls.append(kwargs)
        yield StreamChunk(content="A complete ")
        if self.cancel_stream:
            raise asyncio.CancelledError()
        if self.fail_stream:
            raise RuntimeError("fake upstream failure")
        yield StreamChunk(content="answer.")
        if not self.omit_terminal:
            yield StreamChunk(finish_reason="length", model=kwargs["model"])


class FakeRegistry:
    def __init__(self):
        self.local = FakeProvider()
        self.premium = FakeProvider()

    def get(self, tier):
        return self.local if tier == "local" else self.premium

    async def close(self):
        pass
