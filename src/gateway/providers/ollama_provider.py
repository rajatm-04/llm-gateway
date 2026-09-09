"""Ollama LLM provider."""

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from gateway.config import settings
from gateway.models.schemas import ChatMessage
from gateway.providers.base import LLMResponse


class OllamaProvider:
    """Generate responses using Ollama's chat API."""

    def __init__(
        self,
        host: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")
        self.default_model = default_model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build an Ollama chat request payload."""
        options: dict[str, Any] = {}

        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [message.model_dump() for message in messages],
            "stream": stream,
        }

        if options:
            payload["options"] = options

        return payload

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a complete response from Ollama."""
        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.host}/api/chat",
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        return self._to_response(data)

    async def generate_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield response content chunks from Ollama."""
        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.host}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = json.loads(line)
                    content = data.get("message", {}).get("content")

                    if content:
                        yield content

    def _to_response(self, data: dict[str, Any]) -> LLMResponse:
        """Convert an Ollama response to the common provider format."""
        message = data.get("message", {})

        return LLMResponse(
            content=message.get("content", ""),
            model=data.get("model", self.default_model),
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            completion_tokens=data.get("eval_count", 0) or 0,
            finish_reason=data.get("done_reason", "stop") or "stop",
        )
