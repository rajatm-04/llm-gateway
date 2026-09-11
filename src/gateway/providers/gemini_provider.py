"""Native Gemini provider."""

from collections.abc import AsyncIterator, Sequence

from google import genai
from google.genai import types

from gateway.models.schemas import ChatMessage
from gateway.providers.base import LLMResponse, StreamChunk


class GeminiProvider:
    """Adapt Gemini's native API to Prism's provider interface."""

    def __init__(
        self,
        api_key: str,
        default_model: str,
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )
        self.default_model = default_model

    @staticmethod
    def _convert_messages(
        messages: Sequence[ChatMessage],
    ) -> tuple[str | None, list[dict]]:
        system_instruction = None
        contents = []
        for message in messages:
            if message.role == "system":
                system_instruction = message.content
                continue
            contents.append({
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            })
        return system_instruction, contents

    @staticmethod
    def _config(
        system_instruction: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_instruction, contents = self._convert_messages(messages)
        response = await self.client.aio.models.generate_content(
            model=model or self.default_model,
            contents=contents,
            config=self._config(system_instruction, temperature, max_tokens),
        )
        usage = response.usage_metadata
        return LLMResponse(
            content=response.text or "",
            model=model or self.default_model,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )

    async def generate_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system_instruction, contents = self._convert_messages(messages)
        selected_model = model or self.default_model
        stream = await self.client.aio.models.generate_content_stream(
            model=selected_model,
            contents=contents,
            config=self._config(system_instruction, temperature, max_tokens),
        )
        async for chunk in stream:
            if chunk.text:
                yield StreamChunk(content=chunk.text)
        yield StreamChunk(model=selected_model, finish_reason="stop")

    async def close(self) -> None:
        await self.client.aio.aclose()
