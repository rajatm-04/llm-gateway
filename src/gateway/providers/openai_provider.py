"""OpenAI LLM provider."""

from collections.abc import AsyncIterator, Sequence

from openai import AsyncOpenAI

from gateway.config import settings
from gateway.models.schemas import ChatMessage
from gateway.providers.base import LLMResponse, StreamChunk


class OpenAIProvider:
    """Generate responses through OpenAI's Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 0,
    ) -> None:
        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY is not configured")

        self.client = AsyncOpenAI(
            api_key=key,
            base_url=base_url or settings.openai_base_url,
            timeout=timeout or settings.openai_timeout,
            max_retries=max_retries,
        )
        self.default_model = default_model or settings.openai_model

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a complete response from OpenAI."""
        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[message.model_dump() for message in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response.choices:
            raise RuntimeError("Provider returned no choices")
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    async def generate_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield response content chunks from OpenAI."""
        stream = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[message.model_dump() for message in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        finished = False
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    yield StreamChunk(content=content)
                if chunk.choices[0].finish_reason is not None:
                    finished = True
                    yield StreamChunk(finish_reason=chunk.choices[0].finish_reason, model=chunk.model)
            if not finished:
                raise RuntimeError("Provider stream ended before its completion marker")
        finally:
            await stream.close()

    async def close(self) -> None:
        """Release the shared SDK HTTP connection pool."""
        await self.client.close()
