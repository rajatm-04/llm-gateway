"""Common interface for LLM providers."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from gateway.models.schemas import ChatMessage


@dataclass(slots=True)
class LLMResponse:
    """Standard response returned by an LLM provider."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens used."""
        return self.prompt_tokens + self.completion_tokens


class Provider(Protocol):
    """Interface every LLM provider must implement."""

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a complete response."""
        ...

    async def generate_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield generated response chunks."""
        ...
