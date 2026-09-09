"""Chat completion API."""

import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.config import settings
from gateway.models.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)
from gateway.providers.base import LLMResponse, Provider
from gateway.providers.ollama_provider import OllamaProvider
from gateway.providers.openai_provider import OpenAIProvider
from gateway.streaming.sse_handler import format_done, format_sse


router = APIRouter(prefix="/v1", tags=["chat"])
ollama_provider = OllamaProvider()


def get_provider(model: str | None) -> Provider:
    """Select OpenAI for GPT models; otherwise use Ollama."""
    if model and model.lower().startswith("gpt-"):
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=503,
                detail="OPENAI_API_KEY is not configured",
            )
        return OpenAIProvider()

    return ollama_provider


def build_cache_query(request: ChatCompletionRequest) -> str:
    """Create a cache query from model, options, and conversation."""
    messages = "\n".join(
        f"{message.role}: {message.content}"
        for message in request.messages
    )
    return (
        f"model={request.model or settings.ollama_model}\n"
        f"temperature={request.temperature}\n"
        f"max_tokens={request.max_tokens}\n"
        f"{messages}"
    )


def create_response(
    *,
    content: str,
    model: str,
    created: int | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    finish_reason: str = "stop",
) -> ChatCompletionResponse:
    """Create an OpenAI-compatible chat response."""
    if finish_reason not in {"stop", "length", "None"}:
        finish_reason = "stop"

    return ChatCompletionResponse(
        object="chat.completion",
        created=created or int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def response_from_cache(
    payload: dict,
    requested_model: str | None,
) -> ChatCompletionResponse:
    """Build a cached response with zero new LLM token usage."""
    return create_response(
        content=payload["response"],
        model=payload.get("model", requested_model or settings.ollama_model),
        created=payload.get("created"),
        # Cache hits make no provider call; original usage stays in Qdrant.
        prompt_tokens=0,
        completion_tokens=0,
        finish_reason=payload.get("finish_reason", "stop"),
    )


async def stream_cached(content: str) -> AsyncIterator[str]:
    """Send cached content using SSE."""
    yield format_sse(content=content)
    yield format_sse(finish_reason="stop")
    yield format_done()


async def stream_and_cache(
    *,
    request: ChatCompletionRequest,
    provider: Provider,
    cache,
    query: str,
) -> AsyncIterator[str]:
    """Stream provider chunks and cache the complete response afterward."""
    chunks: list[str] = []

    async for chunk in provider.generate_stream(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    ):
        chunks.append(chunk)
        yield format_sse(content=chunk)

    content = "".join(chunks)
    model = request.model or settings.ollama_model
    created = int(time.time())

    await cache.store(
        query=query,
        response=content,
        metadata={
            "model": model,
            "created": created,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reason": "stop",
        },
    )

    yield format_sse(finish_reason="stop")
    yield format_done()


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
):
    """Return a cached response or generate one through an LLM provider."""
    cache = http_request.app.state.cache
    query = build_cache_query(request)

    started_at = time.perf_counter()
    cached_payload = await cache.lookup(query)
    cache_latency_ms = (time.perf_counter() - started_at) * 1000
    headers = {"X-Cache-Latency-Ms": f"{cache_latency_ms:.2f}"}

    if cached_payload is not None:
        headers["X-Cache"] = "HIT"

        if request.stream:
            return StreamingResponse(
                stream_cached(cached_payload["response"]),
                media_type="text/event-stream",
                headers=headers,
            )

        response = response_from_cache(cached_payload, request.model)
        return JSONResponse(
            content=response.model_dump(mode="json"),
            headers=headers,
        )

    provider = get_provider(request.model)
    headers["X-Cache"] = "MISS"

    if request.stream:
        return StreamingResponse(
            stream_and_cache(
                request=request,
                provider=provider,
                cache=cache,
                query=query,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    provider_response: LLMResponse = await provider.generate(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    response = create_response(
        content=provider_response.content,
        model=provider_response.model,
        prompt_tokens=provider_response.prompt_tokens,
        completion_tokens=provider_response.completion_tokens,
        finish_reason=provider_response.finish_reason,
    )

    await cache.store(
        query=query,
        response=provider_response.content,
        metadata={
            "model": response.model,
            "created": response.created,
            "prompt_tokens": provider_response.prompt_tokens,
            "completion_tokens": provider_response.completion_tokens,
            "total_tokens": provider_response.total_tokens,
            "finish_reason": provider_response.finish_reason,
        },
    )

    return JSONResponse(
        content=response.model_dump(mode="json"),
        headers=headers,
    )
