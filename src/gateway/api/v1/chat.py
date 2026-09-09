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
from gateway.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from gateway.resilience.retry import RetryConfig, retry_async, retry_stream
from gateway.streaming.sse_handler import format_done, format_sse

router = APIRouter(prefix="/v1", tags=["chat"])
ollama_provider = OllamaProvider()
ollama_breaker = CircuitBreaker(
    failure_threshold=settings.circuit_failure_threshold,
    recovery_timeout=settings.circuit_recovery_timeout,
    half_open_max_calls=settings.circuit_half_open_max_calls,
)
openai_breaker = CircuitBreaker(
    failure_threshold=settings.circuit_failure_threshold,
    recovery_timeout=settings.circuit_recovery_timeout,
    half_open_max_calls=settings.circuit_half_open_max_calls,
)
retry_config = RetryConfig(
    max_attempts=settings.retry_max_attempts,
    base_delay=settings.retry_base_delay,
    max_delay=settings.retry_max_delay,
    jitter=settings.retry_jitter,
)


def get_provider(model: str | None) -> tuple[Provider, CircuitBreaker]:
    """Select OpenAI for GPT models; otherwise use Ollama."""
    if model and model.lower().startswith("gpt-"):
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=503,
                detail="OPENAI_API_KEY is not configured",
            )
        return OpenAIProvider(), openai_breaker

    return ollama_provider, ollama_breaker


def is_premium_model(model: str | None) -> bool:
    """Return whether the requested model uses the premium provider."""
    return bool(model and model.lower().startswith("gpt-"))


def fallback_allowed(request: Request) -> bool:
    """Read explicit client consent for premium-to-local fallback."""
    return request.headers.get("x-allow-fallback", "").lower() == "true"


def fallback_unavailable_response() -> JSONResponse:
    """Tell the client that local fallback requires explicit confirmation."""
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "premium_provider_unavailable",
                "message": "The premium model is currently unavailable.",
                "fallback_available": True,
                "fallback_model": settings.ollama_model,
                "requires_confirmation": True,
            }
        },
        headers={
            "Retry-After": str(int(settings.circuit_recovery_timeout)),
        },
    )


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
    breaker: CircuitBreaker,
    cache,
    query: str,
    cache_response: bool = True,
) -> AsyncIterator[str]:
    """Stream provider chunks and cache the complete response afterward."""
    chunks: list[str] = []

    async for chunk in breaker.execute_stream(
        lambda: retry_stream(
            lambda: provider.generate_stream(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
            retry_config,
        )
    ):
        chunks.append(chunk)
        yield format_sse(content=chunk)

    content = "".join(chunks)
    model = request.model or settings.ollama_model
    created = int(time.time())

    if cache_response:
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

    allow_fallback = fallback_allowed(http_request)
    premium_request = is_premium_model(request.model)

    try:
        provider, breaker = get_provider(request.model)
    except HTTPException:
        if premium_request and allow_fallback:
            provider, breaker = ollama_provider, ollama_breaker
            request = request.model_copy(update={"model": settings.ollama_model})
            headers["X-Fallback"] = "true"
            headers["X-Model-Used"] = settings.ollama_model
        elif premium_request:
            return fallback_unavailable_response()
        else:
            raise
    headers["X-Cache"] = "MISS"

    if request.stream and premium_request and breaker.state is CircuitState.OPEN:
        if allow_fallback:
            provider, breaker = ollama_provider, ollama_breaker
            request = request.model_copy(update={"model": settings.ollama_model})
            headers["X-Fallback"] = "true"
            headers["X-Model-Used"] = settings.ollama_model
        else:
            return fallback_unavailable_response()

    if request.stream:
        return StreamingResponse(
            stream_and_cache(
                request=request,
                provider=provider,
                cache=cache,
                query=query,
                cache_response="X-Fallback" not in headers,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    try:
        provider_response: LLMResponse = await breaker.execute(
            lambda: retry_async(
                lambda: provider.generate(
                    messages=request.messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ),
                retry_config,
            )
        )
    except CircuitOpenError as exc:
        if premium_request and allow_fallback:
            headers["X-Fallback"] = "true"
            headers["X-Model-Used"] = settings.ollama_model
            provider_response = await ollama_breaker.execute(
                lambda: retry_async(
                    lambda: ollama_provider.generate(
                        messages=request.messages,
                        model=settings.ollama_model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    ),
                    retry_config,
                )
            )
        elif premium_request:
            return fallback_unavailable_response()
        else:
            raise HTTPException(
                status_code=503,
                detail="LLM provider circuit is open",
                headers={
                    "Retry-After": str(int(settings.circuit_recovery_timeout))
                },
            ) from exc

    response = create_response(
        content=provider_response.content,
        model=provider_response.model,
        prompt_tokens=provider_response.prompt_tokens,
        completion_tokens=provider_response.completion_tokens,
        finish_reason=provider_response.finish_reason,
    )

    if "X-Fallback" not in headers:
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
