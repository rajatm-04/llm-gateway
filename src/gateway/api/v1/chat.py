"""Resolve routing once, enforce cache scope, then generate or stream."""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import aclosing

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIError

from gateway.cache.scope import CacheKey, build_cache_key
from gateway.models.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)
from gateway.providers.base import LLMResponse, Provider, StreamChunk
from gateway.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from gateway.resilience.retry import RetryConfig, retry_async, retry_stream
from gateway.router.model_router import RoutingDecision, RoutingError
from gateway.streaming.sse_handler import format_done, format_error, format_sse

router = APIRouter(prefix="/v1", tags=["chat"])
logger = logging.getLogger(__name__)


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
        created=created if created is not None else int(time.time()),
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


def cache_metadata(decision: RoutingDecision, model: str, **extra) -> dict:
    return {
        "model": model,
        "selected_model": decision.model,
        "model_tier": decision.tier,
        "provider": decision.provider,
        "routing_reason": decision.reason,
        "policy_version": decision.policy_version,
        "created": int(time.time()),
        **extra,
    }


async def safe_store(cache, key: CacheKey, content: str, metadata: dict) -> None:
    try:
        await cache.store(key=key, response=content, metadata=metadata)
    except Exception:  # noqa: BLE001 -- cache failures must not discard an answer.
        logger.warning("Cache store failed; bypassing cache")


async def stream_cached(payload: dict) -> AsyncIterator[str]:
    yield format_sse(content=payload["response"])
    yield format_sse(finish_reason=payload.get("finish_reason", "stop"))
    yield format_done()


def _chunk_content(chunk: StreamChunk | str) -> str:
    return chunk.content if isinstance(chunk, StreamChunk) else chunk


async def stream_and_cache(
    *,
    request: ChatCompletionRequest,
    provider: Provider,
    cache,
    key: CacheKey,
    decision: RoutingDecision,
    breaker: CircuitBreaker | None = None,
    retry: RetryConfig | None = None,
    model: str | None = None,
    cache_response: bool = True,
) -> AsyncIterator[str]:
    """Stream chunks and cache only complete, terminal streams."""
    chunks: list[str] = []
    terminal: StreamChunk | None = None
    dispatch_model = model or decision.model
    operation = lambda: provider.generate_stream(
        messages=request.messages,
        model=dispatch_model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    stream = retry_stream(operation, retry) if retry is not None else operation()
    if breaker is not None:
        guarded_stream = stream
        stream = breaker.execute_stream(lambda: guarded_stream)

    try:
        async with aclosing(stream) as active_stream:
            async for chunk in active_stream:
                content = _chunk_content(chunk)
                if content:
                    chunks.append(content)
                    yield format_sse(content=content)
                if isinstance(chunk, StreamChunk) and chunk.finish_reason is not None:
                    terminal = chunk
            if terminal is None:
                raise RuntimeError("Missing terminal stream event")
    except Exception:  # noqa: BLE001 -- sanitize failures after SSE headers are sent.
        yield format_error()
        yield format_done()
        return

    content = "".join(chunks)
    finish_reason = terminal.finish_reason if terminal.finish_reason in {"stop", "length"} else "stop"
    if cache_response and content.strip():
        await safe_store(
            cache,
            key,
            content,
            cache_metadata(
                decision,
                terminal.model or dispatch_model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                usage_known=False,
                finish_reason=finish_reason,
            ),
        )
    yield format_sse(finish_reason=finish_reason)
    yield format_done()


def _retry_config(config) -> RetryConfig:
    return RetryConfig(
        max_attempts=config.retry_max_attempts,
        base_delay=config.retry_base_delay,
        max_delay=config.retry_max_delay,
        jitter=config.retry_jitter,
    )


def _breaker(state, tier: str, config) -> CircuitBreaker:
    breakers = getattr(state, "breakers", None)
    if breakers is None:
        breakers = {}
        state.breakers = breakers
    if tier not in breakers:
        breakers[tier] = CircuitBreaker(
            failure_threshold=config.circuit_failure_threshold,
            recovery_timeout=config.circuit_recovery_timeout,
            half_open_max_calls=config.circuit_half_open_max_calls,
        )
    return breakers[tier]


def _provider_unavailable(config, *, circuit_open: bool = False) -> JSONResponse:
    message = "The selected provider circuit is open." if circuit_open else "The selected provider is unavailable."
    return JSONResponse(
        status_code=503,
        content={"error": {"code": "provider_unavailable", "message": message}},
        headers={"Retry-After": str(int(config.circuit_recovery_timeout))},
    )


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    x_model_tier: str | None = Header(default=None),
):
    state = http_request.app.state
    config = state.config
    try:
        decision = state.model_router.select(request, x_model_tier)
    except RoutingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    if request.max_tokens is None:
        budget = (
            config.local_max_output_tokens
            if decision.tier == "local"
            else config.premium_max_output_tokens
        )
        request = request.model_copy(update={"max_tokens": budget})
    key = build_cache_key(request, decision, config)
    started = time.perf_counter()
    cache_failed = False
    try:
        payload = await state.cache.lookup(key)
    except Exception:  # noqa: BLE001 -- optional cache failures fall through to generation.
        payload, cache_failed = None, True

    headers = {
        "X-Cache-Latency-Ms": f"{(time.perf_counter() - started) * 1000:.2f}",
        "X-Cache": "HIT" if payload is not None else "BYPASS" if cache_failed else "MISS",
        "X-Model-Tier": decision.tier,
        "X-Model-Used": decision.model,
        "X-Routing-Reason": decision.reason,
        "X-Routing-Policy": decision.policy_version,
    }
    if payload is not None:
        headers["X-Model-Used"] = payload["model"]
        if request.stream:
            return StreamingResponse(stream_cached(payload), media_type="text/event-stream", headers=headers)
        response = create_response(
            content=payload["response"],
            model=payload["model"],
            created=payload.get("created"),
            finish_reason=payload.get("finish_reason", "stop"),
        )
        return JSONResponse(response.model_dump(mode="json"), headers=headers)

    try:
        provider = state.providers.get(decision.tier)
    except RoutingError as exc:
        if exc.status_code == 503:
            return _provider_unavailable(config)
        raise HTTPException(status_code=exc.status_code, detail="Selected provider is unavailable") from exc
    except Exception:  # noqa: BLE001 -- sanitize unexpected provider setup failures.
        return _provider_unavailable(config)

    breaker = _breaker(state, decision.tier, config)
    if breaker.state.value == "open":
        return _provider_unavailable(config, circuit_open=True)

    if request.stream:
        headers["X-Model-Used-Source"] = "selected"
        return StreamingResponse(
            stream_and_cache(
                request=request,
                provider=provider,
                cache=state.cache,
                key=key,
                decision=decision,
                breaker=breaker,
                retry=_retry_config(config),
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    try:
        result: LLMResponse = await breaker.execute(
            lambda: retry_async(
                lambda: provider.generate(
                    messages=request.messages,
                    model=decision.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ),
                _retry_config(config),
            )
        )
    except CircuitOpenError:
        return _provider_unavailable(config, circuit_open=True)
    except (httpx.TimeoutException, TimeoutError) as exc:
        if decision.tier == "premium":
            return _provider_unavailable(config)
        raise HTTPException(status_code=504, detail="Provider timed out") from exc
    except (httpx.HTTPError, APIError, ValueError, RuntimeError) as exc:
        if decision.tier == "premium":
            return _provider_unavailable(config)
        raise HTTPException(status_code=502, detail="Provider request failed") from exc
    except Exception as exc:
        if decision.tier == "premium":
            return _provider_unavailable(config)
        raise HTTPException(status_code=502, detail="Provider request failed") from exc

    if not result.content.strip():
        raise HTTPException(status_code=502, detail="Provider returned no text response")
    response = create_response(
        content=result.content,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        finish_reason=result.finish_reason,
    )
    headers["X-Model-Used"] = result.model
    await safe_store(
        state.cache,
        key,
        result.content,
        cache_metadata(
            decision,
            result.model,
            created=response.created,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            finish_reason=response.choices[0].finish_reason,
        ),
    )
    return JSONResponse(response.model_dump(mode="json"), headers=headers)
