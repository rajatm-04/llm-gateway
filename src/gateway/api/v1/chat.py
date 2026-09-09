"""Resolve routing once, enforce cache scope, then generate or stream."""

import asyncio
import json
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
from gateway.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from gateway.resilience.retry import RetryConfig, retry_async, retry_stream
from gateway.router.model_router import RoutingDecision, RoutingError
from gateway.streaming.sse_handler import format_done, format_error, format_sse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])

def fallback_allowed(request: Request) -> bool:
    """Read explicit client consent for premium-to-local fallback."""
    return request.headers.get("x-allow-fallback", "").lower() == "true"


def fallback_unavailable_response(config) -> JSONResponse:
    """Tell the client that local fallback requires explicit confirmation."""
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "premium_provider_unavailable",
                "message": "The premium model is currently unavailable.",
                "fallback_available": True,
                "fallback_model": config.ollama_model,
                "requires_confirmation": True,
            }
        },
        headers={"Retry-After": str(int(config.circuit_recovery_timeout))},
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
        "profile_status": decision.profile_status,
        "created": int(time.time()),
        **extra,
    }


async def safe_store(cache, key: CacheKey, content: str, metadata: dict) -> None:
    try:
        await cache.store(key=key, response=content, metadata=metadata)
    except Exception:  # noqa: BLE001 -- cache failures must not discard an answer.
        logger.warning("cache_store_failed")


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
    """Stream chunks with resilience and cache only complete, terminal streams."""
    chunks: list[str] = []
    terminal: StreamChunk | None = None
    dispatch_model = model or decision.model
    operation = lambda: provider.generate_stream(
        messages=request.messages,
        model=dispatch_model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    if retry is not None:
        stream = retry_stream(operation, retry)
    else:
        stream = operation()
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
    except asyncio.CancelledError:
        logger.info("provider_stream_cancelled")
        raise
    except Exception:  # noqa: BLE001 -- sanitize failures after SSE headers are sent.
        logger.warning("provider_stream_failed")
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
        logger.warning("cache_lookup_failed")
        payload, cache_failed = None, True

    headers = {
        "X-Cache-Latency-Ms": f"{(time.perf_counter() - started) * 1000:.2f}",
        "X-Cache": "HIT" if payload is not None else "BYPASS" if cache_failed else "MISS",
        "X-Model-Tier": decision.tier,
        "X-Model-Used": decision.model,
        "X-Routing-Reason": decision.reason,
        "X-Routing-Policy": decision.policy_version,
        "X-Routing-Profile": decision.profile_status,
    }
    logger.info(
        json.dumps(
            {
                "event": "routing_decision",
                "tier": decision.tier,
                "selected_model": decision.model,
                "reason": decision.reason,
                "policy": decision.policy_version,
                "profile": decision.profile_status,
                "cache": headers["X-Cache"],
            }
        )
    )
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

    allow_fallback = fallback_allowed(http_request)
    fallback = False
    dispatch_model = decision.model
    try:
        provider = state.providers.get(decision.tier)
    except RoutingError as exc:
        if decision.tier != "premium":
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        if not allow_fallback:
            return fallback_unavailable_response(config)
        provider = state.providers.get("local")
        dispatch_model = config.ollama_model
        fallback = True

    breaker = _breaker(state, "local" if fallback else decision.tier, config)
    if breaker.state is CircuitState.OPEN:
        if decision.tier == "premium" and allow_fallback and not fallback:
            provider = state.providers.get("local")
            dispatch_model = config.ollama_model
            breaker = _breaker(state, "local", config)
            fallback = True
        elif decision.tier == "premium":
            return fallback_unavailable_response(config)
        else:
            raise HTTPException(
                status_code=503,
                detail="LLM provider circuit is open",
                headers={"Retry-After": str(int(config.circuit_recovery_timeout))},
            )

    if request.stream:
        headers["X-Model-Used-Source"] = "selected"
        if fallback:
            headers["X-Fallback"] = "true"
            headers["X-Model-Used"] = dispatch_model
        return StreamingResponse(
            stream_and_cache(
                request=request,
                provider=provider,
                cache=state.cache,
                key=key,
                decision=decision,
                breaker=breaker,
                retry=_retry_config(config),
                model=dispatch_model,
                cache_response=not fallback,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    try:
        result: LLMResponse = await breaker.execute(
            lambda: retry_async(
                lambda: provider.generate(
                    messages=request.messages,
                    model=dispatch_model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ),
                _retry_config(config),
            )
        )
    except CircuitOpenError as exc:
        if decision.tier == "premium" and allow_fallback and not fallback:
            provider = state.providers.get("local")
            breaker = _breaker(state, "local", config)
            dispatch_model = config.ollama_model
            fallback = True
            result = await breaker.execute(
                lambda: retry_async(
                    lambda: provider.generate(
                        messages=request.messages,
                        model=dispatch_model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    ),
                    _retry_config(config),
                )
            )
        elif decision.tier == "premium":
            return fallback_unavailable_response(config)
        else:
            raise HTTPException(
                status_code=503,
                detail="LLM provider circuit is open",
                headers={"Retry-After": str(int(config.circuit_recovery_timeout))},
            ) from exc
    except (httpx.TimeoutException, TimeoutError) as exc:
        raise HTTPException(status_code=504, detail="Provider timed out") from exc
    except (httpx.HTTPError, APIError, ValueError, RuntimeError) as exc:
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
    if not fallback:
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
    else:
        headers["X-Fallback"] = "true"
    return JSONResponse(response.model_dump(mode="json"), headers=headers)
