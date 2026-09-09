"""Resolve routing once, enforce cache scope, then generate or stream."""

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
from gateway.providers.base import Provider
from gateway.router.model_router import RoutingDecision, RoutingError
from gateway.streaming.sse_handler import format_done, format_sse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


def create_response(*, content: str, model: str, created: int | None = None,
                    prompt_tokens: int = 0, completion_tokens: int = 0,
                    finish_reason: str = "stop") -> ChatCompletionResponse:
    if finish_reason not in {"stop", "length", "None"}:
        finish_reason = "stop"
    return ChatCompletionResponse(
        created=created if created is not None else int(time.time()),
        model=model,
        choices=[ChatCompletionChoice(
            index=0, message=ChatMessage(role="assistant", content=content), finish_reason=finish_reason,
        )],
        usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens),
    )


def cache_metadata(decision: RoutingDecision, model: str, **extra) -> dict:
    return {
        "model": model, "selected_model": decision.model, "model_tier": decision.tier,
        "provider": decision.provider, "routing_reason": decision.reason,
        "policy_version": decision.policy_version, "profile_status": decision.profile_status,
        "created": int(time.time()), **extra,
    }


async def safe_store(cache, key: CacheKey, content: str, metadata: dict) -> None:
    try:
        await cache.store(key=key, response=content, metadata=metadata)
    except Exception:  # noqa: BLE001 -- optional cache failures must not discard the generated answer.
        # Never log request text, credentials, or raw upstream exception messages.
        logger.warning("cache_store_failed")


async def stream_cached(payload: dict) -> AsyncIterator[str]:
    yield format_sse(content=payload["response"])
    yield format_sse(finish_reason=payload.get("finish_reason", "stop"))
    yield format_done()


async def stream_and_cache(*, request: ChatCompletionRequest, provider: Provider,
                           cache, key: CacheKey, decision: RoutingDecision) -> AsyncIterator[str]:
    chunks: list[str] = []
    terminal = None
    try:
        async with aclosing(provider.generate_stream(
            messages=request.messages, model=decision.model,
            temperature=request.temperature, max_tokens=request.max_tokens,
        )) as stream:
            async for chunk in stream:
                if chunk.content:
                    chunks.append(chunk.content)
                    yield format_sse(content=chunk.content)
                if chunk.finish_reason is not None:
                    terminal = chunk
            if terminal is None:
                raise RuntimeError("Missing terminal stream event")
    except Exception:  # noqa: BLE001 -- after SSE headers, sanitize failures and never cache partial output.
        # Cancellation is a BaseException and propagates: no partial response caching.
        logger.warning("provider_stream_failed")
        yield 'data: {"error": {"code": "upstream_stream_error", "message": "Provider stream failed"}}\n\n'
        yield format_done()
        return

    content = "".join(chunks)
    finish_reason = terminal.finish_reason if terminal.finish_reason in {"stop", "length"} else "stop"
    if content.strip():
        await safe_store(cache, key, content, cache_metadata(
            decision, terminal.model or decision.model,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            usage_known=False, finish_reason=finish_reason,
        ))
    yield format_sse(finish_reason=finish_reason)
    yield format_done()


@router.post("/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest, http_request: Request,
                           x_model_tier: str | None = Header(default=None)):
    state = http_request.app.state
    try:
        decision = state.model_router.select(request, x_model_tier)
    except RoutingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    # Resolve omitted generation budgets before caching or dispatch.
    if request.max_tokens is None:
        budget = (state.config.local_max_output_tokens if decision.tier == "local"
                  else state.config.premium_max_output_tokens)
        request = request.model_copy(update={"max_tokens": budget})
    key = build_cache_key(request, decision, state.config)
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
    logger.info(json.dumps({
        "event": "routing_decision", "tier": decision.tier, "selected_model": decision.model,
        "reason": decision.reason, "policy": decision.policy_version,
        "profile": decision.profile_status, "cache": headers["X-Cache"],
    }))
    if payload is not None:
        headers["X-Model-Used"] = payload["model"]
        if request.stream:
            return StreamingResponse(stream_cached(payload), media_type="text/event-stream", headers=headers)
        response = create_response(content=payload["response"], model=payload["model"],
                                   created=payload.get("created"),
                                   finish_reason=payload.get("finish_reason", "stop"))
        return JSONResponse(response.model_dump(mode="json"), headers=headers)

    # A scoped cache hit requires no credential; an override promises provenance, not fresh execution.
    try:
        provider = state.providers.get(decision.tier)
    except RoutingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if request.stream:
        headers["X-Model-Used-Source"] = "selected"
        return StreamingResponse(
            stream_and_cache(request=request, provider=provider, cache=state.cache, key=key, decision=decision),
            media_type="text/event-stream", headers=headers,
        )
    try:
        result = await provider.generate(messages=request.messages, model=decision.model,
                                         temperature=request.temperature, max_tokens=request.max_tokens)
    except (httpx.TimeoutException, TimeoutError) as exc:
        raise HTTPException(504, "Provider timed out") from exc
    except (httpx.HTTPError, APIError, ValueError, RuntimeError) as exc:
        raise HTTPException(502, "Provider request failed") from exc
    if not result.content.strip():
        raise HTTPException(502, "Provider returned no text response")
    response = create_response(content=result.content, model=result.model,
                               prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                               finish_reason=result.finish_reason)
    headers["X-Model-Used"] = result.model
    await safe_store(state.cache, key, result.content, cache_metadata(
        decision, result.model, created=response.created, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, total_tokens=result.total_tokens,
        finish_reason=response.choices[0].finish_reason,
    ))
    return JSONResponse(response.model_dump(mode="json"), headers=headers)
