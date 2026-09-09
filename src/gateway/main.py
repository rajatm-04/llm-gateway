"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from gateway.cache.semantic_cache import SemanticCache
from gateway.config import settings
from gateway.resilience.rate_limiter import (
    RateLimiter,
    rate_limit_identity,
    rate_limit_response,
)

cache = SemanticCache()
rate_limiter = RateLimiter(settings.rate_limit_rpm)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed the configured per-client request rate."""

    async def dispatch(self, request: Request, call_next):
        allowed, retry_after = rate_limiter.allow(rate_limit_identity(request))
        if not allowed:
            return rate_limit_response(retry_after)

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up application resources."""
    app.state.cache = cache

    await cache.initialize()

    yield

    await cache.close()


app = FastAPI(
    title="LLM Gateway",
    description="High-Performance LLM Gateway with Semantic Caching & Routing",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(RateLimitMiddleware)

from gateway.api.v1.chat import router as chat_router

app.include_router(chat_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the LLM Gateway!"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": app.version,
    }
