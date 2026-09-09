"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from gateway.cache.semantic_cache import SemanticCache


cache = SemanticCache()


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

