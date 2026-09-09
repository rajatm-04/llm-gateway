"""FastAPI application factory with injectable resources for offline tests."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from gateway.api.v1.chat import router as chat_router
from gateway.config import Settings, settings
from gateway.providers.registry import ProviderRegistry
from gateway.router.model_router import ModelRouter


def create_app(*, config: Settings | None = None, cache=None, model_router=None, providers=None) -> FastAPI:
    config = config if config is not None else settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.model_router = model_router if model_router is not None else ModelRouter(config)
        app.state.providers = providers if providers is not None else ProviderRegistry(config)
        # No Qdrant client or embedding import is needed to collect offline tests.
        if cache is None:
            from gateway.cache.semantic_cache import SemanticCache
            active_cache = SemanticCache(config=config)
        else:
            active_cache = cache
        app.state.cache = active_cache
        try:
            await active_cache.initialize()
            yield
        finally:
            try:
                await app.state.providers.close()
            finally:
                await active_cache.close()

    app = FastAPI(
        title="LLM Gateway", version="1.0.0",
        description="Experimental task-aware routing and scoped semantic caching",
        lifespan=lifespan,
    )
    app.include_router(chat_router)
    web_directory = Path(__file__).resolve().parent / "web"
    app.mount("/assets", StaticFiles(directory=web_directory), name="assets")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(web_directory / "index.html", headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        })

    @app.get("/ui/config", include_in_schema=False)
    async def ui_config():
        # Deliberate public allowlist: never serialize Settings or expose credentials/endpoints.
        return JSONResponse({
            "local_model": config.ollama_model,
            "premium_model": config.openai_model,
            "routing_mode": config.routing_mode,
            "local_max_output_tokens": config.local_max_output_tokens,
            "premium_max_output_tokens": config.premium_max_output_tokens,
        }, headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": app.version}

    return app


app = create_app()
