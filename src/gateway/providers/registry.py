"""Lifecycle-managed provider instances, separate from routing decisions."""

from gateway.config import Settings
from gateway.providers.base import Provider
from gateway.providers.gemini_provider import GeminiProvider
from gateway.providers.ollama_provider import OllamaProvider
from gateway.providers.openai_provider import OpenAIProvider
from gateway.router.model_router import RoutingError


class ProviderRegistry:
    def __init__(self, config: Settings):
        self.config = config
        self.local = OllamaProvider(
            host=config.ollama_host, default_model=config.ollama_model, timeout=config.ollama_timeout,
        )
        self.openai = OpenAIProvider(
            api_key=config.openai_api_key,
            default_model=config.openai_model,
            base_url=config.openai_base_url,
            timeout=config.openai_timeout,
        ) if config.openai_api_key else None
        self.gemini = GeminiProvider(
            api_key=config.gemini_api_key,
            default_model=config.gemini_model,
            timeout=config.gemini_timeout,
        ) if config.gemini_api_key else None
        self.premium = self.gemini

    def get(self, tier: str) -> Provider:
        if tier == "local":
            return self.local
        if tier != "premium":
            raise RoutingError("Unsupported provider tier")
        if self.premium is None:
            raise RoutingError("Premium provider API key is not configured", 503)
        return self.premium

    async def close(self) -> None:
        for provider in (self.openai, self.gemini):
            if provider is not None:
                await provider.close()
