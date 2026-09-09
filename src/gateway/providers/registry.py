"""Lifecycle-managed provider instances, separate from routing decisions."""

from gateway.config import Settings
from gateway.providers.base import Provider
from gateway.providers.ollama_provider import OllamaProvider
from gateway.providers.openai_provider import OpenAIProvider
from gateway.router.model_router import RoutingError


class ProviderRegistry:
    def __init__(self, config: Settings):
        self.config = config
        self.local = OllamaProvider(
            host=config.ollama_host, default_model=config.ollama_model, timeout=config.ollama_timeout,
        )
        self.premium: OpenAIProvider | None = None

    def get(self, tier: str) -> Provider:
        if tier == "local":
            return self.local
        if tier != "premium":
            raise RoutingError("Unsupported provider tier")
        if not self.config.openai_api_key:
            raise RoutingError("OPENAI_API_KEY is not configured", 503)
        if self.premium is None:
            self.premium = OpenAIProvider(
                api_key=self.config.openai_api_key,
                default_model=self.config.openai_model,
                base_url=self.config.openai_base_url,
                timeout=self.config.openai_timeout,
                max_retries=0,
            )
        return self.premium

    async def close(self) -> None:
        if self.premium is not None:
            await self.premium.close()
