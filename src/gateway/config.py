"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway configuration.

    All values can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Providers
    openai_api_key: str = ""

    # Qdrant Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Embedding Model
    embedding_model: str = "phi-3:mini"

    # Cache Settings
    cache_similarity_threshold: float = 0.95

    # Rate Limiting
    rate_limit_rpm: int = 60


# Singleton instance — import this everywhere
settings = Settings()