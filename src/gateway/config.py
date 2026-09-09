"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the development .env relative to the source tree, not the working directory.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Gateway configuration.

    All values can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Providers
    openai_api_key: str = ""
    openai_base_url: str = "https://api.experientiallabs.ai/v1"
    openai_model: str = "gpt-5.6-sol"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "phi4-mini"
    ollama_timeout: float = 120.0

    # Qdrant Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"

    # Cache Settings
    cache_similarity_threshold: float = 0.95

    # Rate Limiting
    rate_limit_rpm: int = 60

    # Circuit Breaker
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 30.0
    circuit_half_open_max_calls: int = 3

    # Provider Retry
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 8.0
    retry_jitter: float = 0.25


# Singleton instance — import this everywhere
settings = Settings()