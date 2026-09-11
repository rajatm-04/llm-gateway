"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import Field, field_validator
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
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout: float = Field(default=120.0, gt=0)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "phi4-mini"
    ollama_timeout: float = Field(default=120.0, gt=0)

    # Qdrant Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"

    # Cache Settings
    cache_similarity_threshold: float = Field(default=0.95, ge=0, le=1)

    # Routing profiles describe the current local-model policy.
    routing_profile_path: Path = Path(__file__).parent / "router" / "profiles" / "phi4-mini.json"
    # Conservative deployment admission guards, NOT exact context-token limits.
    local_max_input_bytes: int = Field(default=12000, gt=0)
    premium_max_input_bytes: int = Field(default=120000, gt=0)
    local_max_output_tokens: int = Field(default=2048, gt=0)
    premium_max_output_tokens: int = Field(default=4096, gt=0)
    openai_timeout: float = Field(default=120.0, gt=0)

    @field_validator("routing_profile_path")
    @classmethod
    def resolve_profile_path(cls, value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else ENV_FILE.parent / value

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