from pathlib import Path

import pytest

from gateway.config import Settings


@pytest.fixture
def config():
    return Settings(
        _env_file=None,
        openai_api_key="",
        ollama_model="phi4-mini",
        openai_model="gpt-5.6-sol",
        routing_profile_path=Path(__file__).resolve().parents[1] / "src/gateway/router/profiles/phi4-mini.json",
        ollama_host="http://local.invalid",
        openai_base_url="https://premium.invalid/v1",
        local_max_input_bytes=12000,
        premium_max_input_bytes=120000,
        local_max_output_tokens=2048,
        premium_max_output_tokens=4096,
    )
