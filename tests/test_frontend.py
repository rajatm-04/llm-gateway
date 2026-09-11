"""Offline checks for frontend delivery; JavaScript behavior needs a browser check."""

import pytest
from conftest import FakeCache, FakeRegistry
from fastapi.testclient import TestClient

from gateway.main import create_app


@pytest.fixture
def frontend(config):
    config.openai_api_key = "test-only-secret-do-not-expose"
    cache, providers = FakeCache(), FakeRegistry()
    with TestClient(create_app(config=config, cache=cache, providers=providers)) as client:
        yield client, cache, providers


def test_frontend_page_and_assets_do_not_generate(frontend):
    client, cache, providers = frontend
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="request-form"' in response.text
    assert 'src="/assets/playground.js"' in response.text
    assert 'href="/assets/playground.css"' in response.text
    assert response.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    for path, content_type in (("/assets/playground.css", "text/css"),
                               ("/assets/playground.js", "javascript")):
        asset = client.get(path)
        assert asset.status_code == 200
        assert content_type in asset.headers["content-type"]
    assert not cache.lookups
    assert not providers.local.calls
    assert not providers.premium.calls


def test_ui_config_is_an_explicit_public_allowlist(frontend, config):
    client, cache, providers = frontend
    response = client.get("/ui/config")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "local_model": config.ollama_model,
        "premium_model": config.openai_model,
        "local_max_output_tokens": config.local_max_output_tokens,
        "premium_max_output_tokens": config.premium_max_output_tokens,
    }
    assert config.openai_api_key not in response.text
    assert config.openai_base_url not in response.text
    assert config.ollama_host not in response.text
    assert not cache.lookups
    assert not providers.local.calls
    assert not providers.premium.calls


def test_ui_config_uses_injected_model_names(config):
    config.ollama_model = "custom-local"
    config.openai_model = "custom-premium"
    with TestClient(create_app(config=config, cache=FakeCache(), providers=FakeRegistry())) as client:
        assert client.get("/ui/config").json()["local_model"] == "custom-local"
        assert client.get("/ui/config").json()["premium_model"] == "custom-premium"


def test_static_mount_does_not_expose_project_files(frontend):
    client, _, _ = frontend
    for path in ("/assets/.env", "/assets/config.py", "/assets/%2e%2e/config.py"):
        assert client.get(path).status_code == 404
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/chat/completions" in paths
    assert "/ui/config" not in paths


def test_browser_shaped_request_still_uses_existing_chat_contract(frontend):
    client, cache, providers = frontend
    body = {
        "messages": [{"role": "user", "content": "Make this email more polite: Send the report today."}],
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
    }
    headers = {"X-Model-Tier": "local"}
    first = client.post("/v1/chat/completions", json=body, headers=headers)
    second = client.post("/v1/chat/completions", json=body, headers=headers)
    assert first.status_code == second.status_code == 200
    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "HIT"
    assert providers.local.calls[0]["max_tokens"] == 128
    assert len(providers.local.calls) == 1
    assert len(cache.writes) == 1
