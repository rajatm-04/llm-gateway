import json
import logging

from conftest import FakeCache, FakeRegistry, request_for
from fastapi.testclient import TestClient

from gateway.main import create_app


def completion_records(caplog):
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "gateway.request_logging"
        and '"event":"request_completed"' in record.message
    ]


def test_completion_log_is_safe_and_request_id_is_propagated(config, caplog):
    cache, providers = FakeCache(), FakeRegistry()
    app = create_app(config=config, cache=cache, providers=providers)
    prompt = "do not log this prompt"

    with TestClient(app) as client, caplog.at_level(logging.INFO):
        response = client.post(
            "/v1/chat/completions",
            json=request_for(prompt).model_dump(),
            headers={
                "X-Request-ID": "request-123",
                "Authorization": "Bearer do-not-log",
                "X-Model-Tier": "local",
            },
        )

    records = completion_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert response.headers["x-request-id"] == "request-123"
    assert record["event"] == "request_completed"
    assert record["request_id"] == "request-123"
    assert record["method"] == "POST"
    assert record["path"] == "/v1/chat/completions"
    assert record["status_code"] == 200
    assert record["cache_status"] == "MISS"
    assert record["model_tier"] == "local"
    assert record["selected_model"] == "phi4-mini"
    assert record["used_model"] == "phi4-mini"
    assert record["provider_outcome"] == "success"
    assert record["stream"] is False
    assert prompt not in caplog.text
    assert "do-not-log" not in caplog.text


def test_stream_completion_log_records_provider_error_without_exception_text(config, caplog):
    cache, providers = FakeCache(), FakeRegistry()
    providers.local.fail_stream = True
    app = create_app(config=config, cache=cache, providers=providers)

    with TestClient(app) as client, caplog.at_level(logging.INFO):
        response = client.post(
            "/v1/chat/completions",
            json=request_for("secret stream prompt", stream=True).model_dump(),
            headers={"X-Model-Tier": "local"},
        )

    records = completion_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert response.status_code == 200
    assert record["stream"] is True
    assert record["provider_outcome"] == "error"
    assert record["error_category"] == "provider_stream_error"
    assert "fake upstream failure" not in caplog.text
    assert "secret stream prompt" not in caplog.text


def test_cache_lookup_failure_is_a_safe_error_category(config, caplog):
    cache, providers = FakeCache(), FakeRegistry()
    cache.fail_lookup = True
    app = create_app(config=config, cache=cache, providers=providers)

    with TestClient(app) as client, caplog.at_level(logging.INFO):
        response = client.post(
            "/v1/chat/completions",
            json=request_for().model_dump(),
        )

    record = completion_records(caplog)[0]
    assert response.status_code == 200
    assert record["cache_status"] == "BYPASS"
    assert record["error_category"] == "cache_lookup_failure"
    assert record["provider_outcome"] == "success"
    assert "fake cache unavailable" not in caplog.text


def test_rate_limit_completion_log_has_safe_category_and_request_id(config, caplog):
    config = config.model_copy(update={"rate_limit_rpm": 1})
    app = create_app(config=config, cache=FakeCache(), providers=FakeRegistry())

    with TestClient(app) as client, caplog.at_level(logging.INFO):
        client.post(
            "/v1/chat/completions",
            json=request_for().model_dump(),
            headers={"X-Request-ID": "rate-limit-request", "X-Model-Tier": "local"},
        )
        limited = client.post(
            "/v1/chat/completions",
            json=request_for().model_dump(),
            headers={"X-Request-ID": "limited-request", "X-Model-Tier": "local"},
        )

    records = completion_records(caplog)
    limited_record = next(record for record in records if record["request_id"] == "limited-request")
    assert limited.status_code == 429
    assert limited.headers["x-request-id"] == "limited-request"
    assert limited_record["error_category"] == "rate_limited"
    assert limited_record["status_code"] == 429
