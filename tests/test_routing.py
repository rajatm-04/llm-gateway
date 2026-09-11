import pytest
from conftest import request_for
from pydantic import ValidationError

from gateway.models.schemas import ChatCompletionRequest
from gateway.router.classifier import classify
from gateway.router.model_router import ModelRouter, RoutingError
from gateway.router.profile import load_profile


@pytest.mark.parametrize("text,task", [
    ("Make this email more polite: Send the report.", "rewrite"),
    ("Instruction: Summarize the following text in three bullets.\nText: The launch was delayed.", "summary"),
    ("Extract the invoice number and total: Invoice: INV-2; Total: $50", "extraction"),
    ('Translate this sentence into French: "Please summarize the meeting."', "unknown"),
    ("Summarize this text and assess legal risks: A contract.", "unknown"),
    ("Is this proof valid?", "unknown"),
    ("Summarize this text:", "unknown"),
    ("Review this code for security: def summarize(): pass", "unknown"),
])
def test_task_recognition(text, task):
    assert classify(request_for(text)).task_type == task


def test_system_and_conversation_unknown():
    request = ChatCompletionRequest(messages=[
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Summarize this text: One sentence."},
    ])
    assert classify(request).task_type == "unknown"


def test_profile_routes_local_candidates_and_unknown_premium(config):
    router = ModelRouter(config)
    decision = router.select(request_for())
    assert decision.tier == "local"
    assert router.select(request_for("Prove the theorem.")).tier == "premium"


@pytest.mark.parametrize("model,tier", [("phi4-mini", "local"), ("gpt-5.6-sol", "premium")])
def test_explicit_model(config, model, tier):
    assert ModelRouter(config).select(request_for(model=model)).tier == tier


@pytest.mark.parametrize("model,tier", [("phi4-mini", "premium"), ("gpt-5.6-sol", "local")])
def test_conflict(config, model, tier):
    with pytest.raises(RoutingError):
        ModelRouter(config).select(request_for(model=model), tier)


@pytest.mark.parametrize("tier", ["", "unknown", "super-premium"])
def test_bad_tier(config, tier):
    with pytest.raises(RoutingError):
        ModelRouter(config).select(request_for(), tier)


def test_unsupported_model_and_normalized_tier(config):
    router = ModelRouter(config)
    with pytest.raises(RoutingError):
        router.select(request_for(model="gpt-made-up"))
    assert router.select(request_for(), " PREMIUM ").tier == "premium"


def test_profile_model_change(config):
    profile = load_profile(config.routing_profile_path)
    assert ModelRouter(config, profile).select(request_for()).tier == "local"
    config.ollama_model = "different-model"
    router = ModelRouter(config, profile)
    assert router.select(request_for()).reason == "local_profile_model_mismatch"
    assert router.select(request_for(), "local").model == "different-model"


def test_task_limits_and_disabled_task(config):
    router = ModelRouter(config)
    assert router.select(request_for("Make this email more polite: " + "word " * 301)).tier == "premium"
    profile = router.profile.model_copy(deep=True)
    profile.tasks["rewrite"] = profile.tasks["rewrite"].model_copy(update={"enabled": False})
    assert ModelRouter(config, profile).select(request_for()).reason == "local_task_disabled"
    assert ModelRouter(config, profile).policy_version != router.policy_version


def test_admission_limits(config):
    router = ModelRouter(config)
    assert router.select(request_for(max_tokens=3000)).tier == "premium"
    with pytest.raises(RoutingError):
        router.select(request_for(max_tokens=3000), "local")
    config.premium_max_input_bytes = 10
    with pytest.raises(RoutingError) as error:
        ModelRouter(config).select(request_for("A very long unfamiliar task."))
    assert error.value.status_code == 413


@pytest.mark.parametrize("body", [
    {"messages": []},
    {"messages": [{"role": "user", "content": " "}]},
    {"messages": [{"role": "user", "content": "Hi"}], "tools": []},
    {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 0},
])
def test_schema_rejects_invalid_or_unsupported_requests(body):
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(body)
