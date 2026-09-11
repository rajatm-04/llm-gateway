"""Resolve caller choices, then apply a model profile to recognized tasks."""

import hashlib
from dataclasses import dataclass

from gateway.config import Settings
from gateway.models.schemas import ChatCompletionRequest
from gateway.router.classifier import CLASSIFIER_VERSION, TaskFeatures, classify
from gateway.router.profile import ModelProfile, load_profile


class RoutingError(ValueError):
    def __init__(self, detail: str, status_code: int = 422):
        super().__init__(detail)
        self.status_code = status_code


@dataclass(frozen=True)
class RoutingDecision:
    tier: str
    model: str
    provider: str
    reason: str
    policy_version: str
    features: TaskFeatures


class ModelRouter:
    def __init__(self, config: Settings, profile: ModelProfile | None = None):
        self.config = config
        self.profile = profile if profile is not None else load_profile(config.routing_profile_path)
        if config.ollama_model == config.openai_model:
            raise ValueError("Local and premium model names must be different")
        fingerprint = hashlib.sha256(self.profile.model_dump_json().encode()).hexdigest()[:12]
        self.policy_version = f"{CLASSIFIER_VERSION}:{fingerprint}"

    def select(self, request: ChatCompletionRequest, tier_override: str | None = None) -> RoutingDecision:
        features = classify(request)
        tier = None
        if tier_override is not None:
            tier = tier_override.strip().lower()
            if tier not in {"local", "premium"}:
                raise RoutingError("X-Model-Tier must be 'local' or 'premium'")

        models = {self.config.ollama_model: "local", self.config.openai_model: "premium"}
        if request.model is not None:
            if request.model not in models:
                raise RoutingError("Unsupported model; use one of the two configured model names")
            model_tier = models[request.model]
            if tier is not None and tier != model_tier:
                raise RoutingError("Explicit model conflicts with X-Model-Tier")
            tier, reason = model_tier, "explicit_model"
        elif tier is not None:
            reason = "explicit_tier"
        else:
            tier, reason = self._automatic(features)

        # Deployment admission limits, NOT native tokenizer/context claims.
        input_bytes = sum(len(m.content.encode("utf-8")) for m in request.messages)
        automatic = request.model is None and tier_override is None
        if automatic and tier == "local" and (
            input_bytes > self.config.local_max_input_bytes
            or (request.max_tokens is not None and request.max_tokens > self.config.local_max_output_tokens)
        ):
            tier, reason = "premium", "outside_local_admission_limits"
        input_limit = (self.config.local_max_input_bytes if tier == "local"
                       else self.config.premium_max_input_bytes)
        output_limit = (self.config.local_max_output_tokens if tier == "local"
                        else self.config.premium_max_output_tokens)
        if input_bytes > input_limit:
            raise RoutingError("Request exceeds selected model's configured input byte limit", 413)
        if request.max_tokens is not None and request.max_tokens > output_limit:
            raise RoutingError("max_tokens exceeds selected model's configured output limit")

        return RoutingDecision(
            tier=tier,
            model=self.config.ollama_model if tier == "local" else self.config.openai_model,
            provider="ollama" if tier == "local" else "openai-compatible",
            reason=reason,
            policy_version=self.policy_version,
            features=features,
        )

    def _automatic(self, features: TaskFeatures) -> tuple[str, str]:
        if self.profile.model != self.config.ollama_model:
            return "premium", "local_profile_model_mismatch"
        policy = self.profile.tasks.get(features.task_type)
        if policy is None or not policy.enabled:
            return "premium", features.reason if policy is None else "local_task_disabled"
        if features.input_words > policy.max_input_words:
            return "premium", "outside_local_test_range"
        return "local", f"{features.task_type}_profile_match"
