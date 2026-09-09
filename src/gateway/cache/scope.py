"""Exact cache boundaries surrounding an optional semantic instruction lookup."""

import hashlib
import json
import re
from dataclasses import dataclass

from gateway.config import Settings
from gateway.models.schemas import ChatCompletionRequest
from gateway.router.model_router import RoutingDecision


def stable_hash(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    scope_id: str
    request_id: str
    query: str
    exact: bool

    @property
    def filters(self) -> dict[str, str]:
        fields = {"cache_schema": "routing-v1", "scope_id": self.scope_id}
        if self.exact:
            fields["request_id"] = self.request_id
        return fields

    @property
    def fields(self) -> dict[str, str]:
        return {**self.filters, "request_id": self.request_id}


def instruction_constraints(instruction: str) -> dict:
    """Extract constraints from the closed grammar in classifier.py, not arbitrary prose."""
    text = instruction.lower()
    for word, digit in {"two": "2", "three": "3", "four": "4", "five": "5"}.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    size = re.search(r"\bin ([1-5]) (bullet points|bullets|sentences)\b", text)
    tone = next((tone for tone in ("polite", "professional", "friendly") if tone in text), None)
    return {
        "tone": tone,
        "grammar_only": "grammar" in text,
        "count": int(size.group(1)) if size else None,
        "format": ("sentences" if size.group(2) == "sentences" else "bullets") if size else None,
    }


def build_cache_key(request: ChatCompletionRequest, decision: RoutingDecision, config: Settings) -> CacheKey:
    messages = [m.model_dump() for m in request.messages]
    features = decision.features
    # Only recognized rewrite/summary instructions can be matched semantically.
    # Source text remains EXACT, so similar documents with changed facts cannot collide.
    semantic = features.task_type in {"rewrite", "summary"}
    scope = {
        "provider": decision.provider,
        "endpoint": (config.ollama_host if decision.tier == "local" else config.openai_base_url).rstrip("/"),
        "model": decision.model,
        "tier": decision.tier,
        "policy": decision.policy_version,
        "embedding_model": config.embedding_model if semantic else None,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "task": features.task_type,
        "mode": "semantic_instruction" if semantic else "exact_request",
        "history": messages[:-1],
        "last_role": messages[-1]["role"],
        "source": features.source if semantic else None,
        # Keep output constraints exact even when instruction wording is similar.
        "output_constraints": instruction_constraints(features.instruction) if semantic else None,
    }
    return CacheKey(
        scope_id=stable_hash(scope),
        request_id=stable_hash(messages),
        query=features.instruction if semantic else json.dumps(messages, ensure_ascii=False),
        exact=not semantic,
    )
