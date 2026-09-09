# Phase 3: code walkthrough and change inventory

Status: implemented in files; execution and model-quality validation pending.
No real credentials were inspected or changed. No live model calls were made.

## Simple mental model

1. **Classifier:** What task does this look like?
2. **Profile:** Have we enabled this task for this local model, within what limits?
3. **Router:** Respect the caller's explicit choices, then choose an eligible tier.
4. **Endpoint:** Use that decision consistently for cache, generation, and headers.

Unknown is a normal classifier result. It means 'not covered by our rules', not
'objectively complex'. Even `What is 2+2?` goes premium automatically in this narrow
v1; explicit local remains available. This trades local coverage for simpler rules.

## New runtime files

### `src/gateway/router/__init__.py`
Marks the new routing package.

### `router/classifier.py`
`classify(request) -> TaskFeatures` is shared across local models. It accepts only
one user message for local candidacy. It splits an instruction from its source at
`Text:` on a new line or the first colon, strips an optional Instruction/Please
prefix, then applies full-match regular expressions to the instruction only.

For example, `Make this email more polite: Send the report.` is recognized as rewrite.
`Translate this sentence: Please summarize the meeting.` is unknown: the quoted
verb is not the actual instruction. Mixed instructions fail the narrow full match.

TaskFeatures contains task, reason, word count, instruction, source. Word count covers
the full input and is a profile operating range, not a tokenizer estimate. This
parser intentionally misses many valid paraphrases. It does not prove that source
material is simple or free of malicious instructions; evaluate those failures.

### `router/profile.py` and `router/profiles/phi4-mini.json`
Pydantic validates the JSON profile: exact model name, version, approval flag, and
per-task enabled/max_input_words settings. Unknown fields/task names fail validation.
The shipped profile enables rewrite (300 words), summary (500), extraction (250).
These are illustrative experimental limits, NOT measured Phi-4 Mini capabilities.
A missing/malformed configured profile fails startup rather than silently inventing policy.

### `router/model_router.py`
`ModelRouter.select()` returns a RoutingDecision with concrete model, provider, tier,
reason, policy fingerprint, profile status and task features. Explicit model/tier
wins over classification; conflicts and unsupported names raise RoutingError.

Automatic selection checks profile/model match, strict-vs-experimental mode, enabled
task and word range. Unknown or unapproved cases go premium. Automatic local choices
that exceed local admission guards are reconsidered for premium. Explicit choices
are not silently changed. Input-byte limits and output-token limits then apply.

Policy fingerprints hash the full profile and include classifier version/mode. Profile
edits change cache scope automatically. Bump CLASSIFIER_VERSION when changing rules.
No numerical heuristic is presented as a calibrated confidence probability.

### `providers/registry.py`
Owns provider instances. Ollama is ready without a request; premium SDK client is
created lazily only on a premium miss. Missing credentials cause 503 without fallback.
Shared premium connections close during app shutdown. No additional SDK retries.

### `cache/scope.py`
`build_cache_key()` hashes canonical JSON into an exact scope ID and request ID.
Scope includes selected provider/model/endpoint, policy, generation settings, history,
and task. For recognized rewrite/summary it includes exact source and parsed output
constraints (tone, grammar-only, bullet/sentence count/format). Only instruction
wording can then be semantically matched within that scope.

All other requests, including extraction, require exact message hashes. Streaming
is excluded from the key so eligible text responses can be reused across response
formats. A tier override promises answer provenance, not fresh generation.
No tenant field exists because authentication/tenant identities are not implemented;
do not treat this as a multi-tenant boundary.

## Modified runtime files

### `config.py`
Adds experimental/strict mode, profile path, per-tier admission/output limits and
premium timeout. Relative custom profile paths resolve against the project root.
Timeouts and threshold values are validated. Default models remain unchanged.

### `models/schemas.py`
Validates nonempty, nonblank text messages; positive max_tokens; temperature range;
and rejects unsupported extra fields. Keeps the existing response structure.
Only system/user/assistant text messages are supported. This is a deliberately limited
OpenAI-style API, not full API compatibility.

### `main.py`
Adds create_app() with injectable configuration, cache, router and providers. Resources
are created at startup rather than network clients at import time. Tests inject fake
services. Lifespan initializes Qdrant and closes provider/cache resources.

### `api/v1/chat.py`
Pipeline: select -> resolve default output budget -> build scoped key -> lookup ->
return hit or execute selected provider -> cache response. Both streaming and normal
paths receive decision.model instead of guessing from the optional request.model.

Headers expose tier, model, routing reason/policy/profile and cache status/latency.
Runtime cache lookup failures yield BYPASS; write failures do not discard answers.
Routing logs contain metadata, not prompt text or raw provider error strings.
The logger uses Python logging; configure INFO handling to see decision events.

On new streams, headers identify the selected model because actual metadata arrives
later. Stream errors after headers are sent use a sanitized SSE error event (HTTP
status can no longer change). Cancellation propagates; partial output is not cached.
Only terminal, cleanly exhausted streams are stored. Token usage is marked unknown
for streamed generations, not asserted to be free. Cached responses use zero new tokens.

### `providers/base.py`
Adds StreamChunk(content, finish_reason, model). The Protocol's generate_stream is a
regular method returning an async iterator, matching how async generators are called.
Terminal chunks let the endpoint distinguish clean completion from interrupted data.

### `providers/ollama_provider.py`
Keeps existing HTTP payload handling; streaming now uses configured timeout, checks
error records, requires a done marker, and yields terminal model/finish metadata.

### `providers/openai_provider.py`
Adds configurable endpoint/timeout/retries and explicit close(). Streaming skips
empty-choice metadata events, emits terminal model/finish metadata and closes the
stream in finally. Missing terminal completion is an error. Non-streaming empty
choices are rejected. Provider-specific parameter compatibility still needs a live test.

### `cache/semantic_cache.py`
Consumes CacheKey instead of a raw query. Exact operations skip embeddings; semantic
embedding work runs through asyncio.to_thread with an instance lock. Scope fields
cannot be overridden by caller metadata. Configuration can be injected for testing.

### `cache/vector_store.py`
Every lookup requires scope/schema filters in Qdrant, not just a similarity score.
Exact lookup uses scroll with request ID. Stable UUIDs overwrite duplicate scoped
requests; wait=True makes writes visible before subsequent requests. Legacy unscoped
entries are ignored without deletion. Exact-only entries have placeholder zero
vectors and are not searched semantically. Collection dimension remains 384.

### `cache/embedder.py`
Defers importing sentence-transformers until model loading and accepts an optional
model name. Type annotations are postponed so offline imports do not require the
heavy model library to initialize.

## New tests

- `tests/conftest.py`: fake cache/provider/registry and deterministic test settings.
- `tests/test_routing.py`: recognition, quote traps, overrides, conflicts, strict mode,
  model/profile mismatch, disabled tasks, limits, request validation.
- `tests/test_cache.py`: scope boundaries, source/output constraints, exact extraction,
  legacy exclusion, actual in-memory Qdrant filtering, embedding bypass.
- `tests/test_chat.py`: normal/stream cache flows, correct selected model, override
  isolation, failures, incomplete/cancelled stream behavior, missing premium key.
- `tests/test_providers.py`: mocked adapter stream metadata and Ollama payloads.
- `tests/test_evaluation.py`: dataset schema/splits, dry-run no-provider behavior,
  ungraded results not counted as quality passes.

Tests are written, not run. In-memory Qdrant is useful but does not replace a live
server smoke test. No provider-quality claims follow from passing software tests.

## New evaluation files

- `evaluation/requests.jsonl`: 30 synthetic requests (15 dev / 15 test), task categories
  and human grading rubrics. This is a starter pilot, not a representative benchmark.
- `scripts/evaluate_routing.py`: dry-run default; explicit paid opt-in; bounded paired
  calls; zero retries; stop on first error; incremental results; new output directory
  per invocation. Includes model/profile snapshot and deployment notes.
- `scripts/report_evaluation.py`: compare always-local, always-premium and recorded
  router decisions using the same sampled responses. Shows quality only with grades,
  local failure rate, sampled latency, premium usage and optional price estimates.

The evaluation pays for both answers. The report's routing cost is a counterfactual
estimate, not the amount spent on the evaluation. Errors are reported separately;
quality figures use fully graded successful pairs, which must be disclosed alongside
failure counts. A 256-token cap may truncate some answers; judge under that setting
or repeat a separately budgeted run. Real deployment budgets can differ from the pilot.

## Other modified files

- `README.md`: setup, contracts, commands, limitations and evaluation workflow.
- `.env.example`: new settings with explanations; actual `.env` is untouched.
- `pyproject.toml`: restrict default pytest collection to tests/; asyncio mode; Ruff target.
  No new package dependency or lockfile change.
- `.gitignore`: ignore test artifacts and local evaluation outputs.

The original blueprint, enums, SSE formatting helper, manual embedding experiment,
lockfile, scratch file and .env were not changed. This document records the refined
Phase 3 design rather than rewriting the historical blueprint.

## Changing local models later

1. Run public representative evaluation requests against the new deployment.
2. Copy the small JSON profile, change exact model name/version and measured ranges.
3. Leave validated=false until you approve quality; record hardware/quantization/digest.
4. Set OLLAMA_MODEL and ROUTING_PROFILE_PATH together, restart the gateway.
5. Re-run software checks and held-out evaluation. Replacing a model behind the same
   alias still requires re-evaluation and a profile version bump to invalidate cache.

The classifier normally stays unchanged. Add a new task category only when you want
broader coverage and have evidence to justify it. Authentication, resilience, model-native
context accounting, TTLs, tenant boundaries and production observability remain later work.
