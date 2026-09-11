# LLM Gateway

FastAPI gateway for one local Ollama model and one premium
OpenAI-compatible provider. It combines deterministic profile-based task
routing with scoped semantic/exact caching, retries, circuit breakers, Qdrant,
and basic Server-Sent Events (SSE). Python 3.13+, managed with `uv`.

## Request flow

```text
validate text request
  -> classify task and consult the local profile
  -> honor explicit model/tier or choose a tier
  -> build an exact cache scope
  -> return a hit or call the selected provider
  -> cache only complete responses
```

The default local model is `phi4-mini`; the premium model is `gpt-5.6-sol`
at the configured OpenAI-compatible endpoint. Both names and all provider
endpoints are configurable. A missing model/tier is routed automatically:
narrow rewrite, summary, and extraction tasks within the local profile use
Ollama, while unknown or out-of-profile tasks use premium. The profile is a
live routing configuration, not a validation or quality gate.

## Setup and checks

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run ruff check src/gateway tests experiments
```

Tests use fake providers and in-memory services; they do not require Ollama,
Qdrant, embeddings, credentials, or paid calls.

## Run the gateway

1. Review `.env.example` and create a local `.env` if needed.
2. Start Ollama and install the configured local model.
3. Start Qdrant, for example:
   `docker run --rm -p 6333:6333 qdrant/qdrant`
4. Run `uv run uvicorn gateway.main:app --reload`.
5. Open `http://localhost:8000` for the playground or `/docs` for the API.

Example automatic local request:

```bash
curl -i http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Make this email more polite: Send the report today."}],"max_tokens":128}'
```

Use `X-Model-Tier: local` or `X-Model-Tier: premium` to make the provider
choice explicit. An explicit model may be supplied instead. Premium failures
and missing credentials return a sanitized `503`; the gateway never silently
switches a premium request to local. Request local explicitly when that is the
desired behavior.

## Routing contract

- Only the two configured model names are accepted.
- `X-Model-Tier` accepts `local` or `premium`, case-insensitively.
- Explicit model/tier conflicts and unknown values return `422`.
- Unknown, multi-turn, or unsupported tasks route to premium automatically.
- Explicit local routing remains available for any task within local limits.
- The profile controls enabled task types and word ranges; changing it changes
  the routing policy fingerprint and cache scope.
- Configured byte/output limits are admission guards, not tokenizer claims.
- Omitted `max_tokens` uses the selected tier's configured output budget.

Responses include model, routing reason/policy, cache status, and cache
latency headers. The custom SSE format is
`data: {"content": "...", "finish_reason": ...}` followed by `data: [DONE]`.
Provider stream errors are sanitized SSE events. Partial or failed streams are
never cached; provider generators are closed normally.

## Cache behavior

Cache scope includes provider endpoint, selected model/tier, routing policy,
generation options, conversation history, task, and output constraints.
Recognized rewrite/summary requests match instruction wording semantically but
require the source text to match exactly. Extraction and unknown requests use
exact whole-conversation hashes. Cache failures bypass caching rather than
discarding a generated answer.

## Optional quality evaluation

Evaluation datasets, grading policy, and scripts are intentionally outside the
core runtime under [`experiments/`](experiments/README.md). The runner is a dry
run by default:

```bash
uv run python experiments/evaluate_routing.py --split dev --limit 10
```

Live calls require the explicit `--allow-paid-calls` flag. Use only public or
synthetic prompts and keep generated results under the ignored experiments
results directory.

## Browser playground and limitations

The root URL serves a small dependency-free playground. It supports local,
automatic, and premium choices, streaming, model selection, and output limits.
It does not persist browser history or make automatic model calls.

This is a single-trust-domain development gateway. It has no authentication,
tenant isolation, privacy policy enforcement, or budget service. Do not expose
it publicly or use it for separate tenants' private data. Pattern routing is
deliberately narrow and does not guarantee model quality.

See [the code walkthrough](docs/code_walkthrough.md) and
[frontend smoke test](docs/frontend_smoke_test.md) for implementation details.
