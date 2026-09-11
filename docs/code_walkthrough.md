# Gateway code walkthrough

The gateway has a deliberately small runtime path:

1. `gateway.models.schemas` accepts text-only chat requests and rejects
   unsupported fields.
2. `gateway.router.classifier` recognizes the narrow rewrite, summary, and
   extraction patterns. `gateway.router.profile` loads the local model's
   enabled tasks and word limits.
3. `gateway.router.model_router.ModelRouter` honors explicit model/tier
   choices, applies admission limits, and otherwise chooses local only when
   the profile matches. Unknown or out-of-profile tasks use premium. The
   profile is an evolving routing configuration rather than a quality
   admission gate.
4. `gateway.cache.scope` builds a provider/model/policy/options/history scope.
   `SemanticCache` uses semantic instruction matching only for recognized
   rewrite/summary requests and exact lookup for other requests.
5. `gateway.providers.registry` owns the Ollama and lazy OpenAI-compatible
   clients. `gateway.resilience.retry` and `CircuitBreaker` protect each
   selected provider. Premium failures return a sanitized 503; clients must
   explicitly select local generation when that is desired.
6. `gateway.api.v1.chat` uses the same routing decision for lookup, generation,
   response headers, and cache writes. Streaming uses a basic async SSE
   generator and caches only after a terminal event and clean completion.
7. `gateway.main.create_app` wires FastAPI, lifecycle-managed providers,
   cache, breakers, static assets, and the public health/UI configuration
   allowlist.

Errors returned to clients are safe summaries and never raw provider
exceptions. The runtime keeps operational logging limited to ordinary safe
warnings.

## Optional experiments

Quality-evaluation datasets and policy are under `experiments/evaluation/`.
`experiments/evaluate_routing.py` performs a dry-run unless paid calls are
explicitly enabled, and `experiments/report_evaluation.py` summarizes paired
results and human grades. These tools are not imported by the gateway.

## Tests

The focused suite covers deterministic routing, cache scope and hit/miss
behavior, provider failures, retries, circuit breakers, explicit local and
premium choices, and successful or failed SSE streams. Run it with:

```bash
uv run --extra dev pytest
uv run ruff check src/gateway tests experiments
```
