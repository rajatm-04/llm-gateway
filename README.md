# LLM Gateway

FastAPI gateway with Ollama / OpenAI-compatible providers, scoped Qdrant caching,
and **experimental task-aware routing**. Python 3.13+, managed with `uv`.

> Phase 3 implementation is written but has not been executed/verified in the
> filesystem-only assistant environment. Run the offline tests before live use.
> The local profile is NOT quality-validated. No cost/latency savings are claimed.

## Request flow

```text
Validate supported text request + explicit model/tier
  -> recognize task + consult local-model profile
  -> resolve concrete model and generation budget
  -> exact cache scope filter
      -> eligible HIT: return cached response
      -> MISS: selected provider -> complete response/stream -> cache
```

Defaults: `phi4-mini` via Ollama; `gpt-5.6-sol` via the configured Experiential Labs
OpenAI-compatible endpoint. These names remain configurable; availability is not verified.
No model/tier means automatic routing (previously omitted model meant Ollama).

## First: offline checks

From the repository root:

```bash
uv sync --extra dev
uv run pytest tests/ -q
uv run ruff check src/gateway tests scripts
```

Tests use fake providers, mocked HTTP, and Qdrant's in-memory client. They do not
need running Ollama/Qdrant, embedding downloads, or paid API calls. The root-level
`test_embeddings.py` is an existing manual experiment, excluded from default pytest collection.
Dependency installation itself can download packages. No dependencies were added for routing.

## Run the gateway

1. Review `.env.example`. Keep your existing `.env`; do not overwrite it. New routing
   settings have defaults, so adding them is optional.
2. Start Ollama and install `phi4-mini` if needed (`ollama pull phi4-mini`).
3. Start Qdrant if not already running, for example:
   `docker run --rm -p 6333:6333 qdrant/qdrant` (temporary data unless you mount storage).
4. Run `uv run uvicorn gateway.main:app --reload`.
5. Open `http://localhost:8000` for the browser playground, or `/docs` for the API reference.

Qdrant is required at startup. The first semantic request can download/load MiniLM.
Localhost inside Docker may not refer to your host machine.

Example automatic local candidate:

```bash
curl -i http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Make this email more polite: Send the report today."}],"max_tokens":128}'
```

Repeat for a cache hit. Add `-H 'X-Model-Tier: premium'` to force premium provenance;
it must not reuse the local entry. Add `"stream":true` to request SSE.
Premium misses require your configured API key and can cost money.

## Browser playground

The root URL serves a small HTML/CSS/JavaScript interface from FastAPI. No Node.js,
frontend build step, CDN, or new dependency is required. The former root welcome
JSON is replaced by HTML; the chat and health endpoints are unchanged.

- Single-prompt requests with Local / Auto / Premium routing, optional explicit
  model, output ceiling, temperature, and streaming controls.
- Local is the UI default to avoid accidental paid calls. This does not change
  automatic routing for API requests that omit model/tier.
- Requests that could use premium require a per-send acknowledgement. This is a
  UI safeguard, not a backend budget/authentication control. Cache misses may cost.
- Real response headers show routing, cache status, model provenance and policy.
  Browser elapsed time is distinct from cache lookup latency. Stream usage is unknown.
- Model text is displayed without executing HTML or rendering Markdown. Partial
  streams/errors are identified; Stop waiting does not guarantee upstream cancellation.
- No automatic requests to models, retries, or browser-persisted conversation history.
  Prompts and answers may still be stored in Qdrant. Use non-sensitive data locally.

The frontend and its serving tests are written, not executed by the assistant.
See [browser smoke-test walkthrough](docs/frontend_smoke_test.md) for local checks
and the bounded live sequence that completes the remaining Phase 3 checkpoint.

## Routing contract

- Only the two exact configured model names are supported. No GPT-prefix guessing.
- `X-Model-Tier` accepts `local` / `premium` (case-insensitive).
- Explicit model and conflicting tier: 422. Unknown model/tier: 422.
- Unknown task, system prompt, or multi-turn request: premium in automatic mode.
- Candidate local tasks: narrow English rewrite, summary, and extraction patterns.
- Explicit local is allowed even for an unknown task, subject to admission guards.
- `ROUTING_MODE=strict` disables automatic local routing unless the matching profile
  has `validated: true`. That flag is a human approval, not an automated guarantee.
- Changing `OLLAMA_MODEL` without a matching profile disables automatic local routing.
- No silent provider fallback. Missing premium credentials on a miss returns 503.
- Unsupported request fields such as tools and response_format are rejected, not ignored.
- Configured byte/output limits are admission guards, not model-native context calculations.
- Omitted max_tokens resolves to the selected tier's configured output limit.

Response headers include `X-Model-Tier`, `X-Model-Used`, `X-Routing-Reason`,
`X-Routing-Policy`, `X-Routing-Profile`, `X-Cache`, and `X-Cache-Latency-Ms`.
On a new stream, `X-Model-Used` means the selected model and
`X-Model-Used-Source: selected` makes that explicit. Headers cannot change after
streaming starts. Cached metadata records the provider-reported model when available.
SSE retains the existing custom `data: {"content":...,"finish_reason":...}` format;
it is not the full OpenAI streaming chunk schema.

## Cache behavior

Exact scope includes provider endpoint, selected model/tier, routing policy fingerprint,
generation options, system/history context, task, and supported output constraints.
For recognized rewrite/summary requests, source text must also match exactly; only
instruction wording is searched semantically. Extraction and unknown requests use
exact whole-conversation hashes, skipping embedding inference.

Old entries without the new scope fields are ignored, not deleted. Repeated exact
writes replace the same scoped entry. Cache hits report zero new provider tokens.
Completed streams are cached only after a terminal event and clean completion;
errors/cancellation do not cache partial output. Length-limited completion keeps
`finish_reason: length`. Stream usage is currently unknown, recorded as such.
Runtime cache failures bypass caching rather than discard a generated answer.

## Evaluate model quality (separate from software tests)

The bundled **30-case synthetic pilot** has 15 development and 15 held-out test cases.
It is intentionally smaller than a representative 60–100+ case evaluation; expand it
with your actual public/non-sensitive tasks. Never tune rules on held-out outcomes.

For mixed-task quality evaluation, explicitly use a 1,024-output-token ceiling.
The script still defaults to 256 for backwards-compatible smoke tests; a higher
ceiling permits greater cost/latency but does not force longer answers. Keep runs
with different budgets separate. A `length` finish reason requires completeness review.

Preview (no calls, no result files):

```bash
uv run python scripts/evaluate_routing.py --split dev --limit 10 --max-output-tokens 1024
```

After reviewing prompts and provider pricing, opt in to live calls:

```bash
uv run python scripts/evaluate_routing.py --split dev --limit 10 \
  --max-premium-calls 10 --max-output-tokens 1024 --allow-paid-calls
```

This makes at most ten local and ten premium generation attempts, with zero automatic
SDK retries. It stops on the first error and saves incremental results. Output-token
limits are requested from the provider, not a guaranteed dollar cap; input billing,
reasoning-token rules, and provider behavior matter. No gateway/Qdrant is needed.
Each invocation is a new paid run; there is no automatic resume or rerun.

Results go into a new ignored `evaluation/results/<timestamp>/` directory:
- `manifest.json`: models, profile snapshot, limits, deployment notes.
- `results.jsonl`: both answers, rubrics, routing decisions, latency and usage/errors.
- `grades.csv`: use `yes`, `no`, or `review` under [substantive-v1](evaluation/grading_policy.md).
  Blank means ungraded. Premium is the comparison benchmark, not an automatic pass.

Generate a comparison:

```bash
uv run python scripts/report_evaluation.py evaluation/results/YOUR_RUN_DIRECTORY
```

Optional `--premium-input-per-million` and `--premium-output-per-million` use prices
you supply; these are estimates, not invoices. Reports compare always-local,
always-premium, and the recorded routing policy on paired responses. Ungraded cases
are not treated as passes; incomplete/error cases are reported separately. Latencies
are sequential samples, not concurrent gateway benchmarks. Record hardware,
quantization, model digest and generation settings using `--deployment-notes`.

### Grading and current evaluation direction

Judge factual accuracy, material meaning preservation and task completion, not style
preferences. Natural paraphrases, verbosity and greetings are not failures by themselves.
Document genuinely ambiguous cases as `review`. Reports separately show unresolved
cases, both/neither acceptable pairs and clear local-selection losses against premium.
The standard was refined after development outputs were inspected; retrospective grades
are not independent validation. Raw outputs and generation manifests remain unchanged.
See [grading policy and audit notes](evaluation/grading_policy.md) for examples.

Tone rewriting stays enabled experimentally. **Do not expand its test set now.**
Further evaluation should prioritize extraction correctness, summary facts/omissions,
required reasoning steps and live cache/routing behavior. The existing rewrite batch
is retained as diagnostic evidence, not a representative production failure estimate.
No new paid run is needed to regrade the saved outputs.

### Focused rewrite development batch (existing diagnostic set)

`evaluation/rewrite_dev.jsonl` adds ten diagnostic development cases: six tone
rewrites, two grammar corrections, one natural paraphrase, and one mixed legal task.
They target preserving uncertainty, negation, conditions, amounts, deadlines and
avoiding invented commitments. This is a deliberately challenging development batch,
not a representative traffic sample or a held-out benchmark. The original 30-case
file and its held-out split remain unchanged. These prompts are NOT model-quality labels.

Preview all ten without calls:

```powershell
uv run python scripts/evaluate_routing.py --dataset evaluation/rewrite_dev.jsonl --split dev --limit 10 --max-output-tokens 1024
```

After offline checks, a live batch attempts at most ten local and ten premium calls:

```powershell
uv run python scripts/evaluate_routing.py --dataset evaluation/rewrite_dev.jsonl --split dev --limit 10 --max-premium-calls 10 --max-output-tokens 1024 --allow-paid-calls
```

Use `--case-ids rw01 rw02` with `--limit 2 --max-premium-calls 2` for a smaller batch.
Use substantive-v1 with the prewritten task rubric for both models; compare acceptability
against premium, not exact wording. Review is excluded from binary quality rates and
reported separately; neither model is assumed correct.
Generic greetings/placeholders are acceptable unless they change meaning; new promises,
changed conditions, lost negation, or invented dates fail. Review completeness separately
when a response is truncated. Keep the profile experimental and unchanged until the
batch has been reviewed; do not tune on held-out outcomes.

The targeted 1,024-token rerun of d13/d14 was assistant-reviewed: local failed both
rubrics and premium passed both. Grades were recorded in that run's grades.csv;
original raw outputs/manifests were preserved. This two-case diagnosis is not a
replacement for a uniformly budgeted full evaluation.

## Limitations / next phases

This is a single-trust-domain development gateway: no authentication, tenant isolation,
privacy policy enforcement, rate limiting, circuit breakers, TTL eviction, or budget
service yet. Do not expose it publicly or use it for separate tenants' private data.
Pattern recognition is not robust natural-language understanding or an injection
boundary; source complexity and hidden requirements can still defeat the rules.
Premium is not guaranteed correct. Semantic matches are not guarantees of equivalence.
The Qdrant collection still assumes 384-dimensional embeddings; changing embedding
models requires checking dimensions and migration.

See [Phase 3 walkthrough](docs/phase3_routing.md) for file-by-file explanations.
The original implementation blueprint is preserved unchanged as historical planning.
