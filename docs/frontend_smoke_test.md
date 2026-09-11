# Browser playground: manual verification

The interface is implemented, but has not been executed or visually verified by
the filesystem-only assistant. Offline route tests do not execute JavaScript.
Run the Python checks and this browser walkthrough locally before marking the
live gateway checkpoint complete.

## Start

Keep Ollama and Qdrant running. From PowerShell in the project root:

```powershell
uv run pytest tests/ -q
uv run ruff check src/gateway tests experiments
uv run uvicorn gateway.main:app --reload
```

Open http://localhost:8000. No Node.js, build step, external font, CDN, or frontend
dependency is required. If a previous page/script is cached, use Ctrl+F5.
The root URL now serves HTML instead of the former welcome JSON; /health and
/v1/chat/completions are unchanged. /docs remains the API reference.

Loading the page fetches only assets and /ui/config, an allowlist of model names
and output limits. It does not check provider health or generate.
Qdrant is still required during application startup. First semantic use can
load/download the embedding model.

## Safety and behavior

- Local is the UI default; the backend's omitted-tier behavior is still Auto.
- Auto/Premium or a premium explicit model requires a per-send acknowledgement.
  This is a browser convenience, NOT authentication or a backend spending limit.
- Only Send request submits a prompt. Example buttons never submit. There are
  no automatic retries, model calls on page load, or persistent browser history.
- Each send contains one user message, not an accumulated conversation.
- Qdrant may store prompts and answers. Use public/non-sensitive examples.
- Stop waiting aborts the browser fetch. Upstream work, billing, or a cache write
  may already have happened; do not interpret stopping as a refund or deletion.
- Output is rendered as plain text, including Markdown, HTML and code. No model
  output is inserted using innerHTML. Streamed text is not announced token by
  token; status updates are announced, and the answer region is keyboard-focusable.
- Keep the application bound to localhost. No authentication or tenant isolation
  has been added. Do not expose it publicly.

## Bounded live smoke test

Use a new reference such as `browser-check-2026-09-09-A` in the SOURCE of each
prompt to avoid old cache entries. Keep that reference and all settings unchanged
when repeating. Check each result before proceeding; stop on any unexpected tier,
BYPASS, HTTP error, or failed stream. No cache deletion is necessary.

This sequence expects two premium generations. If you perform all three premium
submissions despite a cache failure, there can be three premium attempts. Each
uses a 128-output-token ceiling, not a dollar cap. Input billing also applies.

1. **Automatic local:** set Routing to Auto, Explicit model to Unspecified,
   max output tokens to 128, temperature to 0, streaming OFF. Enter:
   `Make this email more polite: Send the report for reference browser-check-2026-09-09-A today.`
   Acknowledge possible premium use (Auto cannot promise local), then send once.
   Expect HTTP 200, local, MISS, rewrite_profile_match. Repeat unchanged,
   acknowledging again: expect HIT and zero new provider usage.
2. **Tier isolation (paid):** keep the same prompt/settings and select Premium.
   Acknowledge and send once. Expect premium, MISS, explicit_tier. It must not
   reuse the local entry. Do not compare exact answer wording.
3. **Automatic premium (paid):** select Auto with no explicit model and enter:
   `What is 2+2? Reply briefly. Request reference: browser-check-2026-09-09-A.`
   Acknowledge and send: expect premium, MISS, no_supported_instruction_pattern.
   Repeat unchanged: expect HIT, with no new generation. This simple question
   is outside the classifier's supported instruction grammar.
4. **Fresh stream:** select Local, keep model Unspecified, enable streaming,
   and use a new source:
   `Summarize this text in two bullets: Reference browser-check-2026-09-09-A. The library opens at nine. It closes at six.`
   Send: expect local, MISS, visible text chunks and Complete status. Model
   metadata must say selected; token usage is unknown. Repeat unchanged: HIT,
   usually a single content event. A length finish means review completeness,
   not a transport failure. A stream error must NOT show Complete even if HTTP
   status is 200.
5. **Conflict:** choose Premium and select the local model under Advanced
   settings. Acknowledge (the UI is conservative) and send. Expect HTTP 422
   with a model/tier conflict message, and no provider call.

MISS reports lookup outcome, not successful storage. A subsequent HIT confirms
reuse. Browser elapsed time includes transport and response reading; cache lookup
latency is separate. Neither is a controlled benchmark.

## Browser-only UI checks

These are manual checks; the assistant has not run them. Some require additional
LOCAL generations. Keep Routing Local and Explicit model Unspecified unless a
specific test above says otherwise.

- Tab through labels, controls, advanced settings, Send, and the answer region.
  Verify visible focus, readable labels, and no keyboard trap. Check a narrow
  mobile viewport, 200% zoom, and long answer/model text for horizontal overflow.
- Click an example: it should fill the prompt and leave results untouched.
- Empty/blank prompts and invalid numeric values should not send. A browser
  network inspector can confirm no POST was made.
- While a request is active, Send and request inputs should be disabled. Stop
  waiting should be available. Stopping should show a clear status, preserve any
  partial text with a warning, and re-enable controls without retrying.
- If the gateway is stopped AFTER the page has loaded, a Local send should show
  a connection error, not stale success metadata. Restart it before proceeding.
- Check with a screen reader that status changes are announced without reading
  every streamed token. Check text contrast and keyboard focus in the browser.

Malformed/split SSE, UTF-8 boundary handling, rendering safety and abort handling
are implemented in JavaScript but have not been browser-automated here. Python
route tests cover delivery/config isolation, not these client behaviors. Existing
provider/stream tests cover server-side incomplete and error paths.
