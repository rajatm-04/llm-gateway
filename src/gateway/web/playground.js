"use strict";

const $ = (id) => document.getElementById(id);
let config = null;
let activeController = null;
const setText = (id, value) => { $(id).textContent = value; };
const resultIds = ["tier", "cache", "http", "elapsed", "model", "source", "reason", "cache-time", "finish", "usage", "profile", "policy"];

function couldUsePremium() {
  // Conservative UI acknowledgement, not a duplicate of backend routing logic.
  return $("tier").value !== "local" || (config && $("model").value === config.premium_model);
}

function updateControls() {
  $("paid-consent").checked = false;
  const paid = couldUsePremium();
  $("paid-notice").hidden = !paid;
  $("paid-consent").required = paid;
  const descriptions = {
    local: "Use the local model. An explicit premium model conflicts with this tier.",
    auto: "Let the gateway choose. Unrecognized tasks can go to premium.",
    premium: "Use the premium tier. Cache misses can incur charges."
  };
  setText("tier-help", descriptions[$("tier").value]);
}

function showError(message) {
  setText("error", message);
  $("error").hidden = false;
}

function appendAnswer(text) {
  $("answer-empty").hidden = true;
  $("answer").hidden = false;
  // Model output is untrusted text, never HTML or executable Markdown.
  $("answer").append(document.createTextNode(text));
}

function displayHeaders(response) {
  const header = (name) => response.headers.get(name) || "Not reported";
  setText("result-http", String(response.status));
  for (const [id, name] of Object.entries({
    tier: "X-Model-Tier", cache: "X-Cache", model: "X-Model-Used",
    reason: "X-Routing-Reason", profile: "X-Routing-Profile", policy: "X-Routing-Policy"
  })) {
    setText(`result-${id}`, header(name));
  }
  const cache = response.headers.get("X-Cache");
  const explanations = {
    HIT: "Saved answer reused. No new provider generation for this request.",
    MISS: "No eligible saved answer found. This does not confirm that the subsequent cache write succeeded.",
    BYPASS: "Cache lookup failed. The gateway attempted generation instead; repeating may call the provider again."
  };
  setText("cache-explanation", explanations[cache] || "Cache information was not returned.");
  const lookup = response.headers.get("X-Cache-Latency-Ms");
  setText("result-cache-time", lookup === null ? "Not reported" : `${lookup} ms (lookup only)`);
  if (response.headers.get("X-Model-Used")) {
    setText("result-source", cache === "HIT" ? "Saved response metadata" :
      response.headers.get("X-Model-Used-Source") === "selected" ?
        "Selected model; actual model metadata is not included in these SSE events" : "Provider response metadata");
  }
}

async function httpError(response) {
  let message = `HTTP ${response.status}: request failed.`;
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") message += ` ${payload.detail}`;
    else if (Array.isArray(payload.detail)) {
      message += " " + payload.detail.map((item) => `${(item.loc || []).join(".")}: ${item.msg}`).join("; ");
    }
  } catch {
    // Do not display a proxy's HTML error page or an arbitrary raw response body.
  }
  return new Error(message);
}

async function readStream(response) {
  if (!response.body) throw new Error("This browser did not provide a readable response stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let doneMarker = false;
  let finish = null;

  function consumeEvent(event) {
    const data = event.split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart()).join("\n");
    if (!data.trim()) return;
    if (doneMarker) throw new Error("Received data after the stream completion marker.");
    if (data.trim() === "[DONE]") {
      doneMarker = true;
      return;
    }
    let chunk;
    try { chunk = JSON.parse(data); }
    catch { throw new Error("The gateway returned an invalid streaming event."); }
    if (!chunk || typeof chunk !== "object") throw new Error("Invalid streaming payload.");
    if (chunk.error) throw new Error("Provider stream failed. Any text shown is partial; the stream did not complete successfully.");
    if (typeof chunk.content === "string" && chunk.content) appendAnswer(chunk.content);
    if (typeof chunk.finish_reason === "string") {
      finish = chunk.finish_reason;
      setText("result-finish", finish);
    }
  }

  function consumeBuffer() {
    // Network chunks can split JSON, UTF-8 characters, or SSE event delimiters.
    let boundary;
    while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
      consumeEvent(buffer.slice(0, boundary.index));
      buffer = buffer.slice(boundary.index + boundary[0].length);
    }
  }

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      consumeBuffer();
    }
    buffer += decoder.decode();
    consumeBuffer();
    if (buffer.trim() || !doneMarker || !finish) {
      throw new Error("The stream ended without a complete terminal event. Any displayed answer may be partial.");
    }
    return finish;
  } finally {
    try { await reader.cancel(); } catch { /* Already closed or aborted. */ }
    reader.releaseLock();
  }
}

$("request-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!config || activeController) return;
  $("prompt").setCustomValidity($("prompt").value.trim() ? "" : "Enter a nonblank prompt.");
  if (!$("request-form").reportValidity()) return;
  if (couldUsePremium() && !$("paid-consent").checked) return;

  const body = {
    messages: [{ role: "user", content: $("prompt").value }],
    temperature: Number($("temperature").value),
    max_tokens: Number($("max-tokens").value),
    stream: $("stream").checked
  };
  const model = $("model").value;
  const tier = $("tier").value;
  if (model) body.model = model;
  const headers = { "Content-Type": "application/json" };
  if (tier !== "auto") headers["X-Model-Tier"] = tier;

  $("error").hidden = true;
  $("answer-note").hidden = true;
  $("answer").hidden = true;
  setText("answer", "");
  $("answer-empty").hidden = false;
  setText("answer-empty", "Waiting for the gateway. First use may load the embedding or local model.");
  for (const id of resultIds) setText(`result-${id}`, "Not reported");
  setText("result-elapsed", "Waiting");
  setText("cache-explanation", "");
  $("details-empty").hidden = true;
  $("request-details").hidden = false;
  setText("request-summary", `Sent: ${tier} routing · ${model || "no explicit model"} · ${body.max_tokens} max output tokens · temperature ${body.temperature} · ${body.stream ? "streaming" : "normal response"}`);
  setText("request-status", "Sending request…");
  $("request-controls").disabled = true;
  $("send").disabled = true;
  $("stop").hidden = false;
  $("stop").disabled = false;
  $("answer").setAttribute("aria-busy", "true");
  activeController = new AbortController();
  const started = performance.now();

  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST", headers, body: JSON.stringify(body), signal: activeController.signal
    });
    displayHeaders(response);
    if (!response.ok) throw await httpError(response);
    let finish;
    if (body.stream) {
      if (!response.headers.get("Content-Type")?.includes("text/event-stream")) {
        throw new Error("Expected an SSE response but received a different content type.");
      }
      setText("request-status", "Receiving stream…");
      setText("result-usage", response.headers.get("X-Cache") === "HIT" ? "0 new provider tokens (cache hit)" : "Unknown for streamed generations");
      finish = await readStream(response);
    } else {
      const payload = await response.json();
      const choice = payload.choices?.[0];
      if (typeof choice?.message?.content !== "string") throw new Error("The gateway returned no readable answer.");
      appendAnswer(choice.message.content);
      finish = choice.finish_reason;
      setText("result-finish", finish || "Not reported");
      if (payload.usage) {
        const usage = payload.usage;
        setText("result-usage", `${usage.prompt_tokens} input + ${usage.completion_tokens} output = ${usage.total_tokens} total${response.headers.get("X-Cache") === "HIT" ? " (new usage; cache hit)" : " (reported)"}`);
      }
    }
    if (!$("answer").textContent.trim()) throw new Error("The response completed without any answer text.");
    setText("request-status", finish === "length" ? "Complete · output limit reached" : "Complete");
    if (finish === "length") {
      setText("answer-note", "The output limit was reached. Check whether the answer is complete. Increasing the limit changes the cache scope and may trigger another generation.");
      $("answer-note").hidden = false;
    }
  } catch (error) {
    const aborted = activeController.signal.aborted;
    setText("request-status", aborted ? "Stopped waiting" : "Request failed");
    showError(aborted ? "Stopped waiting for this request. Provider work or billing may continue; cache storage may already have completed." :
      error instanceof TypeError ? "Could not read the gateway response. Check your connection and the server terminal. No automatic retry was made." : error.message);
    if ($("answer").textContent) {
      setText("answer-note", "Text received before completion could be confirmed. Treat it as potentially partial.");
      $("answer-note").hidden = false;
    } else {
      setText("answer-empty", "No answer received.");
    }
  } finally {
    setText("result-elapsed", `${((performance.now() - started) / 1000).toFixed(2)} s`);
    activeController = null;
    $("request-controls").disabled = false;
    $("send").disabled = false;
    $("stop").hidden = true;
    $("answer").setAttribute("aria-busy", "false");
    $("paid-consent").checked = false;
    $("send").focus();
  }
});

$("stop").addEventListener("click", () => {
  activeController?.abort();
  $("stop").disabled = true;
  setText("request-status", "Stopping…");
});
$("prompt").addEventListener("input", () => $("prompt").setCustomValidity(""));
$("request-controls").addEventListener("input", (event) => {
  if (event.target.id !== "paid-consent") $("paid-consent").checked = false;
});
$("tier").addEventListener("change", updateControls);
$("model").addEventListener("change", updateControls);

const examples = {
  rewrite: "Make this email more polite: Send the report today.",
  summary: "Summarize this text in two bullets: The library opens at nine. It closes at six. It is closed on Sunday.",
  extraction: "Extract the invoice number and total: Invoice number: INV-204. Subtotal: $70. Tax: $3.50. Total: $73.50.",
  question: "What is 2+2? Reply briefly."
};
document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    $("prompt").value = examples[button.dataset.example];
    $("prompt").setCustomValidity("");
    $("paid-consent").checked = false;
    $("prompt").focus();
  });
});

async function initialize() {
  try {
    // Only public UI settings are fetched on load. No prompt or provider call.
    const response = await fetch("/ui/config", { cache: "no-store" });
    if (!response.ok) throw new Error("Settings unavailable");
    config = await response.json();
    for (const [label, name] of [["Local", config.local_model], ["Premium", config.premium_model]]) {
      $("model").add(new Option(`${label} — ${name}`, name));
    }
    $("max-tokens").value = String(Math.min(128, config.local_max_output_tokens, config.premium_max_output_tokens));
    setText("connection", `Gateway settings loaded · Routing mode: ${config.routing_mode}. Provider availability is checked only when you send.`);
    setText("limits", `Configured output limits: local ${config.local_max_output_tokens}, premium ${config.premium_max_output_tokens} tokens. The server validates the selected route; these are not native context-window sizes.`);
    updateControls();
    $("request-controls").disabled = false;
    $("send").disabled = false;
  } catch {
    setText("connection", "Gateway settings could not be loaded.");
    showError("Start the gateway, then reload this page at http://localhost:8000. Do not open the HTML file directly.");
  }
}
initialize();
