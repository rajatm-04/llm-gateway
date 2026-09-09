# Practical grading policy: substantive-v1

User-approved evaluation direction: assess material factual discrepancies and task
completion, not personal stylistic preferences. Premium is the comparison benchmark,
not an infallible ground truth. Apply the same checks to both providers.

## Grades

- `yes`: accomplishes the task without a clear material factual/meaning error.
- `no`: identifiable factual error, changed important meaning, consequential invented
  commitment/obligation, missing required task component, or materially incomplete logic.
- `review`: genuine ambiguity about materiality or completeness; neither pass nor fail.
- Blank: not yet assessed. This is different from review.

Allow natural paraphrases, greetings, placeholders, extra length and different valid
explanations unless they violate an explicit requirement. No byte-for-byte matching
unless the task demands it. Do not fail an answer simply because premium sounds nicer.
Preserve material amounts, deadlines, negation, conditions and authorization status.
A rewrite must actually perform its requested transformation; copying the unchanged
input is not enough when a change was requested and no reason for retaining it is given.

Use per-case rubrics as factual/task checklists, interpreted under this practical
standard. Do not add new hidden requirements after reading answers. If a rubric is
ambiguous, record review with the disputed point. A length finish reason triggers
review of the actual content, not automatic failure: distinguish missing core reasoning
from a cut-off optional elaboration. Stop does not prove correctness or completeness.

Examples agreed during development:
- rw01: both tentative Friday rewrites acceptable; nuance is not a material failure.
- rw04: local changes denial of past approval to present inability: clear failure.
- rw10: local invents legal commitments; premium loses next-month deadline: both fail.
- rw05: whether added nonacceptance of unsigned forms creates a new policy is review,
  not an automatic failure. Required date/time and signed-form condition are preserved.

## Reporting

Only pairs with yes/no for BOTH models enter comparative quality rates. Report review,
ungraded and error counts alongside those rates; exclusions can bias the estimate.
Show both-acceptable, premium-only acceptable, local-only acceptable and neither counts.
For local-selected cases, show local=no/premium=yes count with the resolved-local-pair
denominator. This measures clear losses against the premium benchmark, not stylistic
preference or a calibrated production error probability. Both failing is not a success.
Latency/token summaries may cover more cases than the quality denominator; disclose it.
Never merge 256-token and 1024-token runs as one uniform benchmark.

## Retrospective application / audit

This standard was agreed AFTER inspecting development outputs, not preregistered.
The assistant recorded grades with substantive-v1 notes in the full development and
rewrite runs. Their grade files were blank before this review; raw results, original
rubrics and generation manifests are unchanged. Earlier conversational assessments
that treated nuance as failure are superseded by these documented grades. No held-out
model outcomes were used. Prior d01 smoke and d13/d14 rerun grades remain consistent
with this policy and are retained without changing their notes.

These are assistant judgments, open to human review; they do not approve the local
model profile. Keep rewriting enabled experimentally; stop expanding tone tests now.
Next effort: fact-checkable extraction (missing/conflicting values), summaries (facts,
exceptions and near-limit inputs), reasoning (required steps), and live gateway/cache
integration. Write expected facts before calls. Freeze policy/settings before held-out
assessment; do not tune based on held-out results and still call them held-out.
