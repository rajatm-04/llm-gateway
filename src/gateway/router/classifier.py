"""Conservative English task recognition, independent of model capability.

Only a deliberately small instruction grammar is recognized. The source after
an explicit delimiter is never searched for routing keywords. Unknown is a
normal outcome, not evidence that the task is objectively difficult.
"""

import re
from dataclasses import dataclass
from typing import Literal

from gateway.models.schemas import ChatCompletionRequest

TaskType = Literal["rewrite", "summary", "extraction", "unknown"]
CLASSIFIER_VERSION = "tasks-v1"

# Full matches prevent an easy verb from hiding extra instructions.
_PATTERNS: tuple[tuple[TaskType, re.Pattern], ...] = (
    ("rewrite", re.compile(
        r"(?:rewrite (?:this|the following) (?:text|email) "
        r"(?:to (?:be|sound)|in a) (?:more )?(?:polite|professional|friendly)(?: tone)?"
        r"|make this email (?:more )?(?:polite|professional|friendly)"
        r"|(?:fix|correct) the grammar (?:in|of) (?:this|the following) text)",
        re.IGNORECASE,
    )),
    ("summary", re.compile(
        r"summari[sz]e (?:this|the following) (?:text|paragraph|article)"
        r"(?: in (?:[1-5]|three|two|four|five) (?:bullet points|bullets|sentences))?",
        re.IGNORECASE,
    )),
    ("extraction", re.compile(
        r"extract (?:the )?(?:invoice number|order id|email address|date|total)"
        r"(?: and (?:the )?(?:invoice number|order id|email address|date|total))?"
        r"(?: from (?:this|the following) text)?",
        re.IGNORECASE,
    )),
)


@dataclass(frozen=True)
class TaskFeatures:
    task_type: TaskType
    reason: str
    input_words: int
    instruction: str = ""
    source: str = ""


def classify(request: ChatCompletionRequest) -> TaskFeatures:
    """Recognize only self-contained user-only requests with a clear boundary."""
    words = sum(len(message.content.split()) for message in request.messages)
    if len(request.messages) != 1 or request.messages[0].role != "user":
        return TaskFeatures("unknown", "conversation_or_system_context", words)

    text = request.messages[0].content.strip()
    # Supported forms: 'Instruction: ...\nText: ...', '...\nText: ...', or '...: ...'.
    text = re.sub(r"\Ainstruction:\s*", "", text, count=1, flags=re.IGNORECASE)
    parts = re.split(r"\n\s*text:\s*", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 1:
        parts = text.split(":", 1)
    if len(parts) != 2 or not parts[1].strip():
        return TaskFeatures("unknown", "missing_instruction_source_boundary", words)

    instruction = " ".join(parts[0].strip().rstrip(".:").split())
    instruction = re.sub(r"\Aplease\s+", "", instruction, flags=re.IGNORECASE)
    source = parts[1].strip()
    for task, pattern in _PATTERNS:
        if pattern.fullmatch(instruction):
            return TaskFeatures(task, "recognized_task_pattern", words, instruction, source)
    return TaskFeatures("unknown", "no_supported_instruction_pattern", words)
