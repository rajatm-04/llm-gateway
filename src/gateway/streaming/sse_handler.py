"""Server-Sent Events formatting helpers."""

import json


def format_sse(
    content: str = "",
    finish_reason: str | None = None,
) -> str:
    """Format one chat chunk as an SSE event."""
    data = {
        "content": content,
        "finish_reason": finish_reason,
    }
    return f"data: {json.dumps(data)}\n\n"


def format_done() -> str:
    """Format the final SSE marker."""
    return "data: [DONE]\n\n"
