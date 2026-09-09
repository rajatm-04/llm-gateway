"""Server-Sent Events formatting helpers."""

import json


def _format_event(data: dict) -> str:
    """Format one SSE data event."""
    return f"data: {json.dumps(data)}\n\n"


def format_sse(
    content: str = "",
    finish_reason: str | None = None,
) -> str:
    """Format one chat chunk as an SSE event."""
    return _format_event(
        {
            "content": content,
            "finish_reason": finish_reason,
        }
    )


def format_error(
    code: str = "upstream_stream_error",
    message: str = "Provider stream failed",
) -> str:
    """Format a sanitized stream error event."""
    return _format_event({"error": {"code": code, "message": message}})


def format_done() -> str:
    """Format the final SSE marker."""
    return "data: [DONE]\n\n"
