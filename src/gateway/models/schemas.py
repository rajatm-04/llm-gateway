import uuid
import time
from enum import Enum
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str



class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False



class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "None"]



class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int



def generate_chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4()}"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=generate_chat_id)
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo



