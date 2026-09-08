
import time
import uuid
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from gateway.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
    )