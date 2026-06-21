import logging

from ai_navigator.infra.exceptions import (
    AINavigatorError,
    ParseError,
    PreProcessorError,
    SchemaError,
    StorageError,
)
from ai_navigator.infra.types import (
    CallStatus,
    ContentPart,
    Message,
    Response,
    TokenUsage,
    make_content_part,
    make_message,
)
from ai_navigator.infra.state import RequestState
from ai_navigator.monitor.status_codes import StatusCode, describe as status_describe

# Backward-compat alias: STATUS_DESCRIPTIONS is the live registry dict
STATUS_DESCRIPTIONS = StatusCode._registry

logging.getLogger("ai_navigator").addHandler(logging.NullHandler())

__all__ = [
    # Types
    "CallStatus",
    "ContentPart",
    "Message",
    "Response",
    "TokenUsage",
    "make_content_part",
    "make_message",
    # State
    "RequestState",
    # Status codes
    "StatusCode",
    "STATUS_DESCRIPTIONS",
    "status_describe",
    # Exceptions
    "AINavigatorError",
    "ParseError",
    "PreProcessorError",
    "SchemaError",
    "StorageError",
]
