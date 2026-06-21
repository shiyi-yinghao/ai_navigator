import logging

from ai_navigator.state.status import StatusDetail, StatusCode, describe as status_describe
from ai_navigator.state.data_class import (
    ContentPart,
    Message,
    Response,
    TokenUsage,
    make_content_part,
    make_message,
)
from ai_navigator.infra.state import RequestState

# Backward-compat aliases
CallStatus = StatusDetail
STATUS_DESCRIPTIONS = StatusCode._registry

logging.getLogger("ai_navigator").addHandler(logging.NullHandler())

__all__ = [
    # Types
    "StatusDetail",
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
]
