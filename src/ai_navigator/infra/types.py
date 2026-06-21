"""Backward-compatibility shim — canonical location is ai_navigator.state."""
from ai_navigator.state.status import StatusDetail
from ai_navigator.state.data_class import (
    ContentPart, Message, TokenUsage, Response,
    NavigatorResult, make_content_part, make_message,
)

# Legacy alias
CallStatus = StatusDetail

__all__ = [
    "ContentPart", "Message", "TokenUsage", "Response",
    "StatusDetail", "NavigatorResult", "make_content_part", "make_message",
    "CallStatus",
]
