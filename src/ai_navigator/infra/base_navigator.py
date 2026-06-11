# Backward-compatibility shim.
# Canonical locations:
#   TypedDicts    → ai_navigator.infra.types
#   BaseNavigator → ai_navigator.service.base_navigator
from ai_navigator.infra.types import (  # noqa: F401
    ContentPart, Message, TokenUsage, Response,
    make_content_part, make_message,
)
from ai_navigator.service.base_navigator import BaseNavigator  # noqa: F401
