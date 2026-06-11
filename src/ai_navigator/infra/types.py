"""Core TypedDict definitions — zero internal dependencies.

All field access is dict-style: ``msg["role"]``, ``part["type"]``, etc.
"""
from __future__ import annotations
from typing import Any, Union, TypedDict


class ContentPart(TypedDict, total=False):
    type: str        # "text" | "image_url" | "image_base64"
    text: str
    image_url: str
    image_data: str  # base64-encoded
    media_type: str


class Message(TypedDict):
    role: str        # "system" | "user" | "assistant"
    content: Union[str, list]


class TokenUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int
    total_tokens: int


class Response(TypedDict, total=False):
    content: str
    model: str
    usage: TokenUsage
    finish_reason: str
    raw: Any


class CallStatus(TypedDict):
    ok: bool
    error: str | None
    error_type: str | None


class NavigatorResult(TypedDict):
    result: Any        # server Response TypedDict, or None on error
    usage: TokenUsage  # token usage, or {} on error
    status: CallStatus


def make_content_part(type_: str, **kwargs: Any) -> ContentPart:
    return {"type": type_, **kwargs}  # type: ignore[return-value]


def make_message(role: str, content: Union[str, list]) -> Message:
    return {"role": role, "content": content}  # type: ignore[return-value]
