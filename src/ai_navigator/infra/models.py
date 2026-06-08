from __future__ import annotations
from typing import Any, Literal

from pydantic import BaseModel, Field


class ContentPart(BaseModel):
    """A single part of a multimodal message."""

    type: Literal["text", "image_url", "image_base64"]
    text: str | None = None
    image_url: str | None = None
    image_data: str | None = None  # base64-encoded bytes
    media_type: str | None = None  # e.g. "image/jpeg"


class Message(BaseModel):
    """A single turn in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role="system", content=text)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls(role="assistant", content=text)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0


class Response(BaseModel):
    """Normalised LLM response, provider-agnostic."""

    content: str
    model: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    raw: Any = Field(default=None, exclude=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude={"raw"})
