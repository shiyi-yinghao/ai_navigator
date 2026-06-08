from __future__ import annotations
from typing import Any, ClassVar, Iterator

from ai_navigator.infra.exceptions import AuthenticationError, ProviderError, RateLimitError
from ai_navigator.infra.models import ContentPart, Message, Response, TokenUsage
from ai_navigator.server.base_server import BaseServer


class AnthropicServer(BaseServer):
    """Server for Anthropic Claude models: claude-opus-4, claude-sonnet-4, …

    Credentials dict keys
    ---------------------
    - ``api_key``       (required) — Anthropic API key.
    - ``max_tokens``    (optional) — Default max tokens per response (default 4096).
    """

    provider: ClassVar[str] = "anthropic"
    _supported_methods: ClassVar[list[str]] = ["chat", "response"]

    def _setup(self, **kwargs: Any) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install ai-navigator[anthropic]"
            ) from exc
        self._client = Anthropic(api_key=self.credentials.get("api_key"))
        self._default_max_tokens: int = int(self.credentials.get("max_tokens", 4096))

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """Standard chat completion."""
        return self._invoke("chat", self._normalise(messages, system), **kwargs)

    def response(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """Structured-output completion.

        Appends a JSON instruction to the system prompt so the model replies
        with a JSON object.  Pass ``json_instruction`` in kwargs to override
        the default instruction text.
        """
        return self._invoke("response", self._normalise(messages, system), **kwargs)

    def stream(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Token-by-token streaming."""
        yield from self._stream(self._normalise(messages, system), **kwargs)

    # ── Private implementations ───────────────────────────────────────────────

    def _chat(self, messages: list[Message], **kwargs: Any) -> Response:
        system_text, user_msgs = _split_system(messages)
        kwargs.setdefault("max_tokens", self._default_max_tokens)
        create_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[_to_anthropic_message(m) for m in user_msgs],
            **kwargs,
        )
        if system_text is not None:
            create_kwargs["system"] = system_text
        try:
            resp = self._client.messages.create(**create_kwargs)
        except Exception as exc:
            _raise_anthropic_error(exc)
        content = "".join(
            block.text for block in resp.content if hasattr(block, "text")
        )
        return Response(
            content=content,
            model=resp.model,
            usage=TokenUsage(
                prompt_tokens=resp.usage.input_tokens,
                completion_tokens=resp.usage.output_tokens,
                total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
            ),
            finish_reason=resp.stop_reason,
            raw=resp,
        )

    def _response(self, messages: list[Message], **kwargs: Any) -> Response:
        """Structured output via prompt-level JSON instruction."""
        instruction = kwargs.pop(
            "json_instruction",
            "Respond ONLY with a valid JSON object. No prose, no markdown fences.",
        )
        # Inject the JSON instruction into / alongside the system message
        system_text, user_msgs = _split_system(messages)
        combined_system = "\n\n".join(filter(None, [system_text, instruction]))
        patched = [Message.system(combined_system), *user_msgs]
        return self._chat(patched, **kwargs)

    def _stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        system_text, user_msgs = _split_system(messages)
        kwargs.setdefault("max_tokens", self._default_max_tokens)
        stream_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[_to_anthropic_message(m) for m in user_msgs],
            **kwargs,
        )
        if system_text is not None:
            stream_kwargs["system"] = system_text
        with self._client.messages.stream(**stream_kwargs) as stream:
            for text in stream.text_stream:
                yield text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system: str | None = None
    rest: list[Message] = []
    for msg in messages:
        if msg.role == "system":
            system = msg.content if isinstance(msg.content, str) else ""
        else:
            rest.append(msg)
    return system, rest


def _to_anthropic_message(msg: Message) -> dict[str, Any]:
    if isinstance(msg.content, str):
        return {"role": msg.role, "content": msg.content}
    return {"role": msg.role, "content": [_part_to_anthropic(p) for p in msg.content]}


def _part_to_anthropic(part: ContentPart) -> dict[str, Any]:
    if part.type == "text":
        return {"type": "text", "text": part.text or ""}
    if part.type == "image_url":
        return {"type": "image", "source": {"type": "url", "url": part.image_url or ""}}
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": part.media_type or "image/jpeg",
            "data": part.image_data or "",
        },
    }


def _raise_anthropic_error(exc: Exception) -> None:
    try:
        from anthropic import AuthenticationError as AnthAuth
        from anthropic import RateLimitError as AnthRate
    except ImportError:
        raise ProviderError(str(exc), "anthropic") from exc
    if isinstance(exc, AnthAuth):
        raise AuthenticationError(str(exc), "anthropic") from exc
    if isinstance(exc, AnthRate):
        raise RateLimitError(str(exc), "anthropic") from exc
    raise ProviderError(str(exc), "anthropic") from exc
