from __future__ import annotations
from typing import Any, ClassVar, Iterator

from ai_navigator.infra.types import ContentPart, Message, NavigatorResult
from ai_navigator.infra.status_codes import SC, describe as status_describe
from ai_navigator.server.base_server import BaseServer, server_method


class AnthropicServer(BaseServer):
    """Server for Anthropic Claude models: claude-opus-4, claude-sonnet-4, …

    Credentials dict keys
    ---------------------
    - ``api_key``       (required) — Anthropic API key.
    - ``max_tokens``    (optional) — Default max tokens per response (default 4096).

    Status codes returned
    ----------------------
    200  Ok
    401  Unauthorized  (invalid or missing API key)
    429  Too Many Requests  (provider rate-limit; retried automatically)
    500  Internal Server Error  (unexpected SDK / network error)
    """

    provider: ClassVar[str] = "anthropic"

    def _setup(self, **kwargs: Any) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install ai-navigator[anthropic]"
            ) from exc
        self._client = Anthropic(api_key=self.credentials.get("api_key"))
        self._default_max_tokens: int = int(self.credentials.get("max_tokens", 4096))

    # ── Server methods ────────────────────────────────────────────────────────

    @server_method
    def chat(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> NavigatorResult:
        """Standard chat completion."""
        system_text, user_msgs = _split_system(self._normalise(messages, system))
        return self._call_api(system_text, user_msgs, **kwargs)

    @server_method
    def response(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> NavigatorResult:
        """Structured-output completion (JSON via prompt instruction)."""
        instruction = kwargs.pop(
            "json_instruction",
            "Respond ONLY with a valid JSON object. No prose, no markdown fences.",
        )
        system_text, user_msgs = _split_system(self._normalise(messages, system))
        combined_system = "\n\n".join(filter(None, [system_text, instruction]))
        return self._call_api(combined_system or None, user_msgs, **kwargs)

    # ── Streaming (not a server_method — different return type) ──────────────

    def stream(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Token-by-token streaming."""
        system_text, user_msgs = _split_system(self._normalise(messages, system))
        yield from self._stream(system_text, user_msgs, **kwargs)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_api(
        self,
        system_text: str | None,
        user_msgs: list[Message],
        **kwargs: Any,
    ) -> NavigatorResult:
        """Raw Anthropic API call — shared by chat() and response()."""
        kwargs.setdefault("max_tokens", self._default_max_tokens)
        create_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[_to_anthropic_message(m) for m in user_msgs],
            **kwargs,
        )
        if system_text:
            create_kwargs["system"] = system_text
        try:
            resp = self._client.messages.create(**create_kwargs)
        except Exception as exc:
            code = _classify(exc)
            self.logger.warning("Anthropic API error [%d %s]: %s", code, status_describe(code), exc)
            return NavigatorResult(
                result="",
                status={"status_code": code, "status_desc": status_describe(code), "status_detail": str(exc)},
                usage={},
                reference={},
            )
        content = "".join(
            block.text for block in resp.content if hasattr(block, "text")
        )
        return NavigatorResult(
            result=content,
            status={"status_code": SC.OK, "status_desc": status_describe(SC.OK), "status_detail": ""},
            usage={
                "prompt_tokens":      resp.usage.input_tokens,
                "completion_tokens":  resp.usage.output_tokens,
                "total_tokens":       resp.usage.input_tokens + resp.usage.output_tokens,
            },
            reference={"model": resp.model, "finish_reason": resp.stop_reason},
        )

    def _stream(
        self,
        system_text: str | None,
        user_msgs: list[Message],
        **kwargs: Any,
    ) -> Iterator[str]:
        kwargs.setdefault("max_tokens", self._default_max_tokens)
        stream_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[_to_anthropic_message(m) for m in user_msgs],
            **kwargs,
        )
        if system_text:
            stream_kwargs["system"] = system_text
        with self._client.messages.stream(**stream_kwargs) as stream:
            for text in stream.text_stream:
                yield text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify(exc: Exception) -> int:
    try:
        from anthropic import AuthenticationError, RateLimitError
        if isinstance(exc, AuthenticationError):
            return SC.UNAUTHORIZED
        if isinstance(exc, RateLimitError):
            return SC.TOO_MANY_REQUESTS
    except ImportError:
        pass
    return SC.INTERNAL_ERROR


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system: str | None = None
    rest: list[Message] = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"] if isinstance(msg["content"], str) else ""
        else:
            rest.append(msg)
    return system, rest


def _to_anthropic_message(msg: Message) -> dict[str, Any]:
    content = msg["content"]
    if isinstance(content, str):
        return {"role": msg["role"], "content": content}
    return {"role": msg["role"], "content": [_part_to_anthropic(p) for p in content]}


def _part_to_anthropic(part: ContentPart) -> dict[str, Any]:
    if part["type"] == "text":
        return {"type": "text", "text": part.get("text", "")}
    if part["type"] == "image_url":
        return {"type": "image", "source": {"type": "url", "url": part.get("image_url", "")}}
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": part.get("media_type", "image/jpeg"),
            "data": part.get("image_data", ""),
        },
    }
