from __future__ import annotations
from typing import Any, ClassVar, Iterator

from ai_navigator.state.data_class import ContentPart, Message, NavigatorResult
from ai_navigator.state.status import StatusCode
from ai_navigator.server.base_server import BaseServer, server_method, ok_result, err_result, normalise_messages


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

    def _setup(self) -> None:
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
        messages: list[Message],
        model: str,
        param: dict,
    ) -> NavigatorResult:
        """Standard chat completion."""
        system_text, user_msgs = _split_system(messages)
        effective = dict(param)
        effective.setdefault("max_tokens", self._default_max_tokens)
        create_kwargs: dict[str, Any] = dict(
            model=model,
            messages=[_to_anthropic_message(m) for m in user_msgs],
            **effective,
        )
        if system_text:
            create_kwargs["system"] = system_text
        try:
            resp = self._client.messages.create(**create_kwargs)
        except Exception as exc:
            code = _classify(exc)
            self.logger.warning("Anthropic [%d]: %s", code, exc)
            return err_result(code, str(exc))
        content = "".join(
            block.text for block in resp.content if hasattr(block, "text")
        )
        return ok_result(
            content,
            {
                "prompt_tokens":     resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens":      resp.usage.input_tokens + resp.usage.output_tokens,
            },
            {"model": resp.model, "finish_reason": resp.stop_reason},
        )

    @server_method
    def response(
        self,
        messages: list[Message],
        model: str,
        param: dict,
    ) -> NavigatorResult:
        """Structured-output completion (JSON via prompt instruction)."""
        effective = dict(param)
        instruction = effective.pop(
            "json_instruction",
            "Respond ONLY with a valid JSON object. No prose, no markdown fences.",
        )
        system_text, user_msgs = _split_system(messages)
        combined_system = "\n\n".join(filter(None, [system_text, instruction]))
        effective.setdefault("max_tokens", self._default_max_tokens)
        create_kwargs: dict[str, Any] = dict(
            model=model,
            messages=[_to_anthropic_message(m) for m in user_msgs],
            **effective,
        )
        if combined_system:
            create_kwargs["system"] = combined_system
        try:
            resp = self._client.messages.create(**create_kwargs)
        except Exception as exc:
            code = _classify(exc)
            self.logger.warning("Anthropic [%d]: %s", code, exc)
            return err_result(code, str(exc))
        content = "".join(
            block.text for block in resp.content if hasattr(block, "text")
        )
        return ok_result(
            content,
            {
                "prompt_tokens":     resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens":      resp.usage.input_tokens + resp.usage.output_tokens,
            },
            {"model": resp.model, "finish_reason": resp.stop_reason},
        )

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[Message] | str,
        model: str,
        param: dict,
    ) -> Iterator[str]:
        """Token-by-token streaming."""
        system_text, user_msgs = _split_system(normalise_messages(messages))
        effective = dict(param)
        effective.setdefault("max_tokens", self._default_max_tokens)
        stream_kwargs: dict[str, Any] = dict(
            model=model,
            messages=[_to_anthropic_message(m) for m in user_msgs],
            **effective,
        )
        if system_text:
            stream_kwargs["system"] = system_text
        with self._client.messages.stream(**stream_kwargs) as s:
            for text in s.text_stream:
                yield text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify(exc: Exception) -> StatusCode:
    try:
        from anthropic import AuthenticationError, RateLimitError
        if isinstance(exc, AuthenticationError):
            return StatusCode.UNAUTHORIZED
        if isinstance(exc, RateLimitError):
            return StatusCode.TOO_MANY_REQUESTS
    except ImportError:
        pass
    return StatusCode.INTERNAL_ERROR


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
