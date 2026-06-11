from __future__ import annotations
from typing import Any, ClassVar, Iterator

from ai_navigator.infra.exceptions import AuthenticationError, ProviderError, RateLimitError
from ai_navigator.infra.types import ContentPart, Message, Response, TokenUsage
from ai_navigator.server.base_server import BaseServer


class OpenAIServer(BaseServer):
    """Server for OpenAI models: gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, …

    Credentials dict keys
    ---------------------
    - ``api_key``   (required) — OpenAI API key.
    - ``base_url``  (optional) — Override for Azure OpenAI or compatible proxies.
    """

    provider: ClassVar[str] = "openai"
    _supported_methods: ClassVar[list[str]] = ["chat", "response"]

    def _setup(self, **kwargs: Any) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI SDK not installed. Run: pip install ai-navigator[openai]"
            ) from exc
        self._client = OpenAI(
            api_key=self.credentials.get("api_key"),
            base_url=self.credentials.get("base_url"),
        )

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

        Pass ``response_format`` in kwargs to control the output schema, e.g.::

            server.response(msgs, response_format={"type": "json_object"})
            server.response(msgs, response_format={"type": "json_schema",
                                                    "json_schema": schema_dict})
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
        oai_msgs = [_to_openai_message(m) for m in messages]
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=oai_msgs,  # type: ignore[arg-type]
                **kwargs,
            )
        except Exception as exc:
            _raise_openai_error(exc)
        choice = resp.choices[0]
        return {
            "content": choice.message.content or "",
            "model": resp.model,
            "usage": _parse_usage(resp.usage),
            "finish_reason": choice.finish_reason,
            "raw": resp,
        }

    def _response(self, messages: list[Message], **kwargs: Any) -> Response:
        """Structured output via ``response_format``.

        Defaults to ``json_object`` unless the caller overrides via kwargs.
        """
        kwargs.setdefault("response_format", {"type": "json_object"})
        return self._chat(messages, **kwargs)

    def _stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        oai_msgs = [_to_openai_message(m) for m in messages]
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=oai_msgs,  # type: ignore[arg-type]
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_openai_message(msg: Message) -> dict[str, Any]:
    content = msg["content"]
    if isinstance(content, str):
        return {"role": msg["role"], "content": content}
    return {"role": msg["role"], "content": [_part_to_openai(p) for p in content]}


def _part_to_openai(part: ContentPart) -> dict[str, Any]:
    if part["type"] == "text":
        return {"type": "text", "text": part.get("text", "")}
    if part["type"] == "image_url":
        return {"type": "image_url", "image_url": {"url": part.get("image_url", "")}}
    mt = part.get("media_type", "image/jpeg")
    return {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{part.get('image_data', '')}"}}


def _parse_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _raise_openai_error(exc: Exception) -> None:
    try:
        from openai import AuthenticationError as OAIAuth
        from openai import RateLimitError as OAIRate
    except ImportError:
        raise ProviderError(str(exc), "openai") from exc
    if isinstance(exc, OAIAuth):
        raise AuthenticationError(str(exc), "openai") from exc
    if isinstance(exc, OAIRate):
        raise RateLimitError(str(exc), "openai") from exc
    raise ProviderError(str(exc), "openai") from exc
