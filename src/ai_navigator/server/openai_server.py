from __future__ import annotations
from typing import Any, ClassVar, Iterator

from ai_navigator.infra.types import ContentPart, Message, NavigatorResult, TokenUsage
from ai_navigator.monitor.status_codes import StatusCode, describe as status_describe
from ai_navigator.server.base_server import BaseServer, server_method


class OpenAIServer(BaseServer):
    """Server for OpenAI models: gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, …

    Credentials dict keys
    ---------------------
    - ``api_key``   (required) — OpenAI API key.
    - ``base_url``  (optional) — Override for Azure OpenAI or compatible proxies.

    Status codes returned
    ----------------------
    200  Ok
    401  Unauthorized  (invalid or missing API key)
    429  Too Many Requests  (provider rate-limit; retried automatically)
    500  Internal Server Error  (unexpected SDK / network error)
    """

    provider: ClassVar[str] = "openai"

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

    # ── Server methods ────────────────────────────────────────────────────────

    @server_method
    def chat(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> NavigatorResult:
        """Standard chat completion."""
        return self._call_api(self._normalise(messages, system), **kwargs)

    @server_method
    def response(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> NavigatorResult:
        """Structured-output completion.

        Pass ``response_format`` in kwargs to control the output schema, e.g.::

            server.response(msgs, response_format={"type": "json_object"})
            server.response(msgs, response_format={"type": "json_schema",
                                                    "json_schema": schema_dict})
        """
        kwargs.setdefault("response_format", {"type": "json_object"})
        return self._call_api(self._normalise(messages, system), **kwargs)

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Token-by-token streaming."""
        oai_msgs = [_to_openai_message(m) for m in self._normalise(messages, system)]
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_api(self, messages: list[Message], **kwargs: Any) -> NavigatorResult:
        """Raw OpenAI API call — shared by chat() and response()."""
        oai_msgs = [_to_openai_message(m) for m in messages]
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=oai_msgs,  # type: ignore[arg-type]
                **kwargs,
            )
        except Exception as exc:
            code = _classify(exc)
            self.logger.warning("OpenAI API error [%d %s]: %s", code, status_describe(code), exc)
            return NavigatorResult(
                result="",
                status={"status_code": code, "status_desc": status_describe(code), "status_detail": str(exc)},
                usage={},
                reference={},
            )
        choice = resp.choices[0]
        return NavigatorResult(
            result=choice.message.content or "",
            status={"status_code": StatusCode.OK, "status_desc": status_describe(StatusCode.OK), "status_detail": ""},
            usage=_parse_usage(resp.usage),
            reference={"model": resp.model, "finish_reason": choice.finish_reason},
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify(exc: Exception) -> int:
    try:
        from openai import AuthenticationError, RateLimitError
        if isinstance(exc, AuthenticationError):
            return StatusCode.UNAUTHORIZED
        if isinstance(exc, RateLimitError):
            return StatusCode.TOO_MANY_REQUESTS
    except ImportError:
        pass
    return StatusCode.INTERNAL_ERROR


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


def _parse_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return {}
    return {
        "prompt_tokens":     usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens":      usage.total_tokens,
    }
