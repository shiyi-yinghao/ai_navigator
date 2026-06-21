from __future__ import annotations
from typing import Any, ClassVar, Iterator

from ai_navigator.state.data_class import ContentPart, Message, NavigatorResult, TokenUsage
from ai_navigator.state.status import StatusCode
from ai_navigator.server.base_server import BaseServer, server_method, ok_result, err_result, normalise_messages


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

    def _setup(self) -> None:
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
        messages: list[Message],
        model: str,
        param: dict,
    ) -> NavigatorResult:
        """Standard chat completion."""
        oai_msgs = [_to_openai_message(m) for m in messages]
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=oai_msgs,  # type: ignore[arg-type]
                **param,
            )
        except Exception as exc:
            code = _classify(exc)
            self.logger.warning("OpenAI [%d]: %s", code, exc)
            return err_result(code, str(exc))
        choice = resp.choices[0]
        return ok_result(
            choice.message.content or "",
            _parse_usage(resp.usage),
            {"model": resp.model, "finish_reason": choice.finish_reason},
        )

    @server_method
    def response(
        self,
        messages: list[Message],
        model: str,
        param: dict,
    ) -> NavigatorResult:
        """Structured-output completion.

        Pass ``response_format`` in param to control the output schema, e.g.::

            server.response(msgs, model, {"response_format": {"type": "json_object"}})
            server.response(msgs, model, {"response_format": {"type": "json_schema",
                                                               "json_schema": schema_dict}})
        """
        effective = dict(param)
        effective.setdefault("response_format", {"type": "json_object"})
        oai_msgs = [_to_openai_message(m) for m in messages]
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=oai_msgs,  # type: ignore[arg-type]
                **effective,
            )
        except Exception as exc:
            code = _classify(exc)
            self.logger.warning("OpenAI [%d]: %s", code, exc)
            return err_result(code, str(exc))
        choice = resp.choices[0]
        return ok_result(
            choice.message.content or "",
            _parse_usage(resp.usage),
            {"model": resp.model, "finish_reason": choice.finish_reason},
        )

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[Message] | str,
        model: str,
        param: dict,
    ) -> Iterator[str]:
        """Token-by-token streaming."""
        oai_msgs = [_to_openai_message(m) for m in normalise_messages(messages)]
        stream = self._client.chat.completions.create(
            model=model,
            messages=oai_msgs,  # type: ignore[arg-type]
            stream=True,
            **param,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify(exc: Exception) -> StatusCode:
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
