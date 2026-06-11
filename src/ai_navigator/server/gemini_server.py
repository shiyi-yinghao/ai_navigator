from __future__ import annotations
from typing import Any, ClassVar, Iterator

from ai_navigator.infra.exceptions import AuthenticationError, ProviderError, RateLimitError
from ai_navigator.infra.types import Message, Response, TokenUsage
from ai_navigator.server.base_server import BaseServer


class GeminiServer(BaseServer):
    """Server for Google Gemini models: gemini-2.0-flash, gemini-1.5-pro, …

    Credentials dict keys
    ---------------------
    - ``api_key``           (required) — Google AI API key.
    - ``generation_config`` (optional) — Dict passed to ``GenerationConfig``.
    """

    provider: ClassVar[str] = "gemini"
    _supported_methods: ClassVar[list[str]] = ["chat", "response"]

    def _setup(self, **kwargs: Any) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Google Generative AI SDK not installed. "
                "Run: pip install ai-navigator[gemini]"
            ) from exc
        api_key = self.credentials.get("api_key")
        if api_key:
            genai.configure(api_key=api_key)
        self._genai = genai
        gen_cfg = self.credentials.get("generation_config", {})
        self._gen_config = genai.GenerationConfig(**gen_cfg) if gen_cfg else None

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
        """Structured-output completion (JSON mode).

        Pass ``response_schema`` in kwargs to enforce a specific schema via
        Gemini's ``generation_config.response_schema``.
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
        gemini_msgs, system = _to_gemini_messages(messages)
        client = self._build_client(system=system)
        try:
            resp = client.generate_content(gemini_msgs, **kwargs)
        except Exception as exc:
            _raise_gemini_error(exc)
        return {
            "content": resp.text,
            "model": self.model,
            "usage": _parse_usage(resp),
            "raw": resp,
        }

    def _response(self, messages: list[Message], **kwargs: Any) -> Response:
        """Structured output via ``response_mime_type = application/json``."""
        response_schema = kwargs.pop("response_schema", None)
        gemini_msgs, system = _to_gemini_messages(messages)
        json_config_kwargs: dict[str, Any] = {"response_mime_type": "application/json"}
        if response_schema is not None:
            json_config_kwargs["response_schema"] = response_schema
        json_gen_config = self._genai.GenerationConfig(**json_config_kwargs)
        client = self._build_client(system=system, generation_config=json_gen_config)
        try:
            resp = client.generate_content(gemini_msgs, **kwargs)
        except Exception as exc:
            _raise_gemini_error(exc)
        return {
            "content": resp.text,
            "model": self.model,
            "usage": _parse_usage(resp),
            "raw": resp,
        }

    def _stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        gemini_msgs, system = _to_gemini_messages(messages)
        client = self._build_client(system=system)
        for chunk in client.generate_content(gemini_msgs, stream=True, **kwargs):
            if chunk.text:
                yield chunk.text

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_client(
        self,
        system: str | None = None,
        generation_config: Any = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model_name": self.model,
            "generation_config": generation_config or self._gen_config,
        }
        if system:
            kwargs["system_instruction"] = system
        return self._genai.GenerativeModel(**kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_gemini_messages(
    messages: list[Message],
) -> tuple[list[dict[str, Any]], str | None]:
    system: str | None = None
    turns: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"] if isinstance(msg["content"], str) else ""
            continue
        role = "user" if msg["role"] == "user" else "model"
        content = msg["content"]
        if isinstance(content, str):
            turns.append({"role": role, "parts": [content]})
        else:
            text_parts = [p.get("text") for p in content if p.get("type") == "text" and p.get("text")]
            turns.append({"role": role, "parts": text_parts})
    return turns, system


def _parse_usage(resp: Any) -> TokenUsage | None:
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        return None
    return {
        "prompt_tokens": getattr(meta, "prompt_token_count", 0),
        "completion_tokens": getattr(meta, "candidates_token_count", 0),
        "total_tokens": getattr(meta, "total_token_count", 0),
    }


def _raise_gemini_error(exc: Exception) -> None:
    msg = str(exc).lower()
    if any(k in msg for k in ("api_key", "permission", "unauthorized", "unauthenticated")):
        raise AuthenticationError(str(exc), "gemini") from exc
    if any(k in msg for k in ("quota", "resource_exhausted", "rate_limit")):
        raise RateLimitError(str(exc), "gemini") from exc
    raise ProviderError(str(exc), "gemini") from exc
