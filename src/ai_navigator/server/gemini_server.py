from __future__ import annotations
from typing import Any, ClassVar, Iterator

from ai_navigator.infra.types import Message, NavigatorResult, TokenUsage
from ai_navigator.infra.status_codes import SC, describe as status_describe
from ai_navigator.server.base_server import BaseServer, server_method


class GeminiServer(BaseServer):
    """Server for Google Gemini models: gemini-2.0-flash, gemini-1.5-pro, …

    Credentials dict keys
    ---------------------
    - ``api_key``           (required) — Google AI API key.
    - ``generation_config`` (optional) — Dict passed to ``GenerationConfig``.

    Status codes returned
    ----------------------
    200  Ok
    401  Unauthorized  (invalid API key / permission denied)
    429  Too Many Requests  (quota exhausted; retried automatically)
    500  Internal Server Error  (unexpected SDK / network error)
    """

    provider: ClassVar[str] = "gemini"

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

    # ── Server methods ────────────────────────────────────────────────────────

    @server_method
    def chat(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> NavigatorResult:
        """Standard chat completion."""
        gemini_msgs, system_text = _to_gemini_messages(self._normalise(messages, system))
        client = self._build_client(system=system_text)
        try:
            resp = client.generate_content(gemini_msgs, **kwargs)
        except Exception as exc:
            code = _classify(exc)
            self.logger.warning("Gemini API error [%d %s]: %s", code, status_describe(code), exc)
            return NavigatorResult(
                result="",
                status={"status_code": code, "status_desc": status_describe(code), "status_detail": str(exc)},
                usage={},
                reference={},
            )
        return NavigatorResult(
            result=resp.text,
            status={"status_code": SC.OK, "status_desc": status_describe(SC.OK), "status_detail": ""},
            usage=_parse_usage(resp),
            reference={"model": self.model},
        )

    @server_method
    def response(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> NavigatorResult:
        """Structured-output completion (JSON mode).

        Pass ``response_schema`` in kwargs to enforce a specific schema via
        Gemini's ``generation_config.response_schema``.
        """
        response_schema = kwargs.pop("response_schema", None)
        gemini_msgs, system_text = _to_gemini_messages(self._normalise(messages, system))
        json_config_kwargs: dict[str, Any] = {"response_mime_type": "application/json"}
        if response_schema is not None:
            json_config_kwargs["response_schema"] = response_schema
        json_gen_config = self._genai.GenerationConfig(**json_config_kwargs)
        client = self._build_client(system=system_text, generation_config=json_gen_config)
        try:
            resp = client.generate_content(gemini_msgs, **kwargs)
        except Exception as exc:
            code = _classify(exc)
            self.logger.warning("Gemini API error [%d %s]: %s", code, status_describe(code), exc)
            return NavigatorResult(
                result="",
                status={"status_code": code, "status_desc": status_describe(code), "status_detail": str(exc)},
                usage={},
                reference={},
            )
        return NavigatorResult(
            result=resp.text,
            status={"status_code": SC.OK, "status_desc": status_describe(SC.OK), "status_detail": ""},
            usage=_parse_usage(resp),
            reference={"model": self.model},
        )

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[Message] | str,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Token-by-token streaming."""
        gemini_msgs, system_text = _to_gemini_messages(self._normalise(messages, system))
        client = self._build_client(system=system_text)
        for chunk in client.generate_content(gemini_msgs, stream=True, **kwargs):
            if chunk.text:
                yield chunk.text

    # ── Private helpers ───────────────────────────────────────────────────────

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

def _classify(exc: Exception) -> int:
    msg = str(exc).lower()
    if any(k in msg for k in ("api_key", "permission", "unauthorized", "unauthenticated")):
        return SC.UNAUTHORIZED
    if any(k in msg for k in ("quota", "resource_exhausted", "rate_limit")):
        return SC.TOO_MANY_REQUESTS
    return SC.INTERNAL_ERROR


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


def _parse_usage(resp: Any) -> TokenUsage:
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        return {}
    return {
        "prompt_tokens":     getattr(meta, "prompt_token_count", 0),
        "completion_tokens": getattr(meta, "candidates_token_count", 0),
        "total_tokens":      getattr(meta, "total_token_count", 0),
    }
