from __future__ import annotations
from typing import Any, ClassVar, Iterator

from ai_navigator.state.data_class import Message, NavigatorResult, TokenUsage
from ai_navigator.state.status import StatusCode
from ai_navigator.server.base_server import BaseServer, server_method, ok_result, err_result, normalise_messages


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

    def _setup(self) -> None:
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
        messages: list[Message],
        model: str,
        param: dict,
    ) -> NavigatorResult:
        """Standard chat completion."""
        gemini_msgs, system_text = _to_gemini_messages(messages)
        client = self._build_client(model, system=system_text)
        try:
            resp = client.generate_content(gemini_msgs, **param)
        except Exception as exc:
            code = _classify(exc)
            self.logger.warning("Gemini [%d]: %s", code, exc)
            return err_result(code, str(exc))
        return ok_result(resp.text, _parse_usage(resp), {"model": model})

    @server_method
    def response(
        self,
        messages: list[Message],
        model: str,
        param: dict,
    ) -> NavigatorResult:
        """Structured-output completion (JSON mode).

        Pass ``response_schema`` in param to enforce a specific schema via
        Gemini's ``generation_config.response_schema``.
        """
        effective = dict(param)
        response_schema = effective.pop("response_schema", None)
        gemini_msgs, system_text = _to_gemini_messages(messages)
        json_cfg_kwargs: dict[str, Any] = {"response_mime_type": "application/json"}
        if response_schema is not None:
            json_cfg_kwargs["response_schema"] = response_schema
        json_gen_config = self._genai.GenerationConfig(**json_cfg_kwargs)
        client = self._build_client(model, system=system_text, generation_config=json_gen_config)
        try:
            resp = client.generate_content(gemini_msgs, **effective)
        except Exception as exc:
            code = _classify(exc)
            self.logger.warning("Gemini [%d]: %s", code, exc)
            return err_result(code, str(exc))
        return ok_result(resp.text, _parse_usage(resp), {"model": model})

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[Message] | str,
        model: str,
        param: dict,
    ) -> Iterator[str]:
        """Token-by-token streaming."""
        gemini_msgs, system_text = _to_gemini_messages(normalise_messages(messages))
        client = self._build_client(model, system=system_text)
        for chunk in client.generate_content(gemini_msgs, stream=True, **param):
            if chunk.text:
                yield chunk.text

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_client(
        self,
        model: str,
        system: str | None = None,
        generation_config: Any = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model_name": model,
            "generation_config": generation_config or self._gen_config,
        }
        if system:
            kwargs["system_instruction"] = system
        return self._genai.GenerativeModel(**kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify(exc: Exception) -> StatusCode:
    msg = str(exc).lower()
    if any(k in msg for k in ("api_key", "permission", "unauthorized", "unauthenticated")):
        return StatusCode.UNAUTHORIZED
    if any(k in msg for k in ("quota", "resource_exhausted", "rate_limit")):
        return StatusCode.TOO_MANY_REQUESTS
    return StatusCode.INTERNAL_ERROR


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
