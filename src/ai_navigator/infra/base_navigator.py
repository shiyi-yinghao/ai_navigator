"""Base type definitions and BaseNavigator infrastructure class.

Type aliases
------------
ContentPart, Message, TokenUsage, Response are plain TypedDicts — no Pydantic.
All field access uses dict syntax: ``msg["role"]``, ``part["type"]``, etc.

BaseNavigator
-------------
Handles everything that is NOT the public call API:

- Loads credentials from ``CredentialsLoader`` at init.
- Builds the provider → server-class registry lazily.
- Resolves and caches server instances per ``model_name``.
- Preprocesses ``request_data`` dicts into ``list[Message]``.
- Emits per-stage log lines when ``ConstConfigs.LOGGING_STREAM`` is True.

Subclass :class:`BaseNavigator` and add ``chat()`` / ``response()`` (or any
other call methods) on top.
"""
from __future__ import annotations
from typing import Any, Union, TypedDict

from ai_navigator.infra.const_configs import ConstConfigs
from ai_navigator.infra.credentials import CredentialsLoader
from ai_navigator.infra.exceptions import AINavigatorError
from ai_navigator.monitor.logger import get_logger


# ── TypedDict models ──────────────────────────────────────────────────────────

class ContentPart(TypedDict, total=False):
    type: str        # "text" | "image_url" | "image_base64"
    text: str
    image_url: str
    image_data: str  # base64-encoded
    media_type: str


class Message(TypedDict):
    role: str        # "system" | "user" | "assistant"
    content: Union[str, list]


class TokenUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int
    total_tokens: int


class Response(TypedDict, total=False):
    content: str
    model: str
    usage: TokenUsage
    finish_reason: str
    raw: Any


# ── Factory helpers ───────────────────────────────────────────────────────────

def make_content_part(type_: str, **kwargs: Any) -> ContentPart:
    return {"type": type_, **kwargs}  # type: ignore[return-value]


def make_message(role: str, content: Union[str, list]) -> Message:
    return {"role": role, "content": content}  # type: ignore[return-value]


# ── Provider registry ─────────────────────────────────────────────────────────
# Lazy to avoid circular imports: server files import from this module.

def _build_registry() -> dict[str, Any]:
    from ai_navigator.server.anthropic_server import AnthropicServer
    from ai_navigator.server.openai_server import OpenAIServer
    from ai_navigator.server.gemini_server import GeminiServer
    return {cls.provider: cls for cls in [AnthropicServer, OpenAIServer, GeminiServer]}


# ── BaseNavigator ─────────────────────────────────────────────────────────────

class BaseNavigator:
    """Infrastructure base for Navigator.

    Credentials file structure (``ConstConfigs.CREDENTIALS_PATH``)::

        my_claude:
          - provider_type: anthropic
            model: claude-sonnet-4-6
            api_key: sk-ant-...
            max_tokens: 4096

        my_gpt4:
          - provider_type: openai
            model: gpt-4o
            api_key: sk-openai-...

    Each key is a ``model_name`` used in ``configs["model_name"]`` per call.
    The current implementation always picks the first credential in each list.
    """

    def __init__(self) -> None:
        self._all_creds = CredentialsLoader().fetch()
        self._registry = _build_registry()
        self._server_cache: dict[str, Any] = {}
        self._nav_configs = ConstConfigs()
        self._log = get_logger("navigator")

    # ── Server resolution ─────────────────────────────────────────────────────

    def _get_server(self, model_name: str) -> Any:
        """Return (and cache) the instantiated server for *model_name*."""
        if model_name in self._server_cache:
            return self._server_cache[model_name]

        creds_list = self._all_creds.get(model_name)
        if not creds_list or not isinstance(creds_list, list):
            raise AINavigatorError(
                f"model_name '{model_name}' not found in credentials. "
                f"Available: {list(self._all_creds)}"
            )

        cred = creds_list[0]
        provider_type = cred.get("provider_type", "")
        server_cls = self._registry.get(provider_type)
        if server_cls is None:
            raise AINavigatorError(
                f"Unknown provider_type '{provider_type}' for model_name "
                f"'{model_name}'. Known providers: {list(self._registry)}"
            )

        model = cred.get("model", "")
        if not model:
            raise AINavigatorError(
                f"credentials for model_name '{model_name}' is missing 'model'."
            )

        server = server_cls(model=model, credentials=cred)
        self._server_cache[model_name] = server
        return server

    # ── Request preprocessing ─────────────────────────────────────────────────

    def _preprocess(self, request_data: dict) -> list[Message]:
        """Convert a request_data dict into a list[Message]."""
        rtype = request_data.get("type", "")

        if rtype == "message":
            content = request_data["content"]
            if isinstance(content, str):
                return [make_message("user", content)]
            return [{"role": "user", "content": content}]

        if rtype == "conversation":
            return list(request_data["messages"])

        if rtype == "prompt":
            from ai_navigator.conf_parser.prompt import PromptBuilder
            builder = PromptBuilder(request_data["template"])
            return builder.build(data_dict=request_data.get("data_dict", {}))

        raise AINavigatorError(
            f"Unknown request_data type '{rtype}'. "
            "Expected 'message', 'conversation', or 'prompt'."
        )

    # ── Stage logging ─────────────────────────────────────────────────────────

    def _log_stage(self, stage: str, data: Any = None) -> None:
        if self._nav_configs.LOGGING_STREAM:
            preview = repr(data)[:200] if data is not None else ""
            self._log.info("[%s] %s", stage, preview)