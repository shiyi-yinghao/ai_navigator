"""BaseNavigator — service layer housing credentials, dispatch, and call methods.

Entry Points (replace)
----------------------
Third-party packages can replace BaseNavigator entirely by registering a
subclass under ``ai_navigator.navigator``::

    # their pyproject.toml
    [project.entry-points."ai_navigator.navigator"]
    custom = "my_package.nav:MyNavigator"

The registered class must be a subclass of :class:`BaseNavigator`.  Only ONE
replacement is used.  The replacement is discovered once and cached.

Alternatively, pass your credentials loader or extra servers at construction
time — those take effect regardless of which class is active.
"""
from __future__ import annotations
import logging
from importlib.metadata import entry_points
from typing import Any

from ai_navigator.infra.types import Message, make_message
from ai_navigator.infra.exceptions import AINavigatorError
from ai_navigator.param.const_configs import ConstConfigs
from ai_navigator.param.credentials import get_credentials_class
from ai_navigator.monitor.logger import get_logger
from ai_navigator.server.registry import build_registry

_log_module = logging.getLogger("ai_navigator.service.navigator")


class BaseNavigator:
    """Infrastructure + public call methods.

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

    """

    def __init__(self) -> None:
        self._all_creds = get_credentials_class()().fetch()
        self._registry = build_registry()
        self._server_cache: dict[str, Any] = {}
        self._nav_configs = ConstConfigs()
        self._log = get_logger("navigator")

    # ── Public call methods ───────────────────────────────────────────────────

    def chat(
        self,
        request_data: dict,
        params: dict | None = None,
        configs: dict | None = None,
    ) -> Any:
        """Send a chat request.

        Parameters
        ----------
        request_data:
            ``{"message": str | list}``
            ``{"conversation": list[Message]}``
            ``{"prompt": list, "data_dict": dict}``
        params:
            Provider call parameters (temperature, max_tokens, …).
        configs:
            Must contain ``model_name``.
        """
        params = params or {}
        configs = configs or {}
        self._log_stage("request_receive", request_data)
        model_name = configs.get("model_name", "")
        if not model_name:
            raise ValueError("configs must contain 'model_name'.")
        server = self._get_server(model_name)
        messages = self._preprocess(request_data)
        self._log_stage("request_preprocess", messages)
        result = server.chat(messages, **params)
        self._log_stage("request_executed", result)
        self._log_stage("request_returned", result)
        return result

    def response(
        self,
        request_data: dict,
        params: dict | None = None,
        configs: dict | None = None,
    ) -> Any:
        """Send a structured-output request.

        Parameters
        ----------
        request_data:
            Same shapes as :meth:`chat`.
        params:
            Provider call parameters (response_format, schema, …).
        configs:
            Must contain ``model_name``.
        """
        params = params or {}
        configs = configs or {}
        self._log_stage("request_receive", request_data)
        model_name = configs.get("model_name", "")
        if not model_name:
            raise ValueError("configs must contain 'model_name'.")
        server = self._get_server(model_name)
        messages = self._preprocess(request_data)
        self._log_stage("request_preprocess", messages)
        result = server.response(messages, **params)
        self._log_stage("request_executed", result)
        self._log_stage("request_returned", result)
        return result

    # ── Server resolution ─────────────────────────────────────────────────────

    def _get_server(self, model_name: str) -> Any:
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
        if "message" in request_data:
            content = request_data["message"]
            if isinstance(content, str):
                return [make_message("user", content)]
            return [{"role": "user", "content": content}]

        if "conversation" in request_data:
            return list(request_data["conversation"])

        if "prompt" in request_data:
            from ai_navigator.conf_parser.prompt import PromptBuilder
            builder = PromptBuilder(request_data["prompt"])
            return builder.build(data_dict=request_data.get("data_dict", {}))

        raise AINavigatorError(
            "request_data must contain a 'message', 'conversation', or 'prompt' key."
        )

    # ── Stage logging ─────────────────────────────────────────────────────────

    def _log_stage(self, stage: str, data: Any = None) -> None:
        if self._nav_configs.LOGGING_STREAM:
            preview = repr(data)[:200] if data is not None else ""
            self._log.info("[%s] %s", stage, preview)


# ── Entry Points discovery ────────────────────────────────────────────────────

_navigator_class_cache: type | None = None


def get_navigator_class() -> type:
    """Return the active navigator class.

    Checks ``ai_navigator.navigator`` entry points first; falls back to
    :class:`BaseNavigator`.  Only the first registered plugin is used.
    """
    global _navigator_class_cache
    if _navigator_class_cache is not None:
        return _navigator_class_cache

    eps = list(entry_points(group="ai_navigator.navigator"))
    if not eps:
        _navigator_class_cache = BaseNavigator
        return BaseNavigator

    if len(eps) > 1:
        _log_module.warning(
            "multiple navigator plugins found (%s), using first: %s",
            [e.name for e in eps], eps[0].name,
        )

    try:
        cls = eps[0].load()
        _log_module.info("using navigator plugin: %s", eps[0].name)
        _navigator_class_cache = cls
        return cls
    except Exception as exc:
        _log_module.warning("navigator plugin '%s' failed: %s — using default", eps[0].name, exc)
        _navigator_class_cache = BaseNavigator
        return BaseNavigator
