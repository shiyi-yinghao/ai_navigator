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

from ai_navigator.infra.types import Message, NavigatorResult, make_message
from ai_navigator.infra.exceptions import AINavigatorError
from ai_navigator.param.const_configs import ConstConfigs
from ai_navigator.param.credentials import get_credentials_class
from ai_navigator.monitor.logger import get_logger
from ai_navigator.monitor.traffic import traffic_monitor
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
            retry_max: 3        # optional, caps retries for this account

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

    @traffic_monitor
    def chat(
        self,
        request_data: dict,
        params: dict | None = None,
        configs: dict | None = None,
    ) -> NavigatorResult:
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
            Must contain ``model_name``.  Optional: ``user`` (default
            ``"default"``), ``retry_max`` (capped by credentials ``retry_max``).

        Returns
        -------
        NavigatorResult
            ``result``  — server :class:`~ai_navigator.infra.types.Response`
            ``usage``   — token usage
            ``status``  — ``{"ok": True, ...}`` on success
        """
        params = params or {}
        configs = configs or {}
        self._log_stage("request_receive", request_data)

        model_name = configs.get("model_name", "")
        if not model_name:
            raise ValueError("configs must contain 'model_name'.")

        server  = self._get_server(model_name)
        messages = self._preprocess(request_data)
        self._log_stage("request_preprocess", messages)

        effective_retry = self._effective_retry(model_name, configs)
        server_result = server.chat(messages, _retry_max=effective_retry, **params)
        self._log_stage("request_executed", server_result)

        usage = server_result.get("usage", {}) if isinstance(server_result, dict) else {}
        return NavigatorResult(
            result=server_result,
            usage=usage,
            status={"ok": True, "error": None, "error_type": None},
        )

    @traffic_monitor
    def response(
        self,
        request_data: dict,
        params: dict | None = None,
        configs: dict | None = None,
    ) -> NavigatorResult:
        """Send a structured-output request.

        Parameters
        ----------
        request_data:
            Same shapes as :meth:`chat`.
        params:
            Provider call parameters (response_format, schema, …).
        configs:
            Must contain ``model_name``.  Optional: ``user``, ``retry_max``.

        Returns
        -------
        NavigatorResult
            ``result``  — server :class:`~ai_navigator.infra.types.Response`
            ``usage``   — token usage
            ``status``  — ``{"ok": True, ...}`` on success
        """
        params = params or {}
        configs = configs or {}
        self._log_stage("request_receive", request_data)

        model_name = configs.get("model_name", "")
        if not model_name:
            raise ValueError("configs must contain 'model_name'.")

        server   = self._get_server(model_name)
        messages = self._preprocess(request_data)
        self._log_stage("request_preprocess", messages)

        effective_retry = self._effective_retry(model_name, configs)
        server_result = server.response(messages, _retry_max=effective_retry, **params)
        self._log_stage("request_executed", server_result)

        usage = server_result.get("usage", {}) if isinstance(server_result, dict) else {}
        return NavigatorResult(
            result=server_result,
            usage=usage,
            status={"ok": True, "error": None, "error_type": None},
        )

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

    def _get_account_name(self, model_name: str) -> str:
        creds_list = self._all_creds.get(model_name, [{}])
        cred = creds_list[0] if creds_list else {}
        return cred.get("account_name", model_name)

    def _effective_retry(self, model_name: str, configs: dict) -> int:
        """Return min(credentials retry_max, configs retry_max)."""
        creds_list = self._all_creds.get(model_name, [{}])
        cred = creds_list[0] if creds_list else {}
        cred_max    = int(cred.get("retry_max",     ConstConfigs.RETRY_MAX))
        request_max = int(configs.get("retry_max",  ConstConfigs.RETRY_MAX))
        return min(cred_max, request_max)

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
