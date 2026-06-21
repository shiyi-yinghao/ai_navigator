"""BaseServer — abstract base for all LLM provider servers.

Infrastructure (retry, logging) is applied automatically to any method
decorated with :func:`server_method`.  Concrete servers declare their
capabilities by decorating the relevant methods — no other coupling to
``BaseServer`` is required.

Example
-------
::

    class MyServer(BaseServer):
        provider = "my_llm"

        @server_method
        def chat(self, messages, **kwargs) -> Response:
            # pure provider logic — no retry, no logging
            ...

        @server_method
        def response(self, messages, **kwargs) -> Response:
            ...
"""
from __future__ import annotations

import time
from abc import ABC
from functools import wraps
from typing import Any, ClassVar, Literal

from ai_navigator.infra.types import Message, NavigatorResult
from ai_navigator.monitor.logger import get_logger


# ── Public decorator ──────────────────────────────────────────────────────────

def server_method(fn):
    """Mark a method as a server call.

    ``BaseServer.__init_subclass__`` wraps every decorated method with retry
    and logging.  The method itself should contain only provider logic.
    """
    fn._is_server_method = True
    return fn


# ── Infrastructure wrapper ────────────────────────────────────────────────────

def _wrap_infrastructure(fn):
    """Wrap *fn* with status-code-based retry and logging.

    Retries while the returned :class:`~ai_navigator.infra.types.NavigatorResult`
    carries ``status_code == 429`` (Too Many Requests), up to the effective
    retry limit: ``min(credentials.retry_max, _retry_max kwarg)``.

    The server method itself is responsible for converting all provider
    exceptions into ``NavigatorResult`` — no exceptions cross the server
    boundary under normal operation.
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        retry_max = kwargs.pop("_retry_max", self._cred_retry_max)
        effective = min(self._cred_retry_max, int(retry_max))

        wait = self._retry_wait
        result: NavigatorResult | None = None
        for attempt in range(effective + 1):
            result = fn(self, *args, **kwargs)
            if result["status"]["status_code"] != 429:
                break
            if attempt < effective:
                self.logger.warning(
                    "%s rate limited (attempt %d/%d) — retrying in %.1fs",
                    fn.__name__, attempt + 1, effective + 1, wait,
                )
                time.sleep(wait)
                wait *= self._retry_backoff

        if result["status"]["status_code"] == 200:
            self.logger.debug("%s ok | model=%s", fn.__name__, self.model)
        return result

    wrapper._is_server_method = True
    wrapper._base_wrapped = True
    return wrapper


# ── BaseServer ────────────────────────────────────────────────────────────────

class BaseServer(ABC):
    """Abstract base — provides infrastructure; knows nothing about methods.

    Subclasses
    ----------
    - Set ``provider`` as a class variable.
    - Override ``_setup(**kwargs)`` to initialise the SDK client.
    - Implement provider methods and decorate each with :func:`server_method`.
      ``_supported_methods`` is populated automatically from those decorators.
    """

    provider: ClassVar[str] = "unknown"
    _supported_methods: ClassVar[list[str]] = []

    # ── Auto-wrap on subclass definition ─────────────────────────────────────

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        new_methods: list[str] = []
        for name, obj in list(cls.__dict__.items()):
            if callable(obj) and getattr(obj, "_is_server_method", False) \
                    and not getattr(obj, "_base_wrapped", False):
                setattr(cls, name, _wrap_infrastructure(obj))
                new_methods.append(name)
        if new_methods:
            parent = list(getattr(cls, "_supported_methods", []))
            cls._supported_methods = parent + [m for m in new_methods if m not in parent]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        model: str,
        credentials: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.credentials = credentials
        from ai_navigator.param.const_configs import ConstConfigs
        self._cred_retry_max: int   = int(credentials.get("retry_max",     ConstConfigs.RETRY_MAX))
        self._retry_wait: float     = float(credentials.get("retry_wait",   ConstConfigs.RETRY_WAIT))
        self._retry_backoff: float  = float(credentials.get("retry_backoff", ConstConfigs.RETRY_BACKOFF))
        self._conversation: list[Message] = []
        self.logger = get_logger(f"{self.provider}.{model}")
        self._setup(**kwargs)

    def _setup(self, **kwargs: Any) -> None:
        """Initialise the provider SDK client (called once at end of ``__init__``)."""

    # ── Method-support query ──────────────────────────────────────────────────

    def supports(self, method: str) -> bool:
        """Return True if this server implements *method*."""
        return method in self._supported_methods

    # ── Conversation management ───────────────────────────────────────────────

    def add_message(
        self,
        role: Literal["system", "user", "assistant"],
        content: str,
    ) -> None:
        self._conversation.append({"role": role, "content": content})

    def set_system(self, text: str) -> None:
        self._conversation = [m for m in self._conversation if m["role"] != "system"]
        self._conversation.insert(0, {"role": "system", "content": text})

    def reset_conversation(self) -> None:
        self._conversation.clear()

    @property
    def conversation(self) -> list[Message]:
        return list(self._conversation)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _normalise(
        self,
        messages: list[Message] | str,
        system: str | None = None,
    ) -> list[Message]:
        """Coerce a bare string or message list; prepend system message if given."""
        if isinstance(messages, str):
            msgs: list[Message] = [{"role": "user", "content": messages}]
        else:
            msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
        return msgs