"""BaseServer — abstract base for all LLM provider servers.

Infrastructure (normalisation, retry, logging) is applied automatically to any
method decorated with :func:`server_method`.  Concrete servers declare their
capabilities by decorating the relevant methods and contain only provider logic.

Example
-------
::

    class MyServer(BaseServer):
        provider = "my_llm"

        @server_method
        def chat(self, messages: list[Message], model: str, param: dict) -> NavigatorResult:
            # pure provider logic — messages already normalised, no retry needed
            ...

        @server_method
        def response(self, messages: list[Message], model: str, param: dict) -> NavigatorResult:
            ...
"""
from __future__ import annotations

import time
from abc import ABC
from functools import wraps
from typing import Any, ClassVar, Literal

from ai_navigator.state.data_class import Message, NavigatorResult
from ai_navigator.monitor.logger import get_logger
from ai_navigator.state.status import StatusCode


# ── Public decorator ──────────────────────────────────────────────────────────

def server_method(fn):
    """Mark a method as a server call.

    ``BaseServer.__init_subclass__`` wraps every decorated method with
    normalisation, retry, and logging.  The method itself should contain only
    provider logic, and may assume ``messages`` is already a ``list[Message]``.
    """
    fn._is_server_method = True
    return fn


# ── Shared result helpers ─────────────────────────────────────────────────────

def ok_result(text: str, usage: dict, reference: dict) -> NavigatorResult:
    return {
        "result": text,
        "status": {
            "status_code": StatusCode.OK,
            "status_desc": StatusCode.OK.desc,
            "status_detail": "",
        },
        "usage": usage,
        "reference": reference,
    }


def err_result(code: StatusCode, detail: str) -> NavigatorResult:
    return {
        "result": "",
        "status": {
            "status_code": code,
            "status_desc": code.desc,
            "status_detail": detail,
        },
        "usage": {},
        "reference": {},
    }


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise_messages(messages: list[Message] | str) -> list[Message]:
    """Coerce a bare string to a single-turn user message list."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return list(messages)


# ── Infrastructure wrapper ────────────────────────────────────────────────────

def _wrap_infrastructure(fn):
    """Wrap *fn* with normalisation, status-code-based retry, and logging.

    The wrapper:
    1. Normalises ``messages`` to ``list[Message]`` (standard ChatGPT format).
    2. Copies ``param``, extracts the reserved ``_retry_max`` key (if present),
       then passes the clean copy to the server method.
    3. Retries while ``status_code == 429``, up to ``effective`` attempts.

    The server method itself converts all provider exceptions into
    ``NavigatorResult`` — no exceptions cross the server boundary.
    """
    @wraps(fn)
    def wrapper(
        self,
        messages: list[Message] | str,
        model: str,
        param: dict,
    ) -> NavigatorResult:
        msgs = normalise_messages(messages)

        # Extract infrastructure key without mutating the caller's dict.
        effective_param = dict(param)
        retry_max = int(effective_param.pop("_retry_max", self._cred_retry_max))
        effective = min(self._cred_retry_max, retry_max)

        wait = self._retry_wait
        result: NavigatorResult | None = None
        for attempt in range(effective + 1):
            result = fn(self, msgs, model, effective_param)
            if result["status"]["status_code"] != StatusCode.TOO_MANY_REQUESTS:
                break
            if attempt < effective:
                self.logger.warning(
                    "%s rate limited (attempt %d/%d) — retrying in %.1fs",
                    fn.__name__, attempt + 1, effective + 1, wait,
                )
                time.sleep(wait)
                wait *= self._retry_backoff

        if result["status"]["status_code"] == StatusCode.OK:
            self.logger.debug("%s ok | model=%s", fn.__name__, model)
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
    - Implement ``chat`` and/or ``response`` with the fixed signature::

          def chat(self, messages: list[Message], model: str, param: dict) -> NavigatorResult:
              ...

      Decorate each with :func:`server_method`.
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
    ) -> None:
        self.model = model
        self.credentials = credentials
        from ai_navigator.param.const_configs import ConstConfigs
        self._cred_retry_max: int   = int(credentials.get("retry_max",     ConstConfigs.RETRY_MAX))
        self._retry_wait: float     = float(credentials.get("retry_wait",   ConstConfigs.RETRY_WAIT))
        self._retry_backoff: float  = float(credentials.get("retry_backoff", ConstConfigs.RETRY_BACKOFF))
        self._conversation: list[Message] = []
        self.logger = get_logger(f"{self.provider}.{model}")
        self._setup()

    def _setup(self) -> None:
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
