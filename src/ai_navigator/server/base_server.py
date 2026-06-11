from __future__ import annotations
from abc import ABC
from typing import Any, ClassVar, Iterator, Literal

from ai_navigator.infra.types import Message, Response
from ai_navigator.monitor.logger import get_logger
from ai_navigator.server.retry import RetryPolicy


class BaseServer(ABC):
    """Abstract base for all LLM provider servers.

    Responsibilities
    ----------------
    - Store credentials as an opaque dict (never parsed here).
    - Maintain the generic conversation state (list[Message]).
    - Declare the supported call methods via ``_supported_methods``.
    - Centralise retry, logging, and storage via ``_invoke``.

    Subclasses
    ----------
    - Set ``provider`` and ``_supported_methods`` as class variables.
    - Override ``_setup(**kwargs)`` to initialise the SDK client using
      values read from ``self.credentials``.
    - Implement private ``_chat`` and/or ``_response`` methods.
    - Expose whatever public API they need (``chat``, ``response``,
      ``stream``, …); BaseServer itself does NOT define those.
    """

    provider: ClassVar[str] = "unknown"
    _supported_methods: ClassVar[list[str]] = []

    def __init__(
        self,
        model: str,
        credentials: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.credentials = credentials
        from ai_navigator.param.const_configs import ConstConfigs
        self._cred_retry_max: int = int(credentials.get("retry_max", ConstConfigs.RETRY_MAX))
        self._retry_wait: float = float(credentials.get("retry_wait", ConstConfigs.RETRY_WAIT))
        self._retry_backoff: float = float(credentials.get("retry_backoff", ConstConfigs.RETRY_BACKOFF))
        self._conversation: list[Message] = []
        self.logger = get_logger(f"{self.provider}.{model}")
        self._setup(**kwargs)

    # ── Post-init hook ────────────────────────────────────────────────────────

    def _setup(self, **kwargs: Any) -> None:
        """Initialise the provider SDK client.

        Called once at the end of ``__init__``.  Subclasses read whatever
        keys they need from ``self.credentials`` here and nowhere else.
        """

    # ── Conversation management ───────────────────────────────────────────────

    def add_message(
        self,
        role: Literal["system", "user", "assistant"],
        content: str,
    ) -> None:
        """Append a message to the current conversation."""
        self._conversation.append({"role": role, "content": content})

    def set_system(self, text: str) -> None:
        """Replace (or insert) the system message at position 0."""
        self._conversation = [m for m in self._conversation if m["role"] != "system"]
        self._conversation.insert(0, {"role": "system", "content": text})

    def reset_conversation(self) -> None:
        """Clear all messages from the current conversation."""
        self._conversation.clear()

    @property
    def conversation(self) -> list[Message]:
        """Return a snapshot of the current conversation (read-only copy)."""
        return list(self._conversation)

    # ── Method-support contract ───────────────────────────────────────────────

    def supports(self, method: str) -> bool:
        """Return True if this server implements the given method name."""
        return method in self._supported_methods

    def _require_method(self, method: str) -> None:
        if not self.supports(method):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support '{method}'. "
                f"Supported: {self._supported_methods}"
            )

    # ── Central dispatcher ────────────────────────────────────────────────────

    def _invoke(
        self,
        method: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> Response:
        """Validate support, apply retry logic, dispatch to ``_{method}``.

        Pops the reserved ``_retry_max`` kwarg (set by
        :class:`~ai_navigator.service.base_navigator.BaseNavigator`) before
        forwarding to the provider implementation.  The effective retry count
        is ``min(credentials.retry_max, _retry_max)``.
        """
        self._require_method(method)
        request_retry_max: int | None = kwargs.pop("_retry_max", None)
        effective_max = (
            min(self._cred_retry_max, request_retry_max)
            if request_retry_max is not None
            else self._cred_retry_max
        )
        fn = getattr(self, f"_{method}")
        policy = RetryPolicy(
            max_retries=effective_max,
            initial_wait=self._retry_wait,
            backoff=self._retry_backoff,
        )
        response = policy.execute(fn, messages, **kwargs)
        self.logger.debug(
            "%s ok | model=%s finish=%s tokens=%s",
            method,
            self.model,
            response.get("finish_reason"),
            response.get("usage"),
        )
        return response

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _normalise(
        self,
        messages: list[Message] | str,
        system: str | None = None,
    ) -> list[Message]:
        """Coerce a bare string or message list; prepend a system message if given."""
        if isinstance(messages, str):
            msgs: list[Message] = [{"role": "user", "content": messages}]
        else:
            msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
        return msgs

