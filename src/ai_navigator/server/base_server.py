from __future__ import annotations
import time
from abc import ABC
from typing import Any, ClassVar, Iterator, Literal

from ai_navigator.infra.exceptions import AINavigatorError, RateLimitError
from ai_navigator.infra.base_navigator import Message, Response
from ai_navigator.monitor.logger import get_logger


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
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.credentials = credentials
        self._max_retries = max_retries
        self._retry_delay = retry_delay
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

        All public call methods (``chat``, ``response``, …) on concrete
        servers should go through here so that retry, logging, and storage
        are applied uniformly.
        """
        self._require_method(method)
        fn = getattr(self, f"_{method}")
        response = self._with_retry(fn, messages, **kwargs)
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

    def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Response:
        last_exc: RateLimitError | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except RateLimitError as exc:
                delay = exc.retry_after or self._retry_delay * attempt
                self.logger.warning(
                    "rate limit on attempt %d/%d — retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    delay,
                )
                time.sleep(delay)
                last_exc = exc
            except AINavigatorError:
                raise
            except Exception as exc:
                self.logger.error("unexpected provider error: %s", exc)
                raise
        assert last_exc is not None
        raise last_exc
