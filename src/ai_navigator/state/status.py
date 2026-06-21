"""Status code registry and StatusDetail — the core error-reporting types.

``StatusCode`` is an ``int`` subclass.  Every registered code is a singleton
instance, so it behaves like an ``IntEnum`` member while supporting runtime
extension::

    isinstance(StatusCode.OK, int)          # True
    isinstance(StatusCode.OK, StatusCode)   # True
    StatusCode.OK == 200                    # True

``StatusDetail`` is the ``status`` block inside every ``NavigatorResult``::

    result["status"]  # StatusDetail dict
    result["status"]["status_code"]   # StatusCode instance (== 200, 429, …)
    result["status"]["status_desc"]   # short label — independent of describe()
    result["status"]["status_detail"] # full error string; "" on success

Referencing codes
-----------------
Named attributes::

    StatusCode.OK                  # → StatusCode(200)
    StatusCode.TOO_MANY_REQUESTS   # → StatusCode(429)
    StatusCode.CONTEXT_LIMIT       # → StatusCode(601)

Integer lookup — validates registration and returns the singleton instance::

    StatusCode[200]   # → StatusCode.OK
    StatusCode[429]   # → StatusCode.TOO_MANY_REQUESTS
    StatusCode[709]   # → StatusCode(709)  (if a plugin registered 709)

Building a ``StatusDetail``::

    status: StatusDetail = {
        "status_code":   StatusCode[429],
        "status_desc":   "Gemini quota exceeded",   # caller's own label
        "status_detail": str(exc),
    }

``status_desc`` is **independent** of :func:`describe` — callers may supply
any description they choose.

Registering custom codes via entry points
-----------------------------------------
Expose a ``dict[int, str]`` (or a zero-arg callable returning one) under the
group ``ai_navigator.status_codes``::

    # pyproject.toml
    [project.entry-points."ai_navigator.status_codes"]
    my_plugin = "my_plugin.status:CODES"

    # my_plugin/status.py
    CODES = {709: "Custom Timeout", 710: "Billing Limit Exceeded"}

In-process registration::

    MY_TIMEOUT = StatusCode.register(709, "Custom Timeout")
    # MY_TIMEOUT is a StatusCode(709) instance
"""
from __future__ import annotations

import logging
from importlib.metadata import entry_points
from threading import Lock
from typing import ClassVar, TypedDict

_log = logging.getLogger("ai_navigator.state.status")


# ── Metaclass ─────────────────────────────────────────────────────────────────

class _StatusCodeMeta(type):
    """Metaclass that adds ``StatusCode[code]`` lookup."""

    def __getitem__(cls, code: int) -> StatusCode:
        """Return the registered ``StatusCode`` instance for *code*.

        Raises :class:`KeyError` if *code* is not registered.
        """
        cls._ensure_loaded()
        c = cls._registry.get(int(code))
        if c is None:
            raise KeyError(
                f"Status code {code!r} is not registered. "
                "Use StatusCode.register() or an entry point to add it."
            )
        return c


# ── StatusCode ────────────────────────────────────────────────────────────────

class StatusCode(int, metaclass=_StatusCodeMeta):
    """Status code value — ``int`` subclass; each registered code is a singleton.

    Use named constants or ``StatusCode[code]`` for lookup.  Values compare
    equal to plain ints and pass ``isinstance(..., int)`` checks.
    """

    _registry: ClassVar[dict[int, StatusCode]] = {}
    _ep_loaded: ClassVar[bool] = False
    _ep_lock: ClassVar[Lock] = Lock()

    _desc: str  # set in __new__

    # ── Construction (singleton per int value) ────────────────────────────────

    def __new__(cls, value: int, desc: str = "") -> StatusCode:
        iv = int(value)
        existing = cls._registry.get(iv)
        if existing is not None:
            return existing
        obj = super().__new__(cls, iv)
        obj._desc = str(desc)
        cls._registry[iv] = obj
        return obj

    def __repr__(self) -> str:
        return f"StatusCode({int(self)}, {self._desc!r})"

    @property
    def desc(self) -> str:
        """Default human-readable description registered with this code."""
        return self._desc

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def register(cls, code: int, desc: str = "") -> StatusCode:
        """Register (or update) a status code in-process.

        Returns the ``StatusCode`` singleton for *code*.  If *code* is already
        registered, its description is updated to *desc*.
        """
        iv = int(code)
        existing = cls._registry.get(iv)
        if existing is not None:
            existing._desc = str(desc)
            return existing
        return cls(iv, str(desc))

    # ── Internal ──────────────────────────────────────────────────────────────

    @classmethod
    def _ensure_loaded(cls) -> None:
        if not cls._ep_loaded:
            cls._load_entry_points()

    @classmethod
    def _load_entry_points(cls) -> None:
        with cls._ep_lock:
            if cls._ep_loaded:
                return
            for ep in entry_points(group="ai_navigator.status_codes"):
                try:
                    obj = ep.load()
                    codes: dict = obj() if callable(obj) else obj
                    for c, d in codes.items():
                        cls.register(int(c), str(d))
                    _log.debug(
                        "loaded %d code(s) from entry point '%s'", len(codes), ep.name
                    )
                except Exception as exc:
                    _log.warning(
                        "status code entry point '%s' failed to load: %s", ep.name, exc
                    )
            cls._ep_loaded = True


# ── Built-in constants ────────────────────────────────────────────────────────
# Defined after the class so StatusCode(...) can reference the fully-defined class.
# Each assignment creates a singleton and registers it in _registry.

# 2xx — Success
StatusCode.OK                  = StatusCode(200, "Ok")

# 4xx — Client / caller errors
StatusCode.UNAUTHORIZED        = StatusCode(401, "Unauthorized")
StatusCode.FORBIDDEN           = StatusCode(403, "Forbidden")
StatusCode.TOO_MANY_REQUESTS   = StatusCode(429, "Too Many Requests")

# 5xx — Server / infrastructure errors
StatusCode.INTERNAL_ERROR      = StatusCode(500, "Internal Server Error")
StatusCode.BAD_GATEWAY         = StatusCode(502, "Bad Gateway")
StatusCode.SERVICE_UNAVAILABLE = StatusCode(503, "Service Unavailable")

# 6xx — LLM-specific
StatusCode.CONTEXT_LIMIT       = StatusCode(601, "Context Limit Exceeded")
StatusCode.CONTENT_FILTERED    = StatusCode(602, "Content Filtered")
StatusCode.OUTPUT_TRUNCATED    = StatusCode(603, "Output Truncated")
StatusCode.SCHEMA_MISMATCH     = StatusCode(604, "Schema Mismatch")
StatusCode.EMPTY_RESPONSE      = StatusCode(605, "Empty Response")
StatusCode.PROVIDER_TIMEOUT    = StatusCode(606, "Provider Timeout")


def describe(code: int) -> str:
    """Return the default description for *code*, or ``"Unknown"`` if not registered.

    ``status_desc`` in :class:`StatusDetail` is independent — callers may
    supply any label they choose.  This is a convenience utility for logging
    and default fills.
    """
    StatusCode._ensure_loaded()
    c = StatusCode._registry.get(int(code))
    return c._desc if c is not None else "Unknown"


# ── StatusDetail ──────────────────────────────────────────────────────────────

class StatusDetail(TypedDict):
    """Status block carried by every :class:`~ai_navigator.state.data_class.NavigatorResult`.

    ``status_code``   — :class:`StatusCode` value (HTTP-style int, e.g.
                        ``StatusCode.OK``, ``StatusCode[429]``).
    ``status_desc``   — short human-readable label; callers may supply their
                        own (e.g. ``"Gemini quota exceeded"``), independent of
                        :func:`describe`.
    ``status_detail`` — full error message; empty string on success.
    """
    status_code: StatusCode
    status_desc: str
    status_detail: str
