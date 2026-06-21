"""Status code registry for :class:`~ai_navigator.infra.types.CallStatus`.

Built-in codes follow HTTP conventions (2xx/4xx/5xx).  Custom codes in the
6xx range cover LLM-specific situations.  Third-party packages can register
additional codes via the ``ai_navigator.status_codes`` entry-point group.

Referencing codes
-----------------
Named attributes for built-in codes::

    SC.OK                  # → 200
    SC.TOO_MANY_REQUESTS   # → 429
    SC.CONTEXT_LIMIT       # → 601

Integer lookup (validates the code is registered, returns the int)::

    SC[200]   # → 200
    SC[429]   # → 429
    SC[709]   # → 709  (if a plugin registered 709)

Both forms return a plain :class:`int`, usable directly in ``NavigatorResult``::

    NavigatorResult(
        result="",
        status={
            "status_code": SC[429],
            "status_desc":  "Gemini quota exceeded",   # caller's own desc
            "status_detail": str(exc),
        },
        usage={},
        reference={},
    )

``status_desc`` is **independent** of :func:`describe`.  Callers may supply any
description they like; ``describe(code)`` is a default-fill utility, not an
enforced label.

Registering custom codes via entry points
-----------------------------------------
Expose a ``dict[int, str]`` (or a zero-arg callable returning one) under the
group ``ai_navigator.status_codes``::

    # pyproject.toml
    [project.entry-points."ai_navigator.status_codes"]
    my_plugin = "my_plugin.status:CODES"

    # my_plugin/status.py
    CODES = {
        709: "Custom Timeout",
        710: "Billing Limit Exceeded",
    }

Named attribute aliases are **not** auto-created for plugin codes — reference
them as ``SC[709]``.  Use :meth:`SC.register` for in-process registration.
"""
from __future__ import annotations

import logging
from importlib.metadata import entry_points
from threading import Lock

_log = logging.getLogger("ai_navigator.status_codes")


class _SCMeta(type):
    """Metaclass that adds ``SC[code]`` integer lookup to :class:`SC`."""

    def __getitem__(cls, code: int) -> int:
        """Return *code* if it is registered; raise :class:`KeyError` otherwise."""
        cls._ensure_loaded()
        if code not in cls._registry:
            raise KeyError(
                f"Status code {code!r} is not registered. "
                "Use SC.register() or an entry point to add it."
            )
        return code


class SC(metaclass=_SCMeta):
    """Status code constants and extensible registry.

    All named attributes return plain :class:`int` values.
    ``SC[code]`` validates and returns the int for any registered code —
    built-in or plugin-supplied.
    """

    # ── 2xx — Success ─────────────────────────────────────────────────────────
    OK                  = 200

    # ── 4xx — Client / caller errors ─────────────────────────────────────────
    UNAUTHORIZED        = 401   # Invalid or missing API key.
    FORBIDDEN           = 403   # Key valid but lacks permission for this operation.
    TOO_MANY_REQUESTS   = 429   # Rate limit hit (provider or local limiter).

    # ── 5xx — Server / infrastructure errors ─────────────────────────────────
    INTERNAL_ERROR      = 500   # Unexpected error in ai-navigator itself.
    BAD_GATEWAY         = 502   # Provider returned an unexpected / malformed response.
    SERVICE_UNAVAILABLE = 503   # Provider temporarily unreachable.

    # ── 6xx — LLM-specific (custom) ──────────────────────────────────────────
    CONTEXT_LIMIT       = 601   # Prompt exceeds the model's maximum context length.
    CONTENT_FILTERED    = 602   # Response blocked by provider's content-safety policy.
    OUTPUT_TRUNCATED    = 603   # Model stopped early (finish_reason = "length").
    SCHEMA_MISMATCH     = 604   # Structured-output response does not match the schema.
    EMPTY_RESPONSE      = 605   # Model returned an empty content string.
    PROVIDER_TIMEOUT    = 606   # Provider did not respond within the timeout window.

    # ── Registry ──────────────────────────────────────────────────────────────
    _registry: dict[int, str] = {}   # code → default description
    _ep_loaded: bool = False
    _ep_lock: Lock = Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def register(cls, code: int, default_desc: str = "") -> None:
        """Register a custom status code in-process.

        Parameters
        ----------
        code:
            Integer status code to register.
        default_desc:
            Short human-readable label returned by :func:`describe` when the
            caller does not provide their own ``status_desc``.
        """
        cls._registry[int(code)] = str(default_desc)

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
                    codes: dict[int, str] = obj() if callable(obj) else obj
                    for code, desc in codes.items():
                        cls._registry[int(code)] = str(desc)
                    _log.debug("loaded %d code(s) from entry point '%s'", len(codes), ep.name)
                except Exception as exc:
                    _log.warning(
                        "status code entry point '%s' failed to load: %s", ep.name, exc
                    )
            cls._ep_loaded = True


# Register built-in codes so SC[200], SC[429], etc. work at import time.
SC._registry.update({
    SC.OK:                  "Ok",
    SC.UNAUTHORIZED:        "Unauthorized",
    SC.FORBIDDEN:           "Forbidden",
    SC.TOO_MANY_REQUESTS:   "Too Many Requests",
    SC.INTERNAL_ERROR:      "Internal Server Error",
    SC.BAD_GATEWAY:         "Bad Gateway",
    SC.SERVICE_UNAVAILABLE: "Service Unavailable",
    SC.CONTEXT_LIMIT:       "Context Limit Exceeded",
    SC.CONTENT_FILTERED:    "Content Filtered",
    SC.OUTPUT_TRUNCATED:    "Output Truncated",
    SC.SCHEMA_MISMATCH:     "Schema Mismatch",
    SC.EMPTY_RESPONSE:      "Empty Response",
    SC.PROVIDER_TIMEOUT:    "Provider Timeout",
})


def describe(code: int) -> str:
    """Return the default description for *code*, or ``"Unknown"`` if not registered.

    This is a convenience utility for logging and default fills.
    ``status_desc`` in :class:`~ai_navigator.infra.types.CallStatus` is
    independent — callers may supply any description they choose.
    """
    SC._ensure_loaded()
    return SC._registry.get(code, "Unknown")
