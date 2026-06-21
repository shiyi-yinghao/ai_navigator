from __future__ import annotations

import logging
import sys
from threading import Lock

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_default_checked = False
_check_lock = Lock()


def get_logger(name: str) -> logging.Logger:
    """Return the logger at ``ai_navigator.<name>``.

    On the first call, installs a default stderr handler on the
    ``ai_navigator`` root **only if** neither the root logger nor
    ``ai_navigator`` has any real (non-:class:`~logging.NullHandler`) handlers
    configured.  This lets callers silence or redirect logs by setting up their
    own :mod:`logging` configuration before the first Navigator or Server is
    instantiated.
    """
    _ensure_default_handler()
    return logging.getLogger(f"ai_navigator.{name}")


def _ensure_default_handler() -> None:
    global _default_checked
    if _default_checked:
        return
    with _check_lock:
        if _default_checked:
            return
        _install_if_unconfigured()
        _default_checked = True


def _install_if_unconfigured() -> None:
    ai_nav = logging.getLogger("ai_navigator")
    all_handlers = logging.root.handlers + ai_nav.handlers
    has_real = any(not isinstance(h, logging.NullHandler) for h in all_handlers)
    if not has_real:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_FORMATTER)
        ai_nav.addHandler(handler)
        ai_nav.setLevel(logging.INFO)
