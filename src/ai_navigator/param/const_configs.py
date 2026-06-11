"""Package-wide configuration constants with Entry Points extension support.

Core constants are read from environment variables at import time.

Extending via Entry Points
--------------------------
Third-party packages can contribute extra config parameters by registering a
callable under the ``ai_navigator.configs`` group::

    # their pyproject.toml
    [project.entry-points."ai_navigator.configs"]
    my_plugin = "my_package.config:get_extra_configs"

The callable must return a ``dict[str, Any]``.  Extension values are merged on
top of the base constants and accessible via :meth:`ConstConfigs.get` and
:meth:`ConstConfigs.all`.  Base constants (STORAGE_PATH, CREDENTIALS_PATH,
LOGGING_STREAM) always take precedence over extension values.

Override a constant programmatically (e.g. in tests)::

    from ai_navigator.param.const_configs import ConstConfigs
    ConstConfigs.STORAGE_PATH = "/tmp/test.db"
"""
import logging
import os
from importlib.metadata import entry_points
from typing import Any

_log = logging.getLogger("ai_navigator.param.configs")


class ConstConfigs:
    STORAGE_PATH: str = os.environ.get("AI_NAVIGATOR_STORAGE_PATH", "ai_navigator.db")
    CREDENTIALS_PATH: str = os.environ.get("AI_NAVIGATOR_CREDENTIALS_PATH", "credentials.yaml")
    LOGGING_STREAM: bool = os.environ.get("AI_NAVIGATOR_LOGGING_STREAM", "true").lower() != "false"

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Return a config value by name, including extension params."""
        if hasattr(cls, key):
            return getattr(cls, key)
        return _config_extensions().get(key, default)

    @classmethod
    def all(cls) -> dict[str, Any]:
        """Return all config params — base constants take precedence over extensions."""
        base = {
            k: v for k, v in vars(cls).items()
            if not k.startswith("_")
            and not callable(v)
            and not isinstance(v, (classmethod, staticmethod))
        }
        return {**_config_extensions(), **base}


# ── Entry Points discovery ────────────────────────────────────────────────────

_extensions_cache: dict[str, Any] | None = None


def _config_extensions() -> dict[str, Any]:
    """Load (and cache) config extensions from installed plugins."""
    global _extensions_cache
    if _extensions_cache is not None:
        return _extensions_cache

    result: dict[str, Any] = {}
    for ep in entry_points(group="ai_navigator.configs"):
        try:
            contrib = ep.load()
            if callable(contrib):
                contrib = contrib()
            if isinstance(contrib, dict):
                result.update(contrib)
            else:
                _log.warning("config plugin '%s' must return dict, got %s", ep.name, type(contrib))
        except Exception as exc:
            _log.warning("config plugin '%s' failed: %s", ep.name, exc)

    _extensions_cache = result
    return result
