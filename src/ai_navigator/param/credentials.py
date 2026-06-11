"""Credentials loading with Entry Points replacement support.

Default behaviour reads from a YAML file.  Third-party packages can replace
this entirely by registering their class under ``ai_navigator.credentials``::

    # their pyproject.toml
    [project.entry-points."ai_navigator.credentials"]
    vault = "my_package.credentials:VaultLoader"

The registered class must implement ``fetch() -> dict[str, Any]``.  Only ONE
replacement is active — if multiple plugins are installed the first is used and
a warning is emitted.

Alternatively, pass your loader directly at Navigator construction time (takes
priority over entry-point discovery)::

    nav = Navigator(credentials_loader=VaultLoader())
"""
from __future__ import annotations
import logging
import yaml
from importlib.metadata import entry_points
from typing import Any

_log = logging.getLogger("ai_navigator.param.credentials")


class CredentialsLoader:
    """Default credentials loader — reads a YAML file.

    File path comes from :attr:`~ai_navigator.param.const_configs.ConstConfigs.CREDENTIALS_PATH`.

    Expected structure::

        my_claude:
          - provider_type: anthropic
            model: claude-sonnet-4-6
            api_key: sk-ant-...
            max_tokens: 4096

        my_gpt4:
          - provider_type: openai
            model: gpt-4o
            api_key: sk-openai-...

    Subclass and override :meth:`fetch` to load from Vault, AWS Secrets
    Manager, a database, or any other source.
    """

    def fetch(self) -> dict[str, Any]:
        from ai_navigator.param.const_configs import ConstConfigs
        path = ConstConfigs.CREDENTIALS_PATH
        try:
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            _log.debug("credentials: file not found at %s", path)
            return {}
        except PermissionError as exc:
            _log.warning("credentials: permission denied reading %s (%s)", path, exc)
            return {}
        except Exception as exc:
            _log.warning("credentials: fetch failed (%s)", exc)
            return {}


# ── Entry Points discovery ────────────────────────────────────────────────────

_credentials_class_cache: type | None = None


def get_credentials_class() -> type:
    """Return the active credentials loader class.

    Checks ``ai_navigator.credentials`` entry points first; if none are
    installed, returns the built-in :class:`CredentialsLoader`.
    Only the first registered plugin is used.
    """
    global _credentials_class_cache
    if _credentials_class_cache is not None:
        return _credentials_class_cache

    eps = list(entry_points(group="ai_navigator.credentials"))
    if not eps:
        _credentials_class_cache = CredentialsLoader
        return CredentialsLoader

    if len(eps) > 1:
        _log.warning(
            "multiple credentials plugins found (%s), using first: %s",
            [e.name for e in eps],
            eps[0].name,
        )

    try:
        cls = eps[0].load()
        _log.info("using credentials plugin: %s", eps[0].name)
        _credentials_class_cache = cls
        return cls
    except Exception as exc:
        _log.warning("credentials plugin '%s' failed to load: %s — using default", eps[0].name, exc)
        _credentials_class_cache = CredentialsLoader
        return CredentialsLoader
