"""Provider registry — discovers built-in and plugin server classes.

Server imports are lazy (inside the function body) to avoid circular imports:
the individual server files import TypedDicts from infra.base_navigator, so
this module must not import them at the top level.

Entry Points (supplement)
-------------------------
Third-party packages can add new providers by registering a
:class:`~ai_navigator.server.base_server.BaseServer` subclass under
``ai_navigator.servers``::

    # their pyproject.toml
    [project.entry-points."ai_navigator.servers"]
    cohere = "my_package.server:CohereServer"

All installed plugins are added to the registry alongside the built-ins.
Plugin providers can shadow a built-in name (e.g. to override behaviour), but
the ``extra`` argument passed at construction time always takes highest
priority.
"""
from __future__ import annotations
from importlib.metadata import entry_points
from typing import Any

from ai_navigator.monitor.logger import get_logger

_log = get_logger("server.registry")


def build_registry(extra: list | None = None) -> dict[str, Any]:
    """Return a ``{provider_name: ServerClass}`` registry.

    Resolution order (later entries win):

    1. Built-in servers — anthropic, openai, gemini
    2. Installed plugins via ``ai_navigator.servers`` entry point group
    3. *extra* list passed directly (highest priority)
    """
    from ai_navigator.server.anthropic_server import AnthropicServer
    from ai_navigator.server.openai_server import OpenAIServer
    from ai_navigator.server.gemini_server import GeminiServer

    registry: dict[str, Any] = {
        cls.provider: cls for cls in [AnthropicServer, OpenAIServer, GeminiServer]
    }

    for ep in entry_points(group="ai_navigator.servers"):
        try:
            cls = ep.load()
            registry[cls.provider] = cls
            _log.info("loaded server plugin: %s (provider=%s)", ep.name, cls.provider)
        except Exception as exc:
            _log.warning("server plugin '%s' failed to load: %s", ep.name, exc)

    for cls in (extra or []):
        registry[cls.provider] = cls

    return registry
