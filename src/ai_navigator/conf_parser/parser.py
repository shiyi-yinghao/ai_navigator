"""Configuration file parser — extends SchemaBuilder with multi-schema support.

Placeholder for future implementation. Planned features:
- Load a single YAML file containing multiple named schemas
- Template / inheritance between schemas
- Environment variable substitution (${MY_VAR})
- Per-provider default parameter overrides
"""
from __future__ import annotations
from typing import Any

import yaml

from ai_navigator.schema.composer import SchemaComposer as SchemaBuilder  # SchemaBuilder is not yet implemented


class ConfParser:
    """Parse configuration files containing multiple schemas and settings.

    Expected YAML layout::

        schemas:
          ProductReview:
            name: ProductReview
            description: ...
            fields: [...]
          SentimentResult:
            name: SentimentResult
            ...

        defaults:
          openai:
            temperature: 0.2
            max_tokens: 1024
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ConfParser":
        config = yaml.safe_load(yaml_str)
        return cls(config or {})

    @classmethod
    def from_yaml_file(cls, path: str) -> "ConfParser":
        with open(path, encoding="utf-8") as fh:
            return cls.from_yaml(fh.read())

    def get_schema(self, name: str) -> SchemaBuilder:
        schemas = self._config.get("schemas", {})
        if name not in schemas:
            raise KeyError(f"Schema '{name}' not found in config. Available: {list(schemas)}")
        return SchemaBuilder(schemas[name])

    def list_schemas(self) -> list[str]:
        return list(self._config.get("schemas", {}).keys())

    def get_defaults(self, provider: str | None = None) -> dict[str, Any]:
        defaults: dict[str, Any] = self._config.get("defaults", {})
        if provider:
            return defaults.get(provider, {})
        return defaults
