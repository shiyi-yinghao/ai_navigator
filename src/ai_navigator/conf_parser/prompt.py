"""YAML-driven conversation prompt builder.

Assembles a ``list[Message]`` from a declarative YAML template, resolving
``dynamic_*`` content parts at call time from a ``data_dict``.

YAML format
-----------
The template is a list of message blocks.  Each block has an optional
``role`` (defaults to ``"user"``) and a ``message`` list of content parts::

    - role: system
      message:
        - type: const_text
          content: You are a helpful assistant.

    - message:   # role defaults to "user"
        - type: const_text
          content: "Describe the following product:"
        - type: dynamic_text
          key: product_description
        - type: const_image_url
          content: "https://example.com/product.jpg"

    - role: assistant
      message:
        - type: const_text
          content: "I will analyze the product."

Content-part ``type`` values
-----------------------------
``const_text``          Keep literal ``content`` string as text.
``dynamic_text``        Replace with ``data_dict[key]`` (coerced to str).
``const_image_url``     Literal URL image reference.
``dynamic_image_url``   Replace with ``data_dict[key]`` URL string.
``const_image_base64``  Literal base64 image (bytes or str).
``dynamic_image_base64``Replace with ``data_dict[key]`` base64 bytes/str.

If a message has exactly one text part it is simplified to a plain string
``content``; otherwise ``content`` is a ``list[ContentPart]``.
"""
from __future__ import annotations
from typing import Any

import yaml

from ai_navigator.infra.exceptions import AINavigatorError
from ai_navigator.infra.models import ContentPart, Message


class PromptError(AINavigatorError):
    pass


class PromptBuilder:
    """Build a ``list[Message]`` from a YAML prompt template.

    Usage
    -----
    ::

        pb = PromptBuilder.from_yaml_file("prompt.yaml")
        messages = pb.build(data_dict={"product_description": "Fast laptop"})
    """

    def __init__(self, template: list[dict[str, Any]]) -> None:
        if not isinstance(template, list):
            raise PromptError("Prompt template must be a YAML list of message blocks")
        self._template = template

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "PromptBuilder":
        try:
            parsed = yaml.safe_load(yaml_str)
        except yaml.YAMLError as exc:
            raise PromptError(f"Invalid prompt YAML: {exc}") from exc
        return cls(parsed or [])

    @classmethod
    def from_yaml_file(cls, path: str) -> "PromptBuilder":
        with open(path, encoding="utf-8") as fh:
            return cls.from_yaml(fh.read())

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, data_dict: dict[str, Any] | None = None) -> list[Message]:
        """Materialise the template into a list of Messages.

        Parameters
        ----------
        data_dict:
            Values for ``dynamic_*`` content parts.  Missing keys produce an
            empty string rather than raising, so callers can detect omissions.
        """
        dd = data_dict or {}
        messages: list[Message] = []
        for idx, block in enumerate(self._template):
            if not isinstance(block, dict):
                raise PromptError(f"Template block {idx} is not a mapping")
            role: str = block.get("role", "user")
            parts_spec: list[dict[str, Any]] = block.get("message", [])
            parts = self._build_parts(parts_spec, dd, block_idx=idx)
            if len(parts) == 1 and parts[0].type == "text":
                msg = Message(role=role, content=parts[0].text or "")  # type: ignore[arg-type]
            else:
                msg = Message(role=role, content=parts)  # type: ignore[arg-type]
            messages.append(msg)
        return messages

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_parts(
        self,
        specs: list[dict[str, Any]],
        data_dict: dict[str, Any],
        block_idx: int,
    ) -> list[ContentPart]:
        parts: list[ContentPart] = []
        for i, item in enumerate(specs):
            if not isinstance(item, dict):
                raise PromptError(
                    f"Content part {i} in block {block_idx} is not a mapping"
                )
            part_type: str = item.get("type", "")
            if part_type.startswith("const_"):
                content_kind = part_type[len("const_"):]
                raw = item.get("content", "")
            elif part_type.startswith("dynamic_"):
                content_kind = part_type[len("dynamic_"):]
                key = item.get("key", "")
                raw = data_dict.get(key, "")
            else:
                raise PromptError(
                    f"Unknown content-part type '{part_type}' in block {block_idx}, "
                    "part {i}. Must start with 'const_' or 'dynamic_'."
                )
            parts.append(_make_content_part(content_kind, raw))
        return parts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_content_part(kind: str, raw: Any) -> ContentPart:
    """Convert a content kind + raw value to a ``ContentPart``."""
    if kind == "text":
        return ContentPart(type="text", text=str(raw))
    if kind == "image_url":
        return ContentPart(type="image_url", image_url=str(raw))
    if kind == "image_base64":
        data = raw if isinstance(raw, str) else (raw.decode() if isinstance(raw, bytes) else str(raw))
        return ContentPart(type="image_base64", image_data=data)
    # Unknown kind → fall back to plain text
    return ContentPart(type="text", text=str(raw))
