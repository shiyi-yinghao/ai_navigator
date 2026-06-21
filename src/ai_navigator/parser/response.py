from __future__ import annotations
import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ai_navigator.state.data_class import Response
from ai_navigator.monitor.logger import get_logger

T = TypeVar("T", bound=BaseModel)

_logger = get_logger("parser")

# Matches JSON inside ```json ... ``` or ``` ... ``` fences
_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```",
    re.IGNORECASE,
)
# Fallback: first {...} or [...] block in plain text
_BARE_RE = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])")


class ResponseParser:
    """Parse and validate LLM response text into structured Python objects.

    Handles:
    - JSON embedded in markdown code fences
    - JSON in plain prose
    - Pydantic model validation with error reporting
    - Recursive key search in nested dicts
    - Enum candidate validation

    For schema-aware leaf-node extraction use
    :class:`~ai_navigator.schema.extractor.ResultExtractor`.
    """

    # ── Core extraction ──────────────────────────────────────────────────────

    def extract_json_str(self, text: str) -> str:
        """Extract the first JSON object/array from text, stripping prose."""
        m = _FENCE_RE.search(text)
        if m:
            return m.group(1).strip()
        m = _BARE_RE.search(text)
        if m:
            return m.group(1).strip()
        raise ValueError("No JSON found in LLM response")

    def parse_json(self, text: str) -> Any:
        """Extract and deserialise JSON from LLM response text."""
        json_str = self.extract_json_str(text)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            _logger.error(
                "JSON decode failed: %s | snippet=%s", exc, json_str[:200]
            )
            raise ValueError(f"Invalid JSON: {exc}") from exc

    # ── Pydantic validation ──────────────────────────────────────────────────

    def parse_pydantic(self, text: str, model: Type[T]) -> T:
        """Parse JSON from ``text`` and validate it against ``model``."""
        data = self.parse_json(text)
        try:
            return (
                model(**data)
                if isinstance(data, dict)
                else model.model_validate(data)
            )
        except ValidationError as exc:
            _logger.error(
                "Pydantic validation failed for %s: %s", model.__name__, exc
            )
            raise ValueError(
                f"Validation failed for {model.__name__}: {exc}"
            ) from exc

    # ── Convenience wrappers ─────────────────────────────────────────────────

    def parse_response(self, response: Response) -> Any:
        return self.parse_json(response.get("content", ""))

    def parse_response_pydantic(self, response: Response, model: Type[T]) -> T:
        return self.parse_pydantic(response.get("content", ""), model)

    # ── Soft / non-raising variants ──────────────────────────────────────────

    def try_parse_json(self, text: str, default: Any = None) -> Any:
        """Like ``parse_json`` but returns ``default`` instead of raising."""
        try:
            return self.parse_json(text)
        except ValueError as exc:
            _logger.warning("soft parse failure: %s", exc)
            return default

    def try_parse_pydantic(
        self, text: str, model: Type[T], default: T | None = None
    ) -> T | None:
        """Like ``parse_pydantic`` but returns ``default`` instead of raising."""
        try:
            return self.parse_pydantic(text, model)
        except ValueError as exc:
            _logger.warning("soft pydantic parse failure: %s", exc)
            return default

    # ── Nested search ────────────────────────────────────────────────────────

    def find_value(self, data: Any, key: str) -> Any:
        """Recursively search for the first occurrence of *key* in nested dicts/lists.

        Returns the associated value, or ``None`` if not found.
        """
        if isinstance(data, dict):
            if key in data:
                return data[key]
            for v in data.values():
                result = self.find_value(v, key)
                if result is not None:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self.find_value(item, key)
                if result is not None:
                    return result
        return None

    # ── Enum validation ──────────────────────────────────────────────────────

    def validate_enum(
        self,
        value: str,
        allowed: list[str],
        case_sensitive: bool = True,
    ) -> str:
        """Validate that *value* is one of the *allowed* enum candidates."""
        candidates = allowed if case_sensitive else [c.lower() for c in allowed]
        check = value if case_sensitive else value.lower()
        if check not in candidates:
            raise ValueError(
                f"Value '{value}' is not a valid enum candidate. "
                f"Expected one of: {allowed}"
            )
        if not case_sensitive:
            idx = candidates.index(check)
            return allowed[idx]
        return value

    def validate_enum_dict(
        self,
        data: dict[str, Any],
        enum_map: dict[str, list[str]],
        case_sensitive: bool = True,
    ) -> dict[str, Any]:
        """Validate multiple enum fields in a parsed response dict."""
        result = dict(data)
        for field, allowed in enum_map.items():
            if field not in result:
                _logger.warning("enum field '%s' not found in response", field)
                continue
            result[field] = self.validate_enum(
                str(result[field]), allowed, case_sensitive=case_sensitive
            )
        return result
