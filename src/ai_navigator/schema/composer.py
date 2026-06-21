"""Schema composer — converts YAML term definitions to LLM-ready request schemas.

Responsible for the "request side" of the schema pipeline:
  YAML spec  →  preprocess(data_dict)  →  schema_conversion()  →  response_format dict

YAML layout
-----------
::

    meta:
      name: ProductReview
      description: Extract structured review data
      version: "1.0"

    defs:                              # optional — reusable term definitions
      score_range:
        type: int
        description: Score from 0 to 10
      address:
        type: dict
        terms:
          street:
            type: str
          city:
            type: str

    schema:
      title:
        type: str
        description: Product title

      rating:
        ref: score_range               # → {"$ref": "#/$defs/score_range"}

      category:
        type: enum
        choices: [electronics, clothing, food]
        config_confidence: true        # pkg-internal: include in LogProb extraction

      category_dyn:
        type: enum
        dynamic_choices: cat_list      # choices from data_dict["cat_list"]
        confidence: true

      shipping_addr:
        ref: address

      optional_note:
        type: [str, null]              # → {"anyOf": [{"type": "string"}, {"type": "null"}]}
        description: Optional note

      detail:
        type: dict
        terms:
          reason:
            type: str
          score:
            ref: score_range

      tags:
        type: list
        item_type: str                 # str / int / float / bool  (default: str)
        choices: [fast, light]         # optional — constrains list items

Schema format
-------------
``schema:`` (and nested ``terms:``) is a **dict** whose keys are the term names
and values are the term specs.  There is no ``name:`` field inside the spec.

Supported ``type`` values
-------------------------
Scalar: ``str`` / ``string`` / ``free-text``, ``int`` / ``integer``,
``float`` / ``number``, ``bool`` / ``boolean``, ``null``.
Compound: ``enum``, ``list``, ``dict``, ``any``.
List of the above (e.g. ``[str, null]``) → ``anyOf`` in JSON Schema.

Dynamic attributes (resolved by preprocess, then removed)
---------------------------------------------------------
Any attribute can be made dynamic by prefixing it with ``dynamic_``.
``dynamic_{attr}: key`` → sets ``attr = data_dict[key]`` before any other
processing.  Common examples:

``dynamic_description: key``   → sets ``description``
``dynamic_choices: key``       → sets ``choices``
``dynamic_type: key``          → sets ``type`` (resolved before type dispatch!)
``dynamic_terms: key``         → sets ``terms`` for a ``dict`` term

``config_*`` attributes (pkg-internal convention)
    Any attribute whose name starts with ``config_`` is a package-internal
    directive.  These attributes are **never forwarded** to the JSON Schema
    output — ``schema_conversion()`` silently ignores them.

``config_confidence``
    Optional boolean (default ``false``). When ``true`` the term is included
    in :meth:`confidence_terms`, enabling LogProb probability extraction.
    Meaningful only on ``enum`` terms.

Field-name constraint
---------------------
Term names (and def names) must not contain ``"."``.  A :class:`SchemaError`
is raised at conversion time.
"""
from __future__ import annotations
import copy
import re
from typing import Any

import yaml


# ── JSON Schema type atoms ────────────────────────────────────────────────────

_SCALAR_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "string": "string",
    "free-text": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "null": "null",
}

_ITEM_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
}


class SchemaComposer:
    """Compose an OpenAI structured-output schema from a YAML term specification.

    Workflow
    --------
    1. ``preprocess(data_dict)``  — resolve ``dynamic_*`` attributes; returns a
                                    new :class:`SchemaComposer`.
    2. ``schema_conversion()``   — return the full OpenAI ``response_format``
                                    dict, ready for ``llm.response()``.

    Introspection helpers
    ---------------------
    ``leaf_paths()``        → dot-notation paths of all leaf terms.
    ``confidence_terms()``  → ``{path: [candidates]}`` for
                              ``config_confidence: true`` enum terms
                              (fed to :class:`LogProbParser`).
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self._spec = spec
        _meta: dict[str, Any] = spec.get("meta", {})
        self._name: str = _meta.get("name", "GeneratedModel")
        self._description: str = _meta.get("description", "")
        self._version: str = _meta.get("version", "")
        self._defs: dict[str, Any] = spec.get("defs", {})
        self._terms: dict[str, Any] = spec.get("schema", {})

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SchemaComposer":
        try:
            spec = yaml.safe_load(yaml_str)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML: {exc}") from exc
        if not isinstance(spec, dict):
            raise ValueError("YAML root must be a mapping")
        if "meta" not in spec or "schema" not in spec:
            raise ValueError("YAML must have top-level 'meta' and 'schema' keys")
        if not isinstance(spec["schema"], dict):
            raise ValueError(
                "'schema' must be a dict (term-name → spec). "
                "Use 'term_name:\\n  type: str' not '- name: term_name\\n  type: str'."
            )
        if "defs" in spec and not isinstance(spec["defs"], dict):
            raise ValueError("'defs' must be a dict (def-name → spec)")
        return cls(spec)

    @classmethod
    def from_yaml_file(cls, path: str) -> "SchemaComposer":
        with open(path, encoding="utf-8") as fh:
            return cls.from_yaml(fh.read())

    # ── Phase 1: preprocess ───────────────────────────────────────────────────

    def preprocess(self, data_dict: dict[str, Any] | None = None) -> "SchemaComposer":
        """Return a new SchemaComposer with all ``dynamic_*`` attributes resolved.

        Any ``dynamic_{attr}: key`` entry is resolved to
        ``term[attr] = data_dict[key]`` and then removed.  This applies to all
        attributes, including ``type`` itself.  Resolution is always performed
        before any other logic, so the resulting spec contains only static keys.

        The original spec is never mutated.
        """
        if not data_dict:
            return self
        new_spec = copy.deepcopy(self._spec)
        _resolve_terms(new_spec.get("schema", {}), data_dict)
        _resolve_terms(new_spec.get("defs", {}), data_dict)
        return SchemaComposer(new_spec)

    # ── Phase 2: conversion ───────────────────────────────────────────────────

    def schema_conversion(self, task_name: str | None = None) -> dict[str, Any]:
        """Return an OpenAI ``response_format`` structured-output dict.

        Includes ``$defs`` in the schema when a ``defs:`` section is present.

        Returns
        -------
        dict
            ``{"type": "json_schema", "json_schema": {"name": ..., "strict": True,
            "schema": {...}}}``
        """
        properties: dict[str, Any] = {}
        required_keys: list[str] = []
        for name, term in self._terms.items():
            _validate_term_name(name)
            properties[name] = _term_to_json_schema(name, term)
            required_keys.append(name)

        inner_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required_keys,
            "additionalProperties": False,
        }

        if self._defs:
            inner_schema["$defs"] = {
                def_name: _term_to_json_schema(def_name, def_spec)
                for def_name, def_spec in self._defs.items()
            }

        name_slug = re.sub(r"\W", "_", task_name or self._name)[:64]
        json_schema: dict[str, Any] = {
            "name": name_slug,
            "strict": True,
            "schema": inner_schema,
        }
        if self._description:
            json_schema["description"] = self._description
        return {"type": "json_schema", "json_schema": json_schema}

    # ── Introspection ─────────────────────────────────────────────────────────

    def confidence_terms(self) -> dict[str, list[str]]:
        """Return ``{path: [candidates]}`` for all ``config_confidence: true`` enum terms.

        Only ``enum`` terms with a non-empty ``choices`` list are included.
        Used to feed :class:`~ai_navigator.parser.logprob.LogProbParser`.
        """
        out: dict[str, list[str]] = {}
        _collect_confidence(self._terms, "", out)
        return out

    # ── Prompt helper ─────────────────────────────────────────────────────────

    def build_prompt_instruction(self) -> str:
        """Generate a plain-text system-prompt fragment describing the schema."""
        lines = [
            f"Respond ONLY with valid JSON matching schema: {self._name}",
            f"Description: {self._description}",
            "",
            "Terms:",
        ]
        _append_prompt_terms(lines, self._terms, indent=2)
        lines.append("")
        lines.append("Do not include any explanation outside the JSON object.")
        return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate_term_name(name: str) -> None:
    if "." in name:
        raise ValueError(
            f"Term name '{name}' contains a dot. "
            "Dots are reserved for nested-path output notation."
        )


def _resolve_terms(
    terms: dict[str, Any], data_dict: dict[str, Any]
) -> None:
    """Mutate *terms* in-place, resolving all ``dynamic_*`` attributes.

    For every key matching ``dynamic_{attr}`` in a term spec:
    - Strip the prefix to obtain the target attribute name (``attr``).
    - Look up the value in *data_dict* using the key stored in the dynamic field.
    - Set ``term[attr] = data_dict[value]`` and remove the ``dynamic_*`` key.

    All ``dynamic_*`` keys are processed before any other logic runs, so even
    ``dynamic_type`` works correctly — by the time ``schema_conversion()`` reads
    ``type``, it is already the resolved static value.
    """
    for _name, term in terms.items():
        if not isinstance(term, dict):
            continue

        # Collect all dynamic_* keys first (avoid mutating while iterating)
        dynamic_keys = [k for k in list(term) if k.startswith("dynamic_")]
        for dyn_key in dynamic_keys:
            attr = dyn_key[len("dynamic_"):]   # e.g. "type", "description", "choices"
            lookup = term.pop(dyn_key)
            if lookup in data_dict:
                term[attr] = data_dict[lookup]

        sub: dict[str, Any] = term.get("terms", {})
        if sub and isinstance(sub, dict):
            _resolve_terms(sub, data_dict)


def _type_to_json_schema(type_val: Any, term_name: str) -> dict[str, Any]:
    """Convert a scalar or list ``type`` value to a JSON Schema fragment."""
    # List of types → anyOf
    if isinstance(type_val, list):
        any_of: list[dict[str, Any]] = []
        for t in type_val:
            # YAML parses bare `null` as Python None
            if t is None or t == "null":
                any_of.append({"type": "null"})
            elif isinstance(t, str) and t in _SCALAR_JSON_TYPE:
                any_of.append({"type": _SCALAR_JSON_TYPE[t]})
            else:
                raise ValueError(
                    f"Type '{t}' in list type for term '{term_name}' is not supported "
                    "inside anyOf — only scalar types and 'null' are allowed."
                )
        return {"anyOf": any_of}

    # Normalise None → "str"
    if type_val is None:
        type_val = "str"

    if type_val in _SCALAR_JSON_TYPE:
        return {"type": _SCALAR_JSON_TYPE[type_val]}

    raise ValueError(f"Unexpected type value '{type_val}' for term '{term_name}'")


def _term_to_json_schema(name: str, term: dict[str, Any]) -> dict[str, Any]:
    """Convert a single term spec to its JSON Schema fragment."""

    # ref → $ref shorthand
    ref = term.get("ref")
    if ref:
        return {"$ref": f"#/$defs/{ref}"}

    type_val = term.get("type", "str")
    description: str = term.get("description", "")
    sub_terms: dict[str, Any] = term.get("terms", {})

    # ── list of types → anyOf ──────────────────────────────────────────────
    if isinstance(type_val, list):
        schema = _type_to_json_schema(type_val, name)

    # ── scalar types ───────────────────────────────────────────────────────
    elif type_val in _SCALAR_JSON_TYPE:
        schema: dict[str, Any] = {"type": _SCALAR_JSON_TYPE[type_val]}

    # ── enum ───────────────────────────────────────────────────────────────
    elif type_val == "enum":
        choices: list[str] = term.get("choices", [])
        if not choices:
            raise ValueError(
                f"Term '{name}' has type 'enum' but no 'choices' defined. "
                "Add 'choices:' or 'dynamic_choices:' and call preprocess() first."
            )
        schema = {"type": "string", "enum": choices}

    # ── list ───────────────────────────────────────────────────────────────
    elif type_val == "list":
        raw_item = term.get("item_type", "str")
        item_json = _ITEM_TYPE_MAP.get(raw_item, "string")
        choices = term.get("choices", [])
        item_schema: dict[str, Any] = {"type": item_json}
        if choices:
            item_schema["enum"] = choices
        schema = {"type": "array", "items": item_schema}

    # ── dict ───────────────────────────────────────────────────────────────
    elif type_val == "dict":
        if sub_terms:
            props: dict[str, Any] = {}
            req: list[str] = []
            for sub_name, sub_spec in sub_terms.items():
                _validate_term_name(sub_name)
                props[sub_name] = _term_to_json_schema(sub_name, sub_spec)
                req.append(sub_name)
            schema = {
                "type": "object",
                "properties": props,
                "required": req,
                "additionalProperties": False,
            }
        else:
            schema = {"type": "object"}

    # ── any ────────────────────────────────────────────────────────────────
    elif type_val == "any":
        schema = {}

    else:
        raise ValueError(f"Unknown type '{type_val}' for term '{name}'")

    if description:
        schema["description"] = description
    return schema


def _collect_confidence(
    terms: dict[str, Any], prefix: str, out: dict[str, list[str]]
) -> None:
    for name, term in terms.items():
        if not isinstance(term, dict):
            continue
        path = f"{prefix}.{name}" if prefix else name
        if term.get("config_confidence", False):
            choices = term.get("choices", [])
            if choices:
                out[path] = list(choices)
        sub_terms = term.get("terms", {})
        if sub_terms and isinstance(sub_terms, dict):
            _collect_confidence(sub_terms, path, out)


def _append_prompt_terms(
    lines: list[str], terms: dict[str, Any], indent: int
) -> None:
    pad = " " * indent
    for name, term in terms.items():
        if not isinstance(term, dict):
            continue
        type_str = term.get("type") or f"ref:{term.get('ref', '?')}"
        desc = term.get("description", "")
        choices = term.get("choices")
        suffix = f" choices={choices}" if choices else ""
        lines.append(f"{pad}- {name} ({type_str}){suffix}: {desc}")
        sub = term.get("terms", {})
        if sub and isinstance(sub, dict):
            _append_prompt_terms(lines, sub, indent + 4)
