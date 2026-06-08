"""Result extractor — maps LLM output to a flat result dict using the schema.

The extractor walks the response dict according to the schema's term structure.
Whether a term is "expanded" depends on the active **parse types**:

Default parse types
-------------------
``dict``
    Always active.  Sub-terms are walked recursively; keys become dot-notation
    paths (e.g. ``detail.score``).

Optional parse type
-------------------
``list``  (enabled by ``params["extract_list_elements"] = True``)
    Array values are flattened with a 1-based numeric suffix:
    ``soldiers`` → ``soldiers_1``, ``soldiers_2``, …

Configs keys  (``RequestState.configs``, not forwarded to LLM)
--------------------------------------------------------------
``extract_list_elements`` (bool, default ``False``)
    Add ``list`` to the active parse types.

``term_extract_discard`` (bool, default ``True``)
    Controls what happens to an expanded term's *original* key.

    ``True``  (default) — the parent key is **discarded**; only the expanded
    children appear in the result.

    ``False`` — the parent key is **kept** alongside the expanded children,
    so the result contains both the original nested value and all derived keys.

    Examples (schema: ``detail`` dict with sub-terms ``reason``, ``score``):

    discard=True  → ``{"detail.reason": ..., "detail.score": ...}``
    discard=False → ``{"detail": {...}, "detail.reason": ..., "detail.score": ...}``
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_navigator.schema.composer import SchemaComposer


class ResultExtractor:
    """Map an LLM response dict to a flat result using schema term definitions.

    Example — default (dict expanded, original discarded)
    -----------------------------------------------------
    ::

        data = {
            "title":  "Phone",
            "detail": {"reason": "fast", "score": 9},
            "tags":   ["speed", "price"],
        }
        ResultExtractor().extract(data, composer)
        # → {
        #     "title":         "Phone",
        #     "detail.reason": "fast",   ← dict expanded, "detail" discarded
        #     "detail.score":  9,
        #     "tags":          ["speed", "price"],
        # }

    Example — keep parent key (term_extract_discard=False)
    -------------------------------------------------------
    ::

        ResultExtractor().extract(data, composer,
                                  configs={"term_extract_discard": False})
        # → {
        #     "title":         "Phone",
        #     "detail":        {"reason": "fast", "score": 9},  ← also kept
        #     "detail.reason": "fast",
        #     "detail.score":  9,
        #     "tags":          ["speed", "price"],
        # }

    Example — list expansion
    ------------------------
    ::

        ResultExtractor().extract(data, composer,
                                  configs={"extract_list_elements": True})
        # → {
        #     "title":         "Phone",
        #     "detail.reason": "fast",
        #     "detail.score":  9,
        #     "tags_1":        "speed",   ← list expanded, "tags" discarded
        #     "tags_2":        "price",
        # }
    """

    def extract(
        self,
        data: dict[str, Any],
        composer: "SchemaComposer",
        configs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract values from *data* guided by *composer*'s term schema.

        Parameters
        ----------
        data:
            Parsed LLM output dict (e.g. from ``ResponseParser.parse_response()``).
        composer:
            The :class:`~ai_navigator.schema.composer.SchemaComposer` that
            describes the expected output shape.
        configs:
            Package-internal control knobs (``RequestState.configs``).
            Not forwarded to the LLM provider.

            ``term_extract_discard`` (bool, default ``True``)
                When ``True``, a term whose type is being parsed (expanded) does
                NOT appear in the result under its own key — only its derived
                children do.  When ``False``, the original key is also kept.

            ``extract_list_elements`` (bool, default ``False``)
                Expand list terms into numbered keys
                (``term_1``, ``term_2``, …).
        """
        c = configs or {}
        discard: bool = c.get("term_extract_discard", True)

        parse_types: set[str] = {"dict"}
        if c.get("extract_list_elements", False):
            parse_types.add("list")

        result: dict[str, Any] = {}
        _collect(data, composer._terms, "", result, parse_types, discard)
        return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _collect(
    data: Any,
    terms: dict[str, Any],
    prefix: str,
    result: dict[str, Any],
    parse_types: set[str],
    discard: bool,
) -> None:
    """Recursively walk *terms*, writing values into *result*."""
    for name, term in terms.items():
        if not isinstance(term, dict):
            continue

        path = f"{prefix}.{name}" if prefix else name
        value = data.get(name) if isinstance(data, dict) else None
        type_str: str = term.get("type", "str") or "str"

        if type_str == "dict" and "dict" in parse_types:
            sub_terms: dict[str, Any] = term.get("terms", {})
            if sub_terms and isinstance(sub_terms, dict) and isinstance(value, dict):
                if not discard:
                    result[path] = value          # keep original alongside children
                _collect(value, sub_terms, path, result, parse_types, discard)
                continue                          # skip the plain assignment below

        if type_str == "list" and "list" in parse_types:
            if isinstance(value, list):
                if not discard:
                    result[path] = value          # keep original alongside elements
                for i, item in enumerate(value, 1):
                    result[f"{path}_{i}"] = item
                continue

        result[path] = value
