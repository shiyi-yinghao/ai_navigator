from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StatusCode(Enum):
    PENDING = "pending"
    OK = "ok"
    ERROR = "error"


@dataclass
class Status:
    code: StatusCode = StatusCode.PENDING
    message: str = ""


@dataclass
class RequestState:
    """Pipeline state container passed through the request/response lifecycle.

    Fields
    ------
    request_data:
        Raw input descriptor.  One of three shapes:

        ``{"message": str | list}``
            User-assembled content.  Value is a bare string or a list
            of content-part dicts (keys: ``type``, ``text`` / ``image_url`` …).

        ``{"conversation": list[Message]}``
            Fully pre-assembled conversation passed through unchanged.

        ``{"prompt": list, "data_dict": dict}``
            YAML-driven prompt.  ``"prompt"`` value is the parsed template
            list; ``data_dict`` supplies dynamic substitutions.

    params:
        LLM / server parameters passed directly through to the provider call.
        Examples: ``temperature``, ``max_tokens``, ``top_p``, ``logprobs``,
        ``top_logprobs``, ``stop``.

    configs:
        Package-internal control knobs consumed by ai-navigator stages
        (not forwarded to the provider).  Examples:

        ``term_extract_discard`` (bool, default ``True``)
            When ``True``, a term that is expanded (dict recursed, list
            flattened) is removed from the result under its own key.

        ``extract_list_elements`` (bool, default ``False``)
            Expand list terms into numbered keys (``term_1``, ``term_2``, …).

    reference:
        Derived artefacts shared across pipeline stages, e.g.
        ``{"schema": <SchemaComposer>}``.  The processed schema lives here
        so downstream stages can access it without extra function arguments.

    result:
        Populated by the final processing stage with the parsed LLM output.

    status:
        Current processing status.
    """

    request_data: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    configs: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    status: Status = field(default_factory=Status)
