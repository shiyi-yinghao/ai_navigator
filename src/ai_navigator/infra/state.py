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
            User-only input — the AI has not replied yet.  Value is a bare
            string or a list of content-part dicts (text, images, …).
            Use for straightforward Q&A where no prior context is needed.

        ``{"conversation": list[Message]}``
            Full back-and-forth history with alternating ``user`` /
            ``assistant`` turns, passed through unchanged.  Use when the
            model needs to be aware of prior context.

        ``{"prompt": list, "data_dict": dict}``
            Prompt-engineering preset (zero-shot or few-shot).  The
            ``"prompt"`` value is a YAML-parsed template list that encodes
            the task structure (system instructions, examples, placeholders);
            ``data_dict`` supplies the dynamic values at call time.

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
