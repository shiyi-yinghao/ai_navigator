from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from ai_navigator.infra.types import CallStatus


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
        Examples: ``temperature``, ``max_tokens``, ``top_p``.

    configs:
        Package-internal control knobs consumed by ai-navigator stages
        (not forwarded to the provider).

    reference:
        Derived artefacts shared across pipeline stages, e.g.
        ``{"schema": <SchemaComposer>}``.

    result:
        Populated by the final processing stage with the parsed LLM output.

    status:
        ``None`` while the request is in-flight; set to the
        :class:`~ai_navigator.infra.types.CallStatus` from the completed
        :class:`~ai_navigator.infra.types.NavigatorResult` when done.
    """

    request_data: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    configs: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    status: CallStatus | None = None
