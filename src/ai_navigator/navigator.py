"""Navigator — the primary user-facing entry point for ai-navigator.

Usage::

    from ai_navigator.navigator import Navigator

    nav = Navigator()

    nav.chat(
        request_data={"type": "message", "content": "Hello!"},
        params={"temperature": 0.7},
        configs={"model_name": "my_claude"},
    )

    nav.response(
        request_data={"type": "prompt", "template": [...], "data_dict": {...}},
        params={},
        configs={"model_name": "my_gpt4"},
    )

See :class:`~ai_navigator.infra.base_navigator.BaseNavigator` for credentials
file format, ``request_data`` shapes, and infrastructure details.
"""
from __future__ import annotations
from typing import Any, Union

from ai_navigator.infra.base_navigator import (
    BaseNavigator,
    ContentPart,
    Message,
    TokenUsage,
    Response,
    make_message,
    make_content_part,
)

__all__ = [
    "Navigator",
    "ContentPart",
    "Message",
    "TokenUsage",
    "Response",
    "make_content_part",
    "make_message",
    "user_message",
    "system_message",
    "assistant_message",
]


# ── Convenience constructors ──────────────────────────────────────────────────

def user_message(content: Union[str, list]) -> Message:
    return make_message("user", content)


def system_message(content: Union[str, list]) -> Message:
    return make_message("system", content)


def assistant_message(content: Union[str, list]) -> Message:
    return make_message("assistant", content)


# ── Navigator ─────────────────────────────────────────────────────────────────

class Navigator(BaseNavigator):
    """Core passthrough component.

    Inherits all credentials / server-resolution / preprocessing logic from
    :class:`~ai_navigator.infra.base_navigator.BaseNavigator`.
    Adds the public ``chat`` and ``response`` call methods.
    """

    def chat(
        self,
        request_data: dict,
        params: dict | None = None,
        configs: dict | None = None,
    ) -> Any:
        """Send a chat request.

        Parameters
        ----------
        request_data:
            ``{"type": "message",      "content": str | list}``
            ``{"type": "conversation", "messages": list[Message]}``
            ``{"type": "prompt",       "template": list, "data_dict": dict}``
        params:
            Forwarded to the provider call (temperature, max_tokens, …).
        configs:
            Internal knobs — must contain ``model_name``.  Not forwarded to
            the provider.
        """
        params = params or {}
        configs = configs or {}

        self._log_stage("request_receive", request_data)

        model_name = configs.get("model_name", "")
        if not model_name:
            raise ValueError("configs must contain 'model_name'.")
        server = self._get_server(model_name)

        messages = self._preprocess(request_data)
        self._log_stage("request_preprocess", messages)

        result = server.chat(messages, **params)
        self._log_stage("request_executed", result)
        self._log_stage("request_returned", result)
        return result

    def response(
        self,
        request_data: dict,
        params: dict | None = None,
        configs: dict | None = None,
    ) -> Any:
        """Send a structured-output request.

        Parameters
        ----------
        request_data:
            ``{"type": "message",      "content": str | list}``
            ``{"type": "conversation", "messages": list[Message]}``
            ``{"type": "prompt",       "template": list, "data_dict": dict}``
        params:
            Forwarded to the provider call (response_format, schema, …).
        configs:
            Internal knobs — must contain ``model_name``.  Not forwarded to
            the provider.
        """
        params = params or {}
        configs = configs or {}

        self._log_stage("request_receive", request_data)

        model_name = configs.get("model_name", "")
        if not model_name:
            raise ValueError("configs must contain 'model_name'.")
        server = self._get_server(model_name)

        messages = self._preprocess(request_data)
        self._log_stage("request_preprocess", messages)

        result = server.response(messages, **params)
        self._log_stage("request_executed", result)
        self._log_stage("request_returned", result)
        return result