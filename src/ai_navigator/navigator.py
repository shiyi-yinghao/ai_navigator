"""Navigator — unified user-facing interface.

Wraps :class:`~ai_navigator.service.base_navigator.BaseNavigator` (or its
entry-point replacement) for single requests, and provides inline access to
online and offline batch inference.

Usage::

    from ai_navigator import Navigator

    nav = Navigator()

    # Single request
    result = nav.chat(
        request_data={"message": "Hello!"},
        params={"temperature": 0.7},
        configs={"model_name": "my_claude"},
    )

    # Online batch — blocks until all items finish
    results = nav.online_batch(
        source="requests.jsonl",
        configs={"model_name": "my_claude"},
        method="chat",
    )

    # Offline batch — background processing
    job_id = nav.offline_submit(
        source="requests.jsonl",
        configs={"model_name": "my_claude"},
        method="chat",
    )
    nav.offline_status(job_id)   # {"status": "running", "completed": 42, ...}
    nav.offline_results(job_id)  # list of result dicts
"""
from __future__ import annotations
from typing import Any, Union

from ai_navigator.service.base_navigator import get_navigator_class
from ai_navigator.state.data_class import (
    ContentPart, Message, TokenUsage, Response,
    NavigatorResult, StatusDetail,
    make_message, make_content_part,
)

__all__ = [
    "Navigator",
    "ContentPart",
    "Message",
    "TokenUsage",
    "Response",
    "NavigatorResult",
    "StatusDetail",
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


# ── Navigator facade ──────────────────────────────────────────────────────────

class Navigator:
    """Unified access point: single requests + batch inference.

    The underlying navigator class is resolved from ``ai_navigator.navigator``
    entry points (falls back to :class:`~ai_navigator.service.base_navigator.BaseNavigator`).

    """

    def __init__(self) -> None:
        self._nav = get_navigator_class()()
        self._offline_query: Any = None

    # ── Single-request interface ──────────────────────────────────────────────

    def chat(
        self,
        request_data: dict,
        params: dict | None = None,
        configs: dict | None = None,
    ) -> NavigatorResult:
        return self._nav.chat(request_data, params=params, configs=configs)

    def response(
        self,
        request_data: dict,
        params: dict | None = None,
        configs: dict | None = None,
    ) -> NavigatorResult:
        return self._nav.response(request_data, params=params, configs=configs)

    def __getattr__(self, name: str) -> Any:
        """Delegate any plugin-added methods to the underlying navigator."""
        return getattr(self._nav, name)

    # ── Online batch ──────────────────────────────────────────────────────────

    def online_batch(
        self,
        source: str,
        params: dict | None = None,
        configs: dict | None = None,
        method: str = "chat",
        max_workers: int = 8,
    ) -> list[Any]:
        """Run concurrent batch inference from a JSONL file; blocks until all items complete."""
        from ai_navigator.batch_inference.online import OnlineBatch
        return OnlineBatch(method=method, max_workers=max_workers).run(source, params, configs)

    # ── Offline batch ─────────────────────────────────────────────────────────

    def offline_submit(
        self,
        source: str,
        params: dict | None = None,
        configs: dict | None = None,
        method: str = "chat",
        job_id: str | None = None,
    ) -> str:
        """Stream a JSONL file into storage and start background processing; returns ``job_id`` immediately."""
        from ai_navigator.batch_inference.offline import OfflineBatch
        return OfflineBatch(method=method).submit(source, params=params, configs=configs, job_id=job_id)

    def offline_status(self, job_id: str) -> dict | None:
        """Return progress of a background job."""
        return self._get_offline_query().job_status(job_id)

    def offline_results(self, job_id: str) -> list[dict] | None:
        """Return results for a background job (partial if still running)."""
        return self._get_offline_query().get_results(job_id)

    def _get_offline_query(self) -> Any:
        if self._offline_query is None:
            from ai_navigator.batch_inference.offline import OfflineBatch
            self._offline_query = OfflineBatch()
        return self._offline_query
