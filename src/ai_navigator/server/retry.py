"""Retry policy for transient LLM provider rate-limit errors.

Lives in the ``server`` layer so that
:class:`~ai_navigator.server.base_server.BaseServer` can delegate its retry
logic here.

Effective retry count
---------------------
:meth:`~ai_navigator.server.base_server.BaseServer._invoke` receives
``_retry_max`` as a popped kwarg.  The caller
(:class:`~ai_navigator.service.base_navigator.BaseNavigator`) computes::

    effective = min(
        credentials.get("retry_max", ConstConfigs.RETRY_MAX),
        configs.get("retry_max", ConstConfigs.RETRY_MAX),
    )

The smaller value wins — accounts cap the maximum, callers can lower it further.

Usage::

    from ai_navigator.server.retry import RetryPolicy

    policy = RetryPolicy(max_retries=3, initial_wait=1.0, backoff=2.0)
    result = policy.execute(fn, *args, **kwargs)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ai_navigator.infra.exceptions import RateLimitError

_log = logging.getLogger("ai_navigator.server.retry")


class RetryPolicy:
    """Exponential back-off retry for :class:`~ai_navigator.infra.exceptions.RateLimitError`.

    Parameters
    ----------
    max_retries:
        Number of retry attempts after the initial failure.
    initial_wait:
        Seconds to wait before the first retry.
    backoff:
        Multiplier applied to wait time after each retry.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_wait: float = 1.0,
        backoff: float = 2.0,
    ) -> None:
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.backoff = backoff

    def execute(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call *fn* with retry on :class:`~ai_navigator.infra.exceptions.RateLimitError`.

        Non-rate-limit exceptions propagate immediately without retry.
        """
        wait = self.initial_wait
        last_exc: RateLimitError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except RateLimitError as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                _log.warning(
                    "rate limited (attempt %d/%d) — retrying in %.1fs",
                    attempt + 1, self.max_retries, wait,
                )
                time.sleep(wait)
                wait *= self.backoff

        raise last_exc  # type: ignore[misc]
