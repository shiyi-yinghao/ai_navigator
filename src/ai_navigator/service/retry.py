"""Retry policy for transient LLM provider errors.

Currently handles :class:`~ai_navigator.infra.exceptions.RateLimitError`
with configurable exponential back-off.  Other error types are re-raised
immediately without retry.

Configuration (via :class:`~ai_navigator.param.const_configs.ConstConfigs`
or environment variables):

  AI_NAVIGATOR_RETRY_MAX      — maximum number of retry attempts (default: 3)
  AI_NAVIGATOR_RETRY_WAIT     — initial wait in seconds before first retry (default: 1.0)
  AI_NAVIGATOR_RETRY_BACKOFF  — multiplier applied to wait after each attempt (default: 2.0)

Usage::

    from ai_navigator.service.retry import get_retry_policy

    result = get_retry_policy().execute(server.chat, messages, **params)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ai_navigator.infra.exceptions import RateLimitError

_log = logging.getLogger("ai_navigator.service.retry")


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


# ── Module-level cached instance ──────────────────────────────────────────────

_policy: RetryPolicy | None = None


def get_retry_policy() -> RetryPolicy:
    """Return a :class:`RetryPolicy` configured from :class:`~ai_navigator.param.const_configs.ConstConfigs`."""
    global _policy
    if _policy is None:
        from ai_navigator.param.const_configs import ConstConfigs
        _policy = RetryPolicy(
            max_retries=ConstConfigs.RETRY_MAX,
            initial_wait=ConstConfigs.RETRY_WAIT,
            backoff=ConstConfigs.RETRY_BACKOFF,
        )
    return _policy
