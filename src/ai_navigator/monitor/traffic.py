"""Traffic monitoring — per-account and per-user query / token tracking.

Maintains two rolling windows per entity:
  - current minute  (keyed by UTC "YYYYMMDDHHMM", resets when the minute changes)
  - current day     (keyed by UTC "YYYYMMDD", resets at midnight UTC)

Two-phase update keeps counters accurate under concurrent requests:

  1. :meth:`TrafficMonitor.on_request_enter` — increments query count and adds
     an estimated input-token cost (2 × current-user average; initial default
     5 000 if no history exists yet).
  2. :meth:`TrafficMonitor.on_request_complete` — replaces the estimate with
     the actual token usage reported by the provider.

Metrics are tracked at two granularities (12 variables total):

  - **account level** — keyed by ``account_name`` from credentials
  - **account | user level** — keyed by ``(account_name, user)`` from configs

``@traffic_monitor`` decorator
------------------------------
Apply to :class:`~ai_navigator.service.base_navigator.BaseNavigator` methods.
The decorator:

  1. Reads pre-request stats and invokes the :class:`RequestRateLimiter` hook.
  2. If the limiter returns ``False`` (or raises), returns an error
     :class:`~ai_navigator.infra.types.NavigatorResult` without calling the
     underlying method.
  3. Records the request enter (increments counters with an estimated token
     cost).
  4. Calls the wrapped method; catches any exception and converts it to an error
     ``NavigatorResult``.
  5. Records the request complete (corrects the token estimate with actual
     usage).

Rate limiter Entry Point
------------------------
Register a callable under ``ai_navigator.traffic`` to replace the default
allow-all limiter::

    # pyproject.toml
    [project.entry-points."ai_navigator.traffic"]
    my_limiter = "my_package.hooks:rate_limiter"

Signature::

    def rate_limiter(configs: dict, stats: TrafficStats) -> bool:
        qpm = configs.get("_qpm", 60)
        return stats["account_min_queries"] < qpm

Return ``True`` to allow the request, ``False`` to block it.  Only the first
registered entry point is used.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps
from importlib.metadata import entry_points
from threading import Lock
from typing import Any, Callable, Protocol, TypedDict

from ai_navigator.infra.types import NavigatorResult
from ai_navigator.monitor.status_codes import StatusCode

_log = logging.getLogger("ai_navigator.monitor.traffic")

_DEFAULT_TOKEN_ESTIMATE = 5_000


# ── Stats TypedDict ───────────────────────────────────────────────────────────

class TrafficStats(TypedDict):
    """Snapshot of the 12 traffic metrics at a given point in time."""
    # account level — current minute
    account_min_queries: int
    account_min_input_tokens: int
    account_min_output_tokens: int
    # account level — current day
    account_day_queries: int
    account_day_input_tokens: int
    account_day_output_tokens: int
    # account | user level — current minute
    user_min_queries: int
    user_min_input_tokens: int
    user_min_output_tokens: int
    # account | user level — current day
    user_day_queries: int
    user_day_input_tokens: int
    user_day_output_tokens: int


# ── Rate limiter protocol ─────────────────────────────────────────────────────

class RequestRateLimiter(Protocol):
    """Callable that decides whether a request should be allowed.

    Return ``True`` to allow, ``False`` to block.
    """
    def __call__(self, configs: dict, stats: TrafficStats) -> bool: ...


def _default_limiter(configs: dict, stats: TrafficStats) -> bool:
    return True


# ── Traffic monitor ───────────────────────────────────────────────────────────

class TrafficMonitor:
    """Thread-safe in-memory tracker for query and token usage.

    One shared instance per process — use :func:`get_traffic_monitor`.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        # {(account_name, time_bucket): {"queries": int, "input_tokens": int, "output_tokens": int}}
        self._account: dict = defaultdict(_zero)
        # {(account_name, user, time_bucket): {...}}
        self._user: dict = defaultdict(_zero)

    # ── Public interface ──────────────────────────────────────────────────────

    def on_request_enter(
        self,
        account_name: str,
        user: str,
    ) -> tuple[int, str, str]:
        """Increment counters for a new request.

        Returns
        -------
        (estimated_tokens, minute_key, day_key)
            Pass all three back to :meth:`on_request_complete`.
        """
        mk = _minute_key()
        dk = _day_key()
        estimated = self._estimate(account_name, user, mk)
        with self._lock:
            _inc(self._account[(account_name, mk)],         1, estimated, 0)
            _inc(self._account[(account_name, dk)],         1, estimated, 0)
            _inc(self._user[(account_name, user, mk)],      1, estimated, 0)
            _inc(self._user[(account_name, user, dk)],      1, estimated, 0)
        _log.debug("enter  acct=%s user=%s est=%d", account_name, user, estimated)
        return estimated, mk, dk

    def on_request_complete(
        self,
        account_name: str,
        user: str,
        estimated: int,
        usage: dict,
        minute_key: str,
        day_key: str,
    ) -> None:
        """Correct the token estimate with actual provider usage.

        Parameters
        ----------
        estimated:
            The value returned by the matching :meth:`on_request_enter` call.
        usage:
            ``result["usage"]`` dict with keys ``prompt_tokens``,
            ``completion_tokens``, etc.
        minute_key / day_key:
            The keys returned by the matching :meth:`on_request_enter` call.
        """
        actual_in  = usage.get("prompt_tokens",      0)
        actual_out = usage.get("completion_tokens",   0)
        correction = actual_in - estimated

        with self._lock:
            _inc(self._account[(account_name, minute_key)],      0, correction, actual_out)
            _inc(self._account[(account_name, day_key)],         0, correction, actual_out)
            _inc(self._user[(account_name, user, minute_key)],   0, correction, actual_out)
            _inc(self._user[(account_name, user, day_key)],      0, correction, actual_out)
        _log.debug("complete acct=%s user=%s in=%d out=%d", account_name, user, actual_in, actual_out)

    def get_stats(self, account_name: str, user: str = "default") -> TrafficStats:
        """Return a live stats snapshot for the current minute and day."""
        return self._snapshot(account_name, user, _minute_key(), _day_key())

    # ── Internals ─────────────────────────────────────────────────────────────

    def _estimate(self, account_name: str, user: str, minute_key: str) -> int:
        with self._lock:
            w = self._user.get((account_name, user, minute_key))
            if w and w["queries"] > 0:
                avg = w["input_tokens"] / w["queries"]
                return max(int(avg * 2), 1)
        return _DEFAULT_TOKEN_ESTIMATE

    def _snapshot(self, account_name: str, user: str, mk: str, dk: str) -> TrafficStats:
        with self._lock:
            a_min = dict(self._account.get((account_name, mk), _zero()))
            a_day = dict(self._account.get((account_name, dk), _zero()))
            u_min = dict(self._user.get((account_name, user, mk), _zero()))
            u_day = dict(self._user.get((account_name, user, dk), _zero()))
        return TrafficStats(
            account_min_queries=      a_min["queries"],
            account_min_input_tokens= a_min["input_tokens"],
            account_min_output_tokens=a_min["output_tokens"],
            account_day_queries=      a_day["queries"],
            account_day_input_tokens= a_day["input_tokens"],
            account_day_output_tokens=a_day["output_tokens"],
            user_min_queries=         u_min["queries"],
            user_min_input_tokens=    u_min["input_tokens"],
            user_min_output_tokens=   u_min["output_tokens"],
            user_day_queries=         u_day["queries"],
            user_day_input_tokens=    u_day["input_tokens"],
            user_day_output_tokens=   u_day["output_tokens"],
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_monitor: TrafficMonitor | None = None
_monitor_lock = Lock()


def get_traffic_monitor() -> TrafficMonitor:
    """Return the process-wide shared :class:`TrafficMonitor` instance."""
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = TrafficMonitor()
    return _monitor


# ── Rate limiter entry point ──────────────────────────────────────────────────

_limiter_cache: RequestRateLimiter | None = None
_limiter_lock = Lock()


def get_rate_limiter() -> RequestRateLimiter:
    """Return the active :class:`RequestRateLimiter`.

    Loads from the first ``ai_navigator.traffic`` entry point; falls back to
    the default allow-all limiter if none is registered.
    """
    global _limiter_cache
    if _limiter_cache is not None:
        return _limiter_cache
    with _limiter_lock:
        if _limiter_cache is not None:
            return _limiter_cache
        eps = list(entry_points(group="ai_navigator.traffic"))
        if not eps:
            _limiter_cache = _default_limiter
        else:
            try:
                _limiter_cache = eps[0].load()
                _log.info("rate limiter loaded: %s", eps[0].name)
            except Exception as exc:
                _log.warning("rate limiter '%s' failed to load: %s — using default", eps[0].name, exc)
                _limiter_cache = _default_limiter
    return _limiter_cache


# ── Decorator ────────────────────────────────────────────────────────────────

def traffic_monitor(fn: Callable) -> Callable:
    """Decorator for :class:`~ai_navigator.service.base_navigator.BaseNavigator`
    ``chat`` and ``response`` methods.

    Responsibilities:
    - Invoke the :class:`RequestRateLimiter` hook before the call.
    - Record request enter / complete for traffic reporting.
    - Catch any exception from the wrapped method and convert it to an error
      :class:`~ai_navigator.infra.types.NavigatorResult`.

    The wrapped method must return a :class:`~ai_navigator.infra.types.NavigatorResult`.
    """
    @wraps(fn)
    def wrapper(self: Any, request_data: dict, params: Any = None, configs: Any = None) -> NavigatorResult:
        configs = configs or {}
        model_name = configs.get("model_name", "")
        user = configs.get("user", "default")
        account_name = self._get_account_name(model_name) if model_name else "unknown"

        monitor = get_traffic_monitor()

        # Rate limiter sees pre-request stats (before incrementing)
        stats = monitor.get_stats(account_name, user)
        try:
            allowed = get_rate_limiter()(configs, stats)
        except Exception as exc:
            return _make_err(StatusCode.INTERNAL_ERROR, str(exc))

        if not allowed:
            return _make_err(StatusCode.TOO_MANY_REQUESTS, "request blocked by rate limiter")

        # Record enter — counters increment after limiter allows
        estimated, mk, dk = monitor.on_request_enter(account_name, user)

        try:
            nav_result: NavigatorResult = fn(self, request_data, params=params, configs=configs)
            monitor.on_request_complete(account_name, user, estimated, nav_result.get("usage", {}), mk, dk)
            return nav_result
        except Exception as exc:
            monitor.on_request_complete(account_name, user, estimated, {}, mk, dk)
            return _make_err(StatusCode.INTERNAL_ERROR, str(exc))

    return wrapper


def _make_err(code: int, detail: str) -> NavigatorResult:
    from ai_navigator.monitor.status_codes import describe as status_describe
    return NavigatorResult(
        result="",
        status={"status_code": code, "status_desc": status_describe(code), "status_detail": detail},
        usage={},
        reference={},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zero() -> dict[str, int]:
    return {"queries": 0, "input_tokens": 0, "output_tokens": 0}


def _inc(w: dict[str, int], dq: int, din: int, dout: int) -> None:
    w["queries"]       += dq
    w["input_tokens"]  += din
    w["output_tokens"] += dout


def _minute_key() -> str:
    return datetime.now(timezone.utc).strftime("min_%Y%m%d%H%M")


def _day_key() -> str:
    return datetime.now(timezone.utc).strftime("day_%Y%m%d")
