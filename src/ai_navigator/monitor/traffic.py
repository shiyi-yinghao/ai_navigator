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

Entry Points hook
-----------------
Register a callable under ``ai_navigator.traffic`` to intercept requests at
entry time.  The hook receives ``configs`` and the pre-increment
:class:`TrafficStats` snapshot.  Raise any exception to block the request::

    # pyproject.toml
    [project.entry-points."ai_navigator.traffic"]
    my_hook = "my_package.hooks:check_rate_limit"

Hook signature::

    def check_rate_limit(configs: dict, stats: TrafficStats) -> None:
        qpm = configs.get("_qpm")          # passed through from credentials
        if stats["account_min_queries"] >= qpm:
            from ai_navigator.infra.exceptions import RateLimitError
            raise RateLimitError("QPM exceeded for account")

If no hook is registered, all requests are allowed through.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import entry_points
from threading import Lock
from typing import Callable, TypedDict

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


# ── Traffic monitor ───────────────────────────────────────────────────────────

class TrafficMonitor:
    """Thread-safe in-memory tracker for query and token usage.

    One shared instance per process — use :func:`get_traffic_monitor`.

    Credentials should carry ``account_name``, ``qpm``, and ``tpm`` fields;
    the request ``configs`` should carry ``user`` (default ``"default"``).
    These are resolved by the caller (:class:`~ai_navigator.service.base_navigator.BaseNavigator`)
    before calling the monitor.
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
        configs: dict,
    ) -> tuple[int, str, str]:
        """Record a new incoming request.

        Flow:
        1. Snapshot pre-increment stats.
        2. Call the registered traffic hook (may raise to block).
        3. Compute token estimate (2 × current-user average, min 5 000).
        4. Increment all four counters (account/user × minute/day).

        Parameters
        ----------
        account_name:
            From ``credentials[model_name][0]["account_name"]``.
        user:
            From ``configs.get("user", "default")``.
        configs:
            Full request configs dict (forwarded to the hook as-is).

        Returns
        -------
        (estimated_tokens, minute_key, day_key)
            Pass all three back to :meth:`on_request_complete`.
        """
        mk = _minute_key()
        dk = _day_key()

        # Snapshot before increment — hook sees current state
        stats = self._snapshot(account_name, user, mk, dk)
        _invoke_hook(configs, stats)

        # Estimate and increment
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
            The keys returned by the matching :meth:`on_request_enter` call
            (capturing the minute at request start avoids off-by-one on
            minute boundaries).
        """
        actual_in  = usage.get("prompt_tokens",      0)
        actual_out = usage.get("completion_tokens",   0)
        correction = actual_in - estimated            # may be negative

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


# ── Entry Points hook ─────────────────────────────────────────────────────────

_hook_cache: list[Callable] | None = None
_hook_lock = Lock()


def _load_hooks() -> list[Callable]:
    global _hook_cache
    if _hook_cache is not None:
        return _hook_cache
    with _hook_lock:
        if _hook_cache is not None:
            return _hook_cache
        hooks: list[Callable] = []
        for ep in entry_points(group="ai_navigator.traffic"):
            try:
                hooks.append(ep.load())
                _log.info("traffic hook loaded: %s", ep.name)
            except Exception as exc:
                _log.warning("traffic hook '%s' failed to load: %s", ep.name, exc)
        _hook_cache = hooks
    return _hook_cache


def _invoke_hook(configs: dict, stats: TrafficStats) -> None:
    """Call every registered traffic hook. Propagates exceptions to block requests."""
    for hook in _load_hooks():
        hook(configs, stats)  # intentionally not caught — raise to block


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
