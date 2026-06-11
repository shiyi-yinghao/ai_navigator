"""Online batch inference — file-based, concurrent, blocks until done.

Items are read from a JSONL file in batches of ``batch_size`` and dispatched
concurrently via a thread pool.  Results are returned in the same order as
the input once every batch has resolved.

``batch_size`` is resolved as ``min(ConstConfigs.BATCH_SIZE, configs["batch_size"])``
— the system cap always wins.

Usage::

    from ai_navigator.batch_inference import OnlineBatch

    results = OnlineBatch(method="chat", max_workers=10).run(
        source="requests.jsonl",
        params={"temperature": 0.5},
        configs={"model_name": "my_claude"},
    )

Or via the Navigator facade::

    from ai_navigator import Navigator

    nav = Navigator()
    results = nav.online_batch(source="requests.jsonl", configs={"model_name": "my_claude"})
"""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ai_navigator.monitor.logger import get_logger
from ai_navigator.param.const_configs import ConstConfigs

_log = get_logger("batch_inference.online")


class OnlineBatch:
    """Concurrent batch inference from a JSONL file.

    The navigator is initialised lazily on first :meth:`run` call via
    :func:`~ai_navigator.service.base_navigator.get_navigator_class`.

    Parameters
    ----------
    method:
        Name of the call method to invoke on the navigator (``"chat"``,
        ``"response"``, or any custom method added by a plugin).
    max_workers:
        Maximum concurrent provider calls per batch (default: 8).
    """

    def __init__(self, method: str = "chat", max_workers: int = 8) -> None:
        self._method = method
        self._max_workers = max_workers
        self._nav: Any = None

    def _get_nav(self) -> Any:
        if self._nav is None:
            from ai_navigator.service.base_navigator import get_navigator_class
            self._nav = get_navigator_class()()
        return self._nav

    def run(
        self,
        source: str | Path,
        params: dict | None = None,
        configs: dict | None = None,
    ) -> list[Any]:
        """Stream JSONL file in batches and dispatch each batch concurrently.

        Parameters
        ----------
        source:
            Path to a JSONL file (one ``request_data`` dict per line).
        params:
            Shared params forwarded to every provider call.
        configs:
            Shared configs — must contain ``model_name``.  Optionally
            ``batch_size`` to cap the system default.

        Returns
        -------
        list
            One entry per input item, in input order.  Failed items are
            represented as ``{"error": "<message>"}``.
        """
        params = params or {}
        configs = configs or {}

        sys_size = ConstConfigs.BATCH_SIZE
        req_size = configs.get("batch_size", sys_size)
        batch_size = min(sys_size, req_size)

        fn = getattr(self._get_nav(), self._method)
        results: list[Any] = []

        _log.info("online batch start — source=%s, batch_size=%d, method=%s",
                  source, batch_size, self._method)

        with open(Path(source), encoding="utf-8") as fh:
            batch: list[dict] = []
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                batch.append(json.loads(line))
                if len(batch) >= batch_size:
                    results.extend(self._run_batch(fn, batch, params, configs))
                    batch = []
            if batch:
                results.extend(self._run_batch(fn, batch, params, configs))

        _log.info("online batch complete — %d items total", len(results))
        return results

    def _run_batch(
        self,
        fn: Any,
        batch: list[dict],
        params: dict,
        configs: dict,
    ) -> list[Any]:
        ordered: list[Any] = [None] * len(batch)
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_pos = {
                executor.submit(fn, item, params, configs): pos
                for pos, item in enumerate(batch)
            }
            for future in as_completed(future_to_pos):
                pos = future_to_pos[future]
                try:
                    ordered[pos] = future.result()
                except Exception as exc:
                    _log.error("batch item %d failed: %s", pos, exc)
                    ordered[pos] = {"error": str(exc)}
        _log.info("batch of %d done", len(batch))
        return ordered
