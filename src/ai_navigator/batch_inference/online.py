"""Online batch inference — concurrent processing, connection stays open.

All items are dispatched in parallel via a thread pool.  Results are returned
in the same order as the input once every item has resolved.

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

_log = get_logger("batch_inference.online")


class OnlineBatch:
    """Concurrent batch inference.

    The navigator is initialised lazily on first :meth:`run` call via
    :func:`~ai_navigator.service.base_navigator.get_navigator_class`.

    Parameters
    ----------
    method:
        Name of the call method to invoke on the navigator (``"chat"``,
        ``"response"``, or any custom method added by a plugin).
    max_workers:
        Maximum concurrent provider calls (default: 8).
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
        source: str | list[dict],
        params: dict | None = None,
        configs: dict | None = None,
    ) -> list[Any]:
        """Dispatch all items concurrently and return results in input order.

        Parameters
        ----------
        source:
            Path to a JSONL file (one ``request_data`` dict per line) or a
            plain ``list[dict]``.
        params:
            Shared params forwarded to every provider call.
        configs:
            Shared configs — must contain ``model_name``.

        Returns
        -------
        list
            One entry per input item, in the same order.  Failed items are
            represented as ``{"error": "<message>"}``.
        """
        items = _load_items(source)
        params = params or {}
        configs = configs or {}
        fn = getattr(self._get_nav(), self._method)

        total = len(items)
        results: list[Any] = [None] * total
        _log.info("online batch start — %d items, max_workers=%d, method=%s",
                  total, self._max_workers, self._method)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(fn, item, params, configs): idx
                for idx, item in enumerate(items)
            }
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    _log.error("item %d failed: %s", idx, exc)
                    results[idx] = {"error": str(exc)}
                done += 1
                if done % max(1, total // 10) == 0 or done == total:
                    _log.info("online batch progress: %d/%d", done, total)

        _log.info("online batch complete — %d items", total)
        return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_items(source: str | list[dict]) -> list[dict]:
    if isinstance(source, list):
        return source
    items = []
    with open(Path(source), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
