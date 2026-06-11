"""Offline batch inference — background processing with SQLite progress tracking.

Workflow
--------
1. ``submit()`` — persist all items, launch a daemon thread, return ``job_id``.
2. ``job_status(job_id)`` — query progress at any time.
3. ``get_results(job_id)`` — retrieve results (partial if still running).

Usage::

    from ai_navigator.batch_inference import OfflineBatch

    job_id = OfflineBatch(method="chat").submit(
        source="requests.jsonl",
        params={"temperature": 0.3},
        configs={"model_name": "my_claude"},
    )

    # ... later (same or new process) ...
    status = OfflineBatch().job_status(job_id)
    # {"job_id": "...", "status": "running", "total": 200, "completed": 87, ...}

    results = OfflineBatch().get_results(job_id)

Or via the Navigator facade::

    nav = Navigator()
    job_id = nav.offline_submit(source="requests.jsonl", configs={"model_name": "my_claude"})
    nav.offline_status(job_id)
    nav.offline_results(job_id)
"""
from __future__ import annotations
import json
import threading
import uuid
from pathlib import Path
from typing import Any

from ai_navigator.batch_inference.storage import get_batch_storage_class
from ai_navigator.monitor.logger import get_logger

_log = get_logger("batch_inference.offline")


class OfflineBatch:
    """Background batch inference with SQLite-backed progress tracking.

    The navigator is initialised lazily on first :meth:`submit` call via
    :func:`~ai_navigator.service.base_navigator.get_navigator_class`.
    :meth:`job_status` and :meth:`get_results` only need storage — no
    navigator is created for query-only instances.

    Parameters
    ----------
    method:
        Name of the call method to invoke on the navigator (``"chat"``,
        ``"response"``, or any custom method added by a plugin).
    storage:
        Custom storage backend.  Discovered from ``ai_navigator.storage``
        entry points if omitted (falls back to SQLite).
    """

    def __init__(self, method: str = "chat", storage: Any = None) -> None:
        self._method = method
        if storage is None:
            storage_cls = get_batch_storage_class()
            storage = storage_cls()
        self._storage = storage
        self._nav: Any = None

    def _get_nav(self) -> Any:
        if self._nav is None:
            from ai_navigator.service.base_navigator import get_navigator_class
            self._nav = get_navigator_class()()
        return self._nav

    # ── Submission ────────────────────────────────────────────────────────────

    def submit(
        self,
        source: str | list[dict],
        params: dict | None = None,
        configs: dict | None = None,
        job_id: str | None = None,
    ) -> str:
        """Register and start a batch job in the background.

        Parameters
        ----------
        source:
            Path to a JSONL file or a plain ``list[dict]``.
        params:
            Shared params forwarded to every provider call.
        configs:
            Shared configs — must contain ``model_name``.
        job_id:
            Optional custom job ID.  A UUID is generated if omitted.

        Returns
        -------
        str
            The ``job_id`` — use it with :meth:`job_status` and
            :meth:`get_results`.
        """
        items = _load_items(source)
        params = params or {}
        configs = configs or {}
        job_id = job_id or str(uuid.uuid4())

        self._storage.create_job(
            job_id,
            total=len(items),
            meta={"params": params, "configs": configs, "method": self._method},
        )
        self._storage.add_items(job_id, items)

        thread = threading.Thread(
            target=self._process_job,
            args=(job_id, params, configs),
            daemon=True,
            name=f"offline-batch-{job_id[:8]}",
        )
        thread.start()
        _log.info("job %s submitted — %d items, method=%s", job_id, len(items), self._method)
        return job_id

    # ── Progress & results ────────────────────────────────────────────────────

    def job_status(self, job_id: str) -> dict | None:
        """Return current job progress, or ``None`` if not found.

        Keys: ``job_id``, ``status``, ``total``, ``completed``, ``failed``,
        ``created_at``, ``updated_at``.

        Status values: ``pending`` → ``running`` → ``completed`` |
        ``completed_with_errors``.
        """
        return self._storage.get_job_status(job_id)

    def get_results(self, job_id: str) -> list[dict] | None:
        """Retrieve stored results (partial if job is still running).

        Each element: ``{"item_idx", "status", "result", "error"}``.
        Returns ``None`` if *job_id* is not found.
        """
        return self._storage.get_results(job_id)

    # ── Background worker ─────────────────────────────────────────────────────

    def _process_job(self, job_id: str, params: dict, configs: dict) -> None:
        self._storage.update_job_status(job_id, "running")
        fn = getattr(self._get_nav(), self._method)
        pending = self._storage.get_pending_items(job_id)
        total = len(pending)

        for item_idx, request_data in pending:
            try:
                result = fn(request_data, params=params, configs=configs)
                self._storage.record_item_result(job_id, item_idx, result)
            except Exception as exc:
                _log.error("job %s item %d failed: %s", job_id, item_idx, exc)
                self._storage.record_item_error(job_id, item_idx, str(exc))

            done = item_idx + 1
            if done % max(1, total // 10) == 0 or done == total:
                _log.info("job %s progress: %d/%d", job_id, done, total)

        status_row = self._storage.get_job_status(job_id)
        final = "completed" if (status_row and status_row["failed"] == 0) else "completed_with_errors"
        self._storage.update_job_status(job_id, final)
        _log.info("job %s finished — %s", job_id, final)


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
