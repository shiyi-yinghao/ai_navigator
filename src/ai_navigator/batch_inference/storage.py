"""SQLite-backed storage for offline batch jobs, with Entry Points replacement support.

Tables
------
batch_jobs   — one row per job: status, progress counters, timestamps.
batch_items  — one row per item: request_data, result, error.

All operations open and close their own connection so this class is safe to
call from multiple threads (the offline worker thread + any query thread).

Replacing the storage backend via Entry Points
----------------------------------------------
Register a class that satisfies :class:`BatchStorageProtocol` under
``ai_navigator.storage``::

    # their pyproject.toml
    [project.entry-points."ai_navigator.storage"]
    redis = "my_package.storage:RedisBatchStorage"

Only ONE replacement is active — the first registered plugin wins.

Alternatively, pass your storage directly at construction time::

    batch = OfflineBatch(nav, storage=RedisBatchStorage())
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from importlib.metadata import entry_points
from typing import Any, Iterable, Protocol, runtime_checkable

from ai_navigator.infra.const_configs import ConstConfigs
from ai_navigator.monitor.logger import get_logger

_log = get_logger("batch_inference.storage")


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class BatchStorageProtocol(Protocol):
    """Interface that a custom batch storage backend must implement."""

    def create_job(self, job_id: str, total: int, meta: dict) -> bool: ...
    def add_items(self, job_id: str, items: Iterable[dict]) -> int: ...
    def update_job_total(self, job_id: str, total: int) -> None: ...
    def update_job_status(self, job_id: str, status: str) -> None: ...
    def record_item_result(self, job_id: str, item_idx: int, result: Any) -> None: ...
    def record_item_error(self, job_id: str, item_idx: int, error: str) -> None: ...
    def get_job_status(self, job_id: str) -> dict | None: ...
    def get_results(self, job_id: str) -> list[dict] | None: ...
    def get_pending_items(self, job_id: str) -> list[tuple[int, dict]]: ...


# ── Entry Points discovery ────────────────────────────────────────────────────

_batch_storage_class_cache: type | None = None


def get_batch_storage_class() -> type:
    """Return the active batch storage class.

    Checks ``ai_navigator.storage`` entry points first; if none are installed,
    returns the built-in :class:`BatchStorage`.  Only the first plugin wins.
    """
    global _batch_storage_class_cache
    if _batch_storage_class_cache is not None:
        return _batch_storage_class_cache

    eps = list(entry_points(group="ai_navigator.storage"))
    if not eps:
        _batch_storage_class_cache = BatchStorage
        return BatchStorage

    if len(eps) > 1:
        _log.warning(
            "multiple storage plugins found (%s), using first: %s",
            [e.name for e in eps],
            eps[0].name,
        )

    try:
        cls = eps[0].load()
        _log.info("using storage plugin: %s", eps[0].name)
        _batch_storage_class_cache = cls
        return cls
    except Exception as exc:
        _log.warning("storage plugin '%s' failed to load: %s — using SQLite", eps[0].name, exc)
        _batch_storage_class_cache = BatchStorage
        return BatchStorage


class BatchStorage:
    """Manage batch job state in the shared SQLite database."""

    def _get_db_path(self) -> str:
        return ConstConfigs.STORAGE_PATH

    def _ensure_ready(self) -> bool:
        if not hasattr(self, "_db_ok"):
            self._db_ok = self._init_db()
        return self._db_ok

    def _init_db(self) -> bool:
        try:
            con = sqlite3.connect(self._get_db_path())
            con.executescript("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    job_id     TEXT PRIMARY KEY,
                    status     TEXT NOT NULL,
                    total      INTEGER NOT NULL,
                    completed  INTEGER DEFAULT 0,
                    failed     INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    meta       TEXT
                );
                CREATE TABLE IF NOT EXISTS batch_items (
                    job_id       TEXT NOT NULL,
                    item_idx     INTEGER NOT NULL,
                    status       TEXT NOT NULL,
                    request_data TEXT NOT NULL,
                    result       TEXT,
                    error        TEXT,
                    PRIMARY KEY (job_id, item_idx)
                );
            """)
            con.commit()
            con.close()
            return True
        except Exception as exc:
            _log.warning("batch storage init failed: %s", exc)
            return False

    def _connect(self) -> sqlite3.Connection | None:
        if not self._ensure_ready():
            return None
        try:
            return sqlite3.connect(self._get_db_path(), check_same_thread=False)
        except Exception as exc:
            _log.warning("batch storage connect failed: %s", exc)
            return None

    # ── Job lifecycle ─────────────────────────────────────────────────────────

    def create_job(self, job_id: str, total: int, meta: dict) -> bool:
        con = self._connect()
        if con is None:
            return False
        try:
            now = _now()
            con.execute(
                "INSERT INTO batch_jobs VALUES (?, 'pending', ?, 0, 0, ?, ?, ?)",
                (job_id, total, now, now, json.dumps(meta, ensure_ascii=False)),
            )
            con.commit()
            return True
        except Exception as exc:
            _log.warning("create_job failed: %s", exc)
            return False
        finally:
            con.close()

    def add_items(self, job_id: str, items: Iterable[dict]) -> int:
        """Stream items into storage. Returns count of items written."""
        con = self._connect()
        if con is None:
            return 0
        count = 0
        try:
            def _gen():
                nonlocal count
                for item in items:
                    yield (job_id, count, json.dumps(item, ensure_ascii=False))
                    count += 1
            con.executemany(
                "INSERT INTO batch_items (job_id, item_idx, status, request_data) "
                "VALUES (?, ?, 'pending', ?)",
                _gen(),
            )
            con.commit()
            return count
        except Exception as exc:
            _log.warning("add_items failed: %s", exc)
            return count
        finally:
            con.close()

    def update_job_total(self, job_id: str, total: int) -> None:
        con = self._connect()
        if con is None:
            return
        try:
            con.execute(
                "UPDATE batch_jobs SET total=?, updated_at=? WHERE job_id=?",
                (total, _now(), job_id),
            )
            con.commit()
        except Exception as exc:
            _log.warning("update_job_total failed: %s", exc)
        finally:
            con.close()

    def update_job_status(self, job_id: str, status: str) -> None:
        con = self._connect()
        if con is None:
            return
        try:
            con.execute(
                "UPDATE batch_jobs SET status=?, updated_at=? WHERE job_id=?",
                (status, _now(), job_id),
            )
            con.commit()
        except Exception as exc:
            _log.warning("update_job_status failed: %s", exc)
        finally:
            con.close()

    def record_item_result(self, job_id: str, item_idx: int, result: Any) -> None:
        con = self._connect()
        if con is None:
            return
        try:
            con.execute(
                "UPDATE batch_items SET status='completed', result=? "
                "WHERE job_id=? AND item_idx=?",
                (json.dumps(_sanitize(result), ensure_ascii=False), job_id, item_idx),
            )
            con.execute(
                "UPDATE batch_jobs SET completed=completed+1, updated_at=? WHERE job_id=?",
                (_now(), job_id),
            )
            con.commit()
        except Exception as exc:
            _log.warning("record_item_result failed: %s", exc)
        finally:
            con.close()

    def record_item_error(self, job_id: str, item_idx: int, error: str) -> None:
        con = self._connect()
        if con is None:
            return
        try:
            con.execute(
                "UPDATE batch_items SET status='failed', error=? "
                "WHERE job_id=? AND item_idx=?",
                (error, job_id, item_idx),
            )
            con.execute(
                "UPDATE batch_jobs SET failed=failed+1, updated_at=? WHERE job_id=?",
                (_now(), job_id),
            )
            con.commit()
        except Exception as exc:
            _log.warning("record_item_error failed: %s", exc)
        finally:
            con.close()

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_job_status(self, job_id: str) -> dict | None:
        con = self._connect()
        if con is None:
            return None
        try:
            row = con.execute(
                "SELECT job_id, status, total, completed, failed, created_at, updated_at "
                "FROM batch_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "job_id":     row[0],
                "status":     row[1],
                "total":      row[2],
                "completed":  row[3],
                "failed":     row[4],
                "created_at": row[5],
                "updated_at": row[6],
            }
        except Exception as exc:
            _log.warning("get_job_status failed: %s", exc)
            return None
        finally:
            con.close()

    def get_results(self, job_id: str) -> list[dict] | None:
        """Return all items for *job_id*, ordered by item_idx.

        Each element has ``item_idx``, ``status``, ``result`` (dict or None),
        and ``error`` (str or None).  Partial results are returned if the job
        is still running.
        """
        con = self._connect()
        if con is None:
            return None
        try:
            rows = con.execute(
                "SELECT item_idx, status, result, error FROM batch_items "
                "WHERE job_id=? ORDER BY item_idx",
                (job_id,),
            ).fetchall()
            return [
                {
                    "item_idx": row[0],
                    "status":   row[1],
                    "result":   json.loads(row[2]) if row[2] else None,
                    "error":    row[3],
                }
                for row in rows
            ]
        except Exception as exc:
            _log.warning("get_results failed: %s", exc)
            return None
        finally:
            con.close()

    def get_pending_items(self, job_id: str) -> list[tuple[int, dict]]:
        con = self._connect()
        if con is None:
            return []
        try:
            rows = con.execute(
                "SELECT item_idx, request_data FROM batch_items "
                "WHERE job_id=? AND status='pending' ORDER BY item_idx",
                (job_id,),
            ).fetchall()
            return [(row[0], json.loads(row[1])) for row in rows]
        except Exception as exc:
            _log.warning("get_pending_items failed: %s", exc)
            return []
        finally:
            con.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(result: Any) -> Any:
    """Strip un-serialisable 'raw' SDK object before writing to SQLite."""
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if k != "raw"}
    return result
