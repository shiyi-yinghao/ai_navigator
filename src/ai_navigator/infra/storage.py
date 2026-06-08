"""Storage and metric integration for caching and debugging.

The database path is read from :class:`~ai_navigator.infra.const_configs.ConstConfigs`
(``STORAGE_PATH``) so no constructor arguments are needed.

Default backend: SQLite (three tables).

+-----------------+-------------------------------------------------------+
| table           | used by                                               |
+=================+=======================================================+
| pipeline_data   | five store/fetch pairs (request/reference/response/  |
|                 | status/result), keyed by (bucket, key)               |
+-----------------+-------------------------------------------------------+
| metrics         | metric_report / metric_load                          |
+-----------------+-------------------------------------------------------+
| cache           | cache_store / cache_fetch                            |
+-----------------+-------------------------------------------------------+

All I/O is wrapped in ``try/except`` — permission errors or a locked DB
degrade gracefully (log a warning, return ``None`` / ``StoreStatus.ERROR``).

Customisation
-------------
Subclass :class:`StorageBase` and override any pair(s) you need to replace.
Override :meth:`_get_db_path` to use a different SQLite file::

    class TestStorage(StorageBase):
        def _get_db_path(self) -> str:
            return "/tmp/test.db"

**Store and fetch must always be overridden together.**
"""
from __future__ import annotations
import json
import sqlite3
from typing import Any, Literal

from ai_navigator.infra.const_configs import ConstConfigs
from ai_navigator.infra.logger import get_logger

_log = get_logger("storage")


# --------------------------------------------------------------------------- #
#  Status code                                                                 #
# --------------------------------------------------------------------------- #

class StoreStatus:
    OK    = "ok"
    ERROR = "error"


# --------------------------------------------------------------------------- #
#  StorageBase                                                                 #
# --------------------------------------------------------------------------- #

class StorageBase:
    """Pipeline storage backed by SQLite.

    No constructor arguments — the database path is resolved from
    :attr:`~ai_navigator.infra.const_configs.ConstConfigs.STORAGE_PATH`
    on first use.

    The database and its three tables are created automatically on first
    access.  Override :meth:`_get_db_path` to change the file location.
    """

    # ── Path resolution ───────────────────────────────────────────────────────

    def _get_db_path(self) -> str:
        """Return the SQLite file path.  Override to use a different location."""
        return ConstConfigs.STORAGE_PATH

    # ── Lazy initialisation ───────────────────────────────────────────────────

    def _ensure_ready(self) -> bool:
        """Initialise the database on first call; cache the result."""
        if not hasattr(self, "_db_ok"):
            self._db_ok: bool = self._init_db()
        return self._db_ok

    def _init_db(self) -> bool:
        try:
            con = sqlite3.connect(self._get_db_path())
            con.executescript("""
                CREATE TABLE IF NOT EXISTS pipeline_data (
                    bucket TEXT NOT NULL,
                    key    TEXT NOT NULL,
                    value  TEXT NOT NULL,
                    PRIMARY KEY (bucket, key)
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    name TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cache (
                    name TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                );
            """)
            con.commit()
            con.close()
            return True
        except Exception as exc:
            _log.warning("storage: SQLite init failed (%s) — storage disabled", exc)
            return False

    def _connect(self) -> sqlite3.Connection | None:
        if not self._ensure_ready():
            return None
        try:
            return sqlite3.connect(self._get_db_path())
        except Exception as exc:
            _log.warning("storage: cannot open SQLite (%s)", exc)
            return None

    # ── Shared SQLite helpers ─────────────────────────────────────────────────

    def _sql_store(self, bucket: str, key: str, value: Any) -> str:
        con = self._connect()
        if con is None:
            return StoreStatus.ERROR
        try:
            con.execute(
                "INSERT OR REPLACE INTO pipeline_data VALUES (?, ?, ?)",
                (bucket, key, json.dumps(value, ensure_ascii=False)),
            )
            con.commit()
            return StoreStatus.OK
        except Exception as exc:
            _log.warning("storage: store failed bucket=%s key=%s (%s)", bucket, key, exc)
            return StoreStatus.ERROR
        finally:
            con.close()

    def _sql_fetch(self, bucket: str, key: str) -> Any:
        con = self._connect()
        if con is None:
            return None
        try:
            row = con.execute(
                "SELECT value FROM pipeline_data WHERE bucket = ? AND key = ?",
                (bucket, key),
            ).fetchone()
            return json.loads(row[0]) if row else None
        except Exception as exc:
            _log.warning("storage: fetch failed bucket=%s key=%s (%s)", bucket, key, exc)
            return None
        finally:
            con.close()

    def _sql_metric(
        self,
        table: str,
        name: str,
        method: Literal["add", "update"],
        data: dict[str, Any],
    ) -> dict[str, Any] | None:
        con = self._connect()
        if con is None:
            return None
        try:
            row = con.execute(
                f"SELECT data FROM {table} WHERE name = ?", (name,)
            ).fetchone()
            current: dict[str, Any] = json.loads(row[0]) if row else {}
            if method == "add":
                for k, v in data.items():
                    current[k] = current.get(k, 0) + v
            else:
                current.update(data)
            con.execute(
                f"INSERT OR REPLACE INTO {table} VALUES (?, ?)",
                (name, json.dumps(current, ensure_ascii=False)),
            )
            con.commit()
            return current
        except Exception as exc:
            _log.warning("storage: metric op failed table=%s name=%s (%s)", table, name, exc)
            return None
        finally:
            con.close()

    def _sql_metric_load(self, table: str, name: str) -> dict[str, Any] | None:
        con = self._connect()
        if con is None:
            return None
        try:
            row = con.execute(
                f"SELECT data FROM {table} WHERE name = ?", (name,)
            ).fetchone()
            return json.loads(row[0]) if row else None
        except Exception as exc:
            _log.warning("storage: metric load failed table=%s name=%s (%s)", table, name, exc)
            return None
        finally:
            con.close()

    # ── 1. Raw user request ───────────────────────────────────────────────────

    def request_store(self, key: str, value: Any) -> str:
        return self._sql_store("request", key, value)

    def request_fetch(self, key: str) -> Any:
        return self._sql_fetch("request", key)

    # ── 2. Processed reference ────────────────────────────────────────────────

    def reference_store(self, key: str, value: Any) -> str:
        return self._sql_store("reference", key, value)

    def reference_fetch(self, key: str) -> Any:
        return self._sql_fetch("reference", key)

    # ── 3. Raw LLM response ───────────────────────────────────────────────────

    def response_store(self, key: str, value: Any) -> str:
        return self._sql_store("response", key, value)

    def response_fetch(self, key: str) -> Any:
        return self._sql_fetch("response", key)

    # ── 4. Pipeline status ────────────────────────────────────────────────────

    def status_store(self, key: str, value: Any) -> str:
        return self._sql_store("status", key, value)

    def status_fetch(self, key: str) -> Any:
        return self._sql_fetch("status", key)

    # ── 5. Extracted result ───────────────────────────────────────────────────

    def result_store(self, key: str, value: Any) -> str:
        return self._sql_store("result", key, value)

    def result_fetch(self, key: str) -> Any:
        return self._sql_fetch("result", key)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def metric_report(
        self,
        metric_name: str,
        method: Literal["add", "update"],
        data: dict[str, Any],
    ) -> None:
        """Push a metric event.

        ``"add"``    — accumulate: numeric values are summed.
        ``"update"`` — replace: existing keys are overwritten.
        """
        self._sql_metric("metrics", metric_name, method, data)

    def metric_load(self, metric_name: str) -> dict[str, Any] | None:
        """Return the current metric dict, or ``None`` if not found."""
        return self._sql_metric_load("metrics", metric_name)

    # ── Cache (high-frequency) ────────────────────────────────────────────────

    def cache_store(
        self,
        metric_name: str,
        method: Literal["add", "update"],
        data: dict[str, Any],
    ) -> Any:
        """Write to the cache table and return the post-write value."""
        return self._sql_metric("cache", metric_name, method, data)

    def cache_fetch(
        self,
        metric_name: str,
        method: Literal["add", "update"],
        data: dict[str, Any],
    ) -> Any:
        """Read from the cache table.  ``method`` and ``data`` are unused in
        the default implementation but kept in the signature so subclasses can
        exploit them (e.g. a Redis GET with optional side-effects)."""
        return self._sql_metric_load("cache", metric_name)
