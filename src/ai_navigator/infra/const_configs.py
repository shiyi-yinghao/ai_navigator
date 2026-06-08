"""Package-wide configuration constants.

Each constant is read from the corresponding environment variable at import
time, falling back to a hard-coded default.  Override programmatically when
needed (e.g. in tests)::

    from ai_navigator.infra.const_configs import ConstConfigs
    ConstConfigs.STORAGE_PATH = "/tmp/test.db"

Or via environment before the process starts::

    AI_NAVIGATOR_STORAGE_PATH=/data/ai.db python ...
"""
import os


class ConstConfigs:
    # ── Storage ───────────────────────────────────────────────────────────────
    # SQLite database file used by StorageBase.
    STORAGE_PATH: str = os.environ.get(
        "AI_NAVIGATOR_STORAGE_PATH", "ai_navigator.db"
    )

    # ── Credentials ───────────────────────────────────────────────────────────
    # YAML file read by CredentialsLoader.fetch().
    CREDENTIALS_PATH: str = os.environ.get(
        "AI_NAVIGATOR_CREDENTIALS_PATH", "credentials.yaml"
    )
