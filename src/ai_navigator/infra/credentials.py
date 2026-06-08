"""Credentials loading.

:class:`CredentialsLoader` reads a YAML file whose path comes from
:class:`~ai_navigator.infra.const_configs.ConstConfigs`.  Subclass it and
override :meth:`fetch` to load from a different source (Vault, AWS Secrets
Manager, a database, …)::

    from ai_navigator.infra.credentials import CredentialsLoader

    class VaultLoader(CredentialsLoader):
        def fetch(self) -> dict:
            # call your secrets backend here
            return vault_client.read("secret/ai_navigator")["data"]

Usage::

    loader = CredentialsLoader()          # or VaultLoader()
    creds = loader.fetch()
    api_key = creds.get("openai_api_key", "")
"""
from __future__ import annotations
import yaml
from typing import Any

from ai_navigator.infra.const_configs import ConstConfigs
from ai_navigator.infra.logger import get_logger

_log = get_logger("credentials")


class CredentialsLoader:
    """Load credentials from a YAML file.

    Default behaviour
    -----------------
    Reads the YAML file at ``ConstConfigs.CREDENTIALS_PATH`` and returns its
    contents as a plain ``dict``.  Returns an empty dict on any error (file
    not found, permission denied, malformed YAML) so callers always receive a
    safe value.

    Override
    --------
    Subclass and override :meth:`fetch` to pull credentials from any source.
    The only contract is that :meth:`fetch` returns a ``dict``.
    """

    def fetch(self) -> dict[str, Any]:
        """Return credentials as a plain dict.

        Falls back to ``{}`` on any I/O or parse error.
        """
        path = ConstConfigs.CREDENTIALS_PATH
        try:
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            _log.debug("credentials: file not found at %s", path)
            return {}
        except PermissionError as exc:
            _log.warning("credentials: permission denied reading %s (%s)", path, exc)
            return {}
        except Exception as exc:
            _log.warning("credentials: fetch failed (%s)", exc)
            return {}
