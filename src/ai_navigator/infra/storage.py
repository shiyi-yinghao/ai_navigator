# Backward-compat shim — storage has moved to ai_navigator.monitor.storage
from ai_navigator.monitor.storage import StorageBase, StoreStatus

__all__ = ["StorageBase", "StoreStatus"]
