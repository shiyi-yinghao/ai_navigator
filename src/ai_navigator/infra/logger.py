# Backward-compat shim — logger has moved to ai_navigator.monitor.logger
from ai_navigator.monitor.logger import get_logger

__all__ = ["get_logger"]
