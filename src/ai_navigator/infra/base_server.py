# BaseServer has moved to ai_navigator.server.base_server.
# This file is kept only so that old imports emit a clear error instead of
# a silent AttributeError.

from ai_navigator.server.base_server import BaseServer as BaseServer  # re-export

__all__ = ["BaseServer"]
