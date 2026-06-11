"""ai-navigator — simplified LLM calls and prompt construction."""

from ai_navigator.navigator import Navigator
from ai_navigator.server.base_server import BaseServer
from ai_navigator.infra.exceptions import (
    AINavigatorError,
    AuthenticationError,
    ParseError,
    ProviderError,
    RateLimitError,
    SchemaError,
)
from ai_navigator.infra.types import ContentPart, Message, Response, TokenUsage, NavigatorResult, CallStatus
from ai_navigator.service.base_navigator import BaseNavigator, get_navigator_class
from ai_navigator.infra.state import RequestState, Status, StatusCode
from ai_navigator.param.const_configs import ConstConfigs
from ai_navigator.param.credentials import CredentialsLoader, get_credentials_class
from ai_navigator.monitor.storage import StorageBase, StoreStatus
from ai_navigator.monitor.logger import get_logger
from ai_navigator.parser.response import ResponseParser
from ai_navigator.pre_processor.image import ImageProcessor
from ai_navigator.schema.composer import SchemaComposer
from ai_navigator.schema.extractor import ResultExtractor
from ai_navigator.conf_parser.prompt import PromptBuilder
from ai_navigator.batch_inference.online import OnlineBatch
from ai_navigator.batch_inference.offline import OfflineBatch
from ai_navigator.batch_inference.storage import BatchStorageProtocol, get_batch_storage_class

__version__ = "0.4.0"

__all__ = [
    # Core
    "Navigator",
    "BaseNavigator",
    "BaseServer",
    # Data models
    "Message",
    "Response",
    "TokenUsage",
    "ContentPart",
    "NavigatorResult",
    "CallStatus",
    # Pipeline state
    "RequestState",
    "Status",
    "StatusCode",
    # Config & credentials
    "ConstConfigs",
    "CredentialsLoader",
    "get_credentials_class",
    "get_navigator_class",
    # Storage & logging
    "StorageBase",
    "StoreStatus",
    "get_logger",
    # Exceptions
    "AINavigatorError",
    "AuthenticationError",
    "ParseError",
    "ProviderError",
    "RateLimitError",
    "SchemaError",
    # Schema
    "SchemaComposer",
    "ResultExtractor",
    # Utilities
    "ResponseParser",
    "ImageProcessor",
    "PromptBuilder",
    # Batch inference
    "OnlineBatch",
    "OfflineBatch",
    "BatchStorageProtocol",
    "get_batch_storage_class",
]
