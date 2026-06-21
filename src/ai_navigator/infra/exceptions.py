from __future__ import annotations


class AINavigatorError(Exception):
    """Base exception for all ai-navigator errors."""



class ParseError(AINavigatorError):
    """Raised when LLM response parsing fails."""

    def __init__(self, message: str, raw_content: str | None = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class StorageError(AINavigatorError):
    """Raised when conversation storage operations fail."""


class SchemaError(AINavigatorError):
    """Raised when schema definition or validation fails."""


class PreProcessorError(AINavigatorError):
    """Raised when pre-processing (e.g. image encoding) fails."""
