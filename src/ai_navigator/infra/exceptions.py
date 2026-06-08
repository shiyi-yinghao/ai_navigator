from __future__ import annotations


class AINavigatorError(Exception):
    """Base exception for all ai-navigator errors."""


class ProviderError(AINavigatorError):
    """Raised when a provider API call fails."""

    def __init__(self, message: str, provider: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Raised when the provider returns a rate limit / quota error."""

    def __init__(
        self, message: str, provider: str, retry_after: float | None = None
    ) -> None:
        super().__init__(message, provider, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Raised when the API key is invalid or missing."""

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(message, provider, status_code=401)


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
