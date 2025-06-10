"""
Custom exception classes for the thesis assistant.
"""


class ThesisAssistantError(Exception):
    """Base exception for thesis assistant errors."""
    pass


class ApplicationError(ThesisAssistantError):
    """Raised when there's a general application error."""
    pass


class ConfigurationError(ThesisAssistantError):
    """Raised when there's a configuration error."""
    pass


class DocumentProcessingError(ThesisAssistantError):
    """Raised when document processing fails."""
    pass


class IndexingError(ThesisAssistantError):
    """Raised when indexing operations fail."""
    pass


class SearchError(ThesisAssistantError):
    """Raised when search operations fail."""
    pass


class APIError(ThesisAssistantError):
    """Raised when API calls fail."""
    pass


class ThesisProjectError(ThesisAssistantError):
    """Raised when thesis project operations fail."""
    pass


class FileTrackingError(ThesisAssistantError):
    """Raised when file tracking operations fail."""
    pass


class ProcessingError(ThesisAssistantError):
    """Raised when general processing operations fail."""
    pass


class ValidationError(ThesisAssistantError):
    """Raised when validation operations fail."""
    pass


class AnalysisError(ThesisAssistantError):
    """Raised when analysis operations fail."""
    pass


class ReasoningError(ThesisAssistantError):
    """Raised when reasoning operations fail."""
    pass


class SemanticMatchingError(ThesisAssistantError):
    """Raised when semantic matching operations fail."""
    pass

