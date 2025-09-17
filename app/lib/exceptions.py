"""
Custom exception classes for the FinApp application.

Provides domain-specific exceptions for better error handling and user experience.
"""

import uuid
from typing import Optional, Dict, Any


class FinAppException(Exception):
    """Base exception class for all FinApp-specific exceptions."""

    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        error_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize FinApp exception.

        Args:
            message: Technical error message for logging
            user_message: User-friendly error message for UI display
            error_id: Unique identifier for this error instance
            context: Additional context data for debugging
        """
        super().__init__(message)
        self.message = message
        self.user_message = user_message or "An unexpected error occurred"
        self.error_id = error_id or str(uuid.uuid4())[:8]
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_id": self.error_id,
            "message": self.message,
            "user_message": self.user_message,
            "context": self.context,
            "exception_type": self.__class__.__name__,
        }


class FlowException(FinAppException):
    """Exception raised when flow execution fails."""

    def __init__(
        self,
        workflow: str,
        step: str,
        message: str,
        user_message: Optional[str] = None,
        **kwargs,
    ):
        if user_message is None:
            user_message = (
                "Analysis flow failed. Please try again or contact support."
            )

        context = {"workflow": workflow, "step": step}
        context.update(kwargs.get("context", {}))

        super().__init__(
            message=message,
            user_message=user_message,
            context=context,
            **{k: v for k, v in kwargs.items() if k != "context"},
        )
        self.workflow = workflow
        self.step = step


class PageInputException(FinAppException):
    """Exception raised for page-level input validation errors."""

    def __init__(
        self,
        field: str,
        value: Any,
        message: str,
        user_message: Optional[str] = None,
        **kwargs,
    ):
        if user_message is None:
            user_message = (
                f"Invalid input for {field}. Please check your input and try again."
            )

        context = {"field": field, "value": str(value)}
        context.update(kwargs.get("context", {}))

        super().__init__(
            message=message,
            user_message=user_message,
            context=context,
            **{k: v for k, v in kwargs.items() if k != "context"},
        )
        self.field = field
        self.value = value


class PageOutputException(FinAppException):
    """Exception raised when page output/rendering fails."""

    def __init__(
        self,
        output_type: str,
        message: str,
        user_message: Optional[str] = None,
        **kwargs,
    ):
        if user_message is None:
            user_message = (
                f"Failed to generate {output_type}. Please try refreshing the data."
            )

        context = {"output_type": output_type}
        context.update(kwargs.get("context", {}))

        super().__init__(
            message=message,
            user_message=user_message,
            context=context,
            **{k: v for k, v in kwargs.items() if k != "context"},
        )
        self.output_type = output_type
