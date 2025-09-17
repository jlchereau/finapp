"""
Unit tests for custom exception classes.
"""

from app.lib.exceptions import (
    FinAppException,
    FlowException,
    PageInputException,
    PageOutputException,
)


class TestFinAppException:
    """Test cases for the base FinAppException class."""

    def test_basic_exception_creation(self):
        """Test basic exception creation with minimal parameters."""
        exc = FinAppException("Test error message")

        assert exc.message == "Test error message"
        assert exc.user_message == "An unexpected error occurred"
        assert exc.error_id is not None
        assert len(exc.error_id) == 8  # UUID truncated to 8 chars
        assert exc.context == {}

    def test_exception_with_all_parameters(self):
        """Test exception creation with all parameters."""
        context = {"key": "value", "number": 42}
        exc = FinAppException(
            message="Technical error",
            user_message="User-friendly error",
            error_id="test123",
            context=context,
        )

        assert exc.message == "Technical error"
        assert exc.user_message == "User-friendly error"
        assert exc.error_id == "test123"
        assert exc.context == context

    def test_to_dict(self):
        """Test conversion to dictionary."""
        context = {"test": "data"}
        exc = FinAppException(
            message="Test message",
            user_message="User message",
            error_id="abc123",
            context=context,
        )

        result = exc.to_dict()

        assert result["error_id"] == "abc123"
        assert result["message"] == "Test message"
        assert result["user_message"] == "User message"
        assert result["context"] == context
        assert result["exception_type"] == "FinAppException"


class TestFlowException:
    """Test cases for FlowException."""

    def test_workflow_exception_defaults(self):
        """Test FlowException with default user message."""
        exc = FlowException(
            workflow="test_workflow", step="data_fetch", message="Step failed"
        )

        assert exc.workflow == "test_workflow"
        assert exc.step == "data_fetch"
        assert exc.message == "Step failed"
        assert "flow failed" in exc.user_message
        assert exc.context["workflow"] == "test_workflow"
        assert exc.context["step"] == "data_fetch"


class TestPageInputException:
    """Test cases for PageInputException."""

    def test_page_input_exception_defaults(self):
        """Test PageInputException with default user message."""
        exc = PageInputException(
            field="ticker", value="INVALID", message="Invalid ticker format"
        )

        assert exc.field == "ticker"
        assert exc.value == "INVALID"
        assert exc.message == "Invalid ticker format"
        assert "ticker" in exc.user_message
        assert exc.context["field"] == "ticker"
        assert exc.context["value"] == "INVALID"


class TestPageOutputException:
    """Test cases for PageOutputException."""

    def test_page_output_exception_defaults(self):
        """Test PageOutputException with default user message."""
        exc = PageOutputException(
            output_type="returns chart", message="Chart rendering failed"
        )

        assert exc.output_type == "returns chart"
        assert exc.message == "Chart rendering failed"
        assert "returns chart" in exc.user_message
        assert exc.context["output_type"] == "returns chart"
