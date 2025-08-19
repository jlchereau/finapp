"""
Unit tests for custom exception classes.
"""

from app.lib.exceptions import (
    FinAppException,
    DataFetchException,
    DataProcessingException,
    WorkflowException,
    UserInputException,
    ChartException,
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


class TestDataFetchException:
    """Test cases for DataFetchException."""

    def test_data_fetch_exception_defaults(self):
        """Test DataFetchException with default user message."""
        exc = DataFetchException(
            provider="test_provider", query="test_query", message="Provider failed"
        )

        assert exc.provider == "test_provider"
        assert exc.query == "test_query"
        assert exc.message == "Provider failed"
        assert "test_provider" in exc.user_message
        assert exc.context["provider"] == "test_provider"
        assert exc.context["query"] == "test_query"

    def test_data_fetch_exception_custom_message(self):
        """Test DataFetchException with custom user message."""
        exc = DataFetchException(
            provider="yahoo",
            query="AAPL",
            message="API rate limited",
            user_message="Custom user message",
        )

        assert exc.user_message == "Custom user message"


class TestDataProcessingException:
    """Test cases for DataProcessingException."""

    def test_data_processing_exception_defaults(self):
        """Test DataProcessingException with default user message."""
        exc = DataProcessingException(
            operation="normalize_data", message="Normalization failed"
        )

        assert exc.operation == "normalize_data"
        assert exc.message == "Normalization failed"
        assert "normalize_data" in exc.user_message
        assert exc.context["operation"] == "normalize_data"


class TestWorkflowException:
    """Test cases for WorkflowException."""

    def test_workflow_exception_defaults(self):
        """Test WorkflowException with default user message."""
        exc = WorkflowException(
            workflow="test_workflow", step="data_fetch", message="Step failed"
        )

        assert exc.workflow == "test_workflow"
        assert exc.step == "data_fetch"
        assert exc.message == "Step failed"
        assert "workflow failed" in exc.user_message
        assert exc.context["workflow"] == "test_workflow"
        assert exc.context["step"] == "data_fetch"


class TestUserInputException:
    """Test cases for UserInputException."""

    def test_user_input_exception_defaults(self):
        """Test UserInputException with default user message."""
        exc = UserInputException(
            field="ticker", value="INVALID", message="Invalid ticker format"
        )

        assert exc.field == "ticker"
        assert exc.value == "INVALID"
        assert exc.message == "Invalid ticker format"
        assert "ticker" in exc.user_message
        assert exc.context["field"] == "ticker"
        assert exc.context["value"] == "INVALID"


class TestChartException:
    """Test cases for ChartException."""

    def test_chart_exception_defaults(self):
        """Test ChartException with default user message."""
        exc = ChartException(chart_type="returns", message="Chart rendering failed")

        assert exc.chart_type == "returns"
        assert exc.message == "Chart rendering failed"
        assert "returns chart" in exc.user_message
        assert exc.context["chart_type"] == "returns"
