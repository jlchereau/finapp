"""
Unit tests for app.flows.helpers module.

Tests cover provider result validation, asyncio.gather utilities,
and error handling scenarios.
"""

import pytest
from unittest.mock import Mock
import pandas as pd

from app.flows.helpers import (
    validate_provider_result,
    process_multiple_provider_results,
    validate_single_provider_task,
)
from app.providers.base import ProviderResult


class TestValidateProviderResult:
    """Test validate_provider_result function."""

    def test_successful_validation(self):
        """Test successful provider result validation."""
        # Create mock successful result
        result = Mock(spec=ProviderResult)
        result.success = True
        result.data = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )

        # Validate
        data = validate_provider_result(result, "TestData")

        # Verify
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3
        assert list(data.columns) == ["A"]

    def test_exception_result(self):
        """Test validation when result is an exception."""
        exception = ValueError("API error")

        # Should raise the same exception
        with pytest.raises(ValueError, match="API error"):
            validate_provider_result(exception, "TestData")

    def test_provider_failure(self):
        """Test validation when provider reports failure."""
        result = Mock(spec=ProviderResult)
        result.success = False
        result.error_message = "Provider API failed"

        # Should raise exception with provider error
        with pytest.raises(
            Exception, match="TestData data fetch failed: Provider API failed"
        ):
            validate_provider_result(result, "TestData")

    def test_provider_failure_no_error_message(self):
        """Test validation when provider fails without error message."""
        result = Mock(spec=ProviderResult)
        result.success = False
        # No error_message attribute

        # Should raise exception with default message
        with pytest.raises(
            Exception, match="TestData data fetch failed: Unknown TestData fetch error"
        ):
            validate_provider_result(result, "TestData")

    def test_invalid_data_attribute(self):
        """Test validation when data attribute is missing or invalid."""
        result = Mock(spec=ProviderResult)
        result.success = True
        # No data attribute

        # Should raise exception about invalid data
        with pytest.raises(
            Exception, match="TestData data fetch failed: Unknown TestData data error"
        ):
            validate_provider_result(result, "TestData")

    def test_non_dataframe_data(self):
        """Test validation when data is not a DataFrame."""
        result = Mock(spec=ProviderResult)
        result.success = True
        result.data = "not a dataframe"

        # Should raise exception about invalid data
        with pytest.raises(Exception, match="TestData data fetch failed"):
            validate_provider_result(result, "TestData")

    def test_empty_dataframe(self):
        """Test validation when DataFrame is empty."""
        result = Mock(spec=ProviderResult)
        result.success = True
        result.data = pd.DataFrame()

        # Should raise exception about empty data
        with pytest.raises(Exception, match="No testdata data available"):
            validate_provider_result(result, "TestData")

    def test_success_attribute_missing(self):
        """Test validation when success attribute is missing."""
        result = Mock()
        # No success attribute

        # Should raise exception with default message
        with pytest.raises(Exception, match="TestData data fetch failed"):
            validate_provider_result(result, "TestData")


class TestProcessMultipleProviderResults:
    """Test process_multiple_provider_results function."""

    @pytest.mark.asyncio
    async def test_empty_tasks(self):
        """Test processing with empty tasks dictionary."""
        result = await process_multiple_provider_results({})
        assert result == {}

    @pytest.mark.asyncio
    async def test_successful_multiple_tasks(self):
        """Test processing multiple successful tasks."""

        # Create mock tasks
        async def mock_task1():
            result = Mock(spec=ProviderResult)
            result.success = True
            result.data = pd.DataFrame(
                {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
            )
            result.execution_time = 0.5
            return result

        async def mock_task2():
            result = Mock(spec=ProviderResult)
            result.success = True
            result.data = pd.DataFrame(
                {"B": [4, 5, 6]}, index=pd.date_range("2023-01-01", periods=3)
            )
            result.execution_time = 0.3
            return result

        tasks = {"ticker1": mock_task1(), "ticker2": mock_task2()}

        # Process tasks
        results = await process_multiple_provider_results(tasks)

        # Verify results
        assert len(results) == 2

        # Check ticker1 result
        assert results["ticker1"]["success"] is True
        assert isinstance(results["ticker1"]["data"], pd.DataFrame)
        assert results["ticker1"]["execution_time"] == 0.5

        # Check ticker2 result
        assert results["ticker2"]["success"] is True
        assert isinstance(results["ticker2"]["data"], pd.DataFrame)
        assert results["ticker2"]["execution_time"] == 0.3

    @pytest.mark.asyncio
    async def test_mixed_success_failure(self):
        """Test processing with mix of successful and failed tasks."""

        # Create mock tasks
        async def mock_task_success():
            result = Mock(spec=ProviderResult)
            result.success = True
            result.data = pd.DataFrame(
                {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
            )
            return result

        async def mock_task_exception():
            raise ValueError("Network error")

        async def mock_task_provider_failure():
            result = Mock(spec=ProviderResult)
            result.success = False
            result.error_message = "API rate limit"
            return result

        tasks = {
            "success_ticker": mock_task_success(),
            "exception_ticker": mock_task_exception(),
            "failure_ticker": mock_task_provider_failure(),
        }

        # Process tasks
        results = await process_multiple_provider_results(tasks)

        # Verify results
        assert len(results) == 3

        # Check successful result
        assert results["success_ticker"]["success"] is True
        assert isinstance(results["success_ticker"]["data"], pd.DataFrame)

        # Check exception result
        assert results["exception_ticker"]["success"] is False
        assert "Network error" in results["exception_ticker"]["error"]

        # Check provider failure result
        assert results["failure_ticker"]["success"] is False
        assert "API rate limit" in results["failure_ticker"]["error"]


class TestValidateSingleProviderTask:
    """Test validate_single_provider_task function."""

    def test_successful_validation_with_empty_check(self):
        """Test successful validation with empty check enabled."""
        result = Mock(spec=ProviderResult)
        result.success = True
        result.data = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )

        data = validate_single_provider_task(result, "TestData", check_empty=True)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3

    def test_successful_validation_no_empty_check(self):
        """Test successful validation with empty check disabled."""
        result = Mock(spec=ProviderResult)
        result.success = True
        result.data = pd.DataFrame()  # Empty DataFrame

        # Should not raise exception when check_empty=False
        data = validate_single_provider_task(result, "TestData", check_empty=False)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 0

    def test_empty_dataframe_with_check(self):
        """Test validation fails on empty DataFrame when check enabled."""
        result = Mock(spec=ProviderResult)
        result.success = True
        result.data = pd.DataFrame()

        # Should raise exception when check_empty=True (default)
        with pytest.raises(Exception, match="No testdata data available"):
            validate_single_provider_task(result, "TestData")
