"""
Unit tests for app.flows.base module.

Tests cover FlowResult class functionality, type safety,
and integration patterns.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
import pandas as pd

from app.flows.base import FlowResult


class TestFlowResult:
    """Test FlowResult class."""

    def test_success_result_creation(self):
        """Test successful FlowResult creation."""
        # Create test data
        test_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        base_date = datetime(2024, 1, 1)
        execution_time = 1.5
        successful_items = ["AAPL", "MSFT"]
        failed_items = ["INVALID"]
        metadata = {"test_key": "test_value"}

        # Create successful result
        result = FlowResult.success_result(
            data=test_data,
            base_date=base_date,
            execution_time=execution_time,
            successful_items=successful_items,
            failed_items=failed_items,
            metadata=metadata,
        )

        # Verify all fields
        assert result.success is True
        assert result.data is test_data
        assert result.error_message is None
        assert result.base_date == base_date
        assert result.execution_time == execution_time
        assert result.successful_items == successful_items
        assert result.failed_items == failed_items
        assert result.metadata == metadata
        assert isinstance(result.timestamp, datetime)

    def test_error_result_creation(self):
        """Test error FlowResult creation."""
        error_message = "Test error message"
        base_date = datetime(2024, 1, 1)
        execution_time = 0.5
        failed_items = ["AAPL", "MSFT"]
        metadata = {"error_code": 500}

        # Create error result
        result = FlowResult.error_result(
            error_message=error_message,
            base_date=base_date,
            execution_time=execution_time,
            failed_items=failed_items,
            metadata=metadata,
        )

        # Verify all fields
        assert result.success is False
        assert result.data is None
        assert result.error_message == error_message
        assert result.base_date == base_date
        assert result.execution_time == execution_time
        assert result.successful_items == []
        assert result.failed_items == failed_items
        assert result.metadata == metadata
        assert isinstance(result.timestamp, datetime)

    def test_minimal_success_result(self):
        """Test successful result with minimal parameters."""
        test_data = pd.DataFrame({"A": [1, 2, 3]})

        result = FlowResult.success_result(data=test_data)

        assert result.success is True
        assert result.data is test_data
        assert result.error_message is None
        assert result.base_date is None
        assert result.execution_time is None
        assert result.successful_items == []
        assert result.failed_items == []
        assert result.metadata == {}

    def test_minimal_error_result(self):
        """Test error result with minimal parameters."""
        error_message = "Minimal error"

        result = FlowResult.error_result(error_message=error_message)

        assert result.success is False
        assert result.data is None
        assert result.error_message == error_message
        assert result.base_date is None
        assert result.execution_time is None
        assert result.successful_items == []
        assert result.failed_items == []
        assert result.metadata == {}

    def test_generic_typing(self):
        """Test generic typing with different data types."""
        # DataFrame type
        df_result = FlowResult.success_result(data=pd.DataFrame({"A": [1, 2, 3]}))
        assert isinstance(df_result.data, pd.DataFrame)

        # Dict type
        dict_data = {"key": "value", "number": 42}
        dict_result = FlowResult.success_result(data=dict_data)
        assert dict_result.data == dict_data

        # String type
        str_result = FlowResult.success_result(data="test string")
        assert str_result.data == "test string"

        # List type
        list_data = [1, 2, 3, 4]
        list_result = FlowResult.success_result(data=list_data)
        assert list_result.data == list_data

    def test_pydantic_validation(self):
        """Test Pydantic model validation."""
        # Create FlowResult directly (not via class methods)
        result = FlowResult(
            success=True,
            data=pd.DataFrame({"A": [1, 2, 3]}),
            successful_items=["AAPL"],
            failed_items=[],
            metadata={"test": True},
        )

        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)
        assert result.successful_items == ["AAPL"]
        assert result.metadata == {"test": True}
        assert isinstance(result.timestamp, datetime)

    def test_default_values(self):
        """Test default field values."""
        result = FlowResult(success=True)

        assert result.success is True
        assert result.data is None
        assert result.error_message is None
        assert result.base_date is None
        assert result.execution_time is None
        assert result.successful_items == []
        assert result.failed_items == []
        assert result.metadata == {}
        assert isinstance(result.timestamp, datetime)

    def test_timestamp_creation(self):
        """Test that timestamp is automatically set on creation."""
        before_creation = datetime.now()
        result = FlowResult.success_result(data="test")
        after_creation = datetime.now()

        # Timestamp should be between before and after
        assert before_creation <= result.timestamp <= after_creation

    def test_result_immutability_via_copy(self):
        """Test that we can create variations of results."""
        original_data = pd.DataFrame({"A": [1, 2, 3]})
        original_result = FlowResult.success_result(
            data=original_data,
            successful_items=["AAPL"],
            metadata={"original": True},
        )

        # Create new result with additional failed items
        modified_result = FlowResult.success_result(
            data=original_result.data,
            successful_items=original_result.successful_items,
            failed_items=["INVALID"],
            metadata={**original_result.metadata, "modified": True},
        )

        # Original should be unchanged
        assert original_result.failed_items == []
        assert "modified" not in original_result.metadata

        # Modified should have new data
        assert modified_result.failed_items == ["INVALID"]
        assert modified_result.metadata["modified"] is True
        assert modified_result.metadata["original"] is True

    def test_error_result_no_data(self):
        """Test that error results always have data=None."""
        # Even if we try to pass data to error_result, it should be None
        result = FlowResult.error_result(
            error_message="Test error",
            metadata={"attempted_data": "should not be in data field"},
        )

        assert result.success is False
        assert result.data is None
        assert result.error_message == "Test error"
        assert result.metadata["attempted_data"] == "should not be in data field"
