"""
Unit tests for app.flows.base module.

Tests cover FlowResult class functionality, type safety,
and integration patterns.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock
import pandas as pd

from app.flows.base import FlowResult, FlowRunner, FlowResultEvent
from llama_index.core.workflow import Workflow


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


class TestFlowRunner:
    """Test FlowRunner class."""

    def test_flow_runner_initialization(self):
        """Test FlowRunner initialization with workflow."""
        mock_workflow = Mock()
        runner = FlowRunner[pd.DataFrame](mock_workflow)

        assert runner.workflow is mock_workflow
        assert runner._start_time is None

    @pytest.mark.asyncio
    async def test_flow_runner_success_with_dict_result(self):
        """Test FlowRunner with workflow returning dictionary result."""
        mock_workflow = Mock()

        # Mock workflow result as FlowResultEvent
        mock_result_event = FlowResultEvent.success_result(
            data=pd.DataFrame({"A": [1, 2, 3]}),
            successful_items=["AAPL"],
            failed_items=[],
        )
        mock_workflow.run = AsyncMock(return_value=mock_result_event)

        runner = FlowRunner[pd.DataFrame](mock_workflow)
        result = await runner.run(base_date=datetime(2024, 1, 1))

        # Verify FlowResultEvent structure
        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 3
        assert result.successful_items == ["AAPL"]
        assert result.failed_items == []
        assert result.metadata["execution_time"] is not None
        assert result.metadata["execution_time"] > 0
        assert result.metadata["workflow"] == "Mock"

        # Verify workflow was called correctly
        mock_workflow.run.assert_called_once_with(base_date=datetime(2024, 1, 1))

    @pytest.mark.asyncio
    async def test_flow_runner_success_with_flow_result(self):
        """Test FlowRunner with workflow already returning FlowResult."""
        mock_workflow = Mock()

        # Mock workflow result as FlowResultEvent already
        mock_result_event = FlowResultEvent.success_result(
            data=pd.DataFrame({"A": [1, 2, 3]}),
            successful_items=["MSFT"],
        )
        mock_workflow.run = AsyncMock(return_value=mock_result_event)

        runner = FlowRunner[pd.DataFrame](mock_workflow)
        result = await runner.run(symbol="MSFT")

        # Verify FlowResultEvent is preserved but execution time is updated
        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)
        assert result.successful_items == ["MSFT"]
        assert result.metadata["execution_time"] is not None
        assert result.metadata["execution_time"] > 0  # Should be updated by FlowRunner

    @pytest.mark.asyncio
    async def test_flow_runner_success_with_direct_data(self):
        """Test FlowRunner with workflow returning direct data."""
        mock_workflow = Mock()

        # Mock workflow result as FlowResultEvent with direct DataFrame
        test_data = pd.DataFrame({"B": [4, 5, 6]})
        mock_result_event = FlowResultEvent.success_result(data=test_data)
        mock_workflow.run = AsyncMock(return_value=mock_result_event)

        runner = FlowRunner[pd.DataFrame](mock_workflow)
        result = await runner.run(period="1Y")

        # Verify FlowResultEvent wraps direct data
        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)
        assert result.data.equals(test_data)
        assert result.metadata["execution_time"] is not None
        assert result.metadata["workflow"] == "Mock"

    @pytest.mark.asyncio
    async def test_flow_runner_error_from_dict_result(self):
        """Test FlowRunner with workflow returning error dictionary."""
        mock_workflow = Mock()

        # Mock workflow result as FlowResultEvent with error
        mock_result_event = FlowResultEvent.error_result(
            error_message="Test workflow error", failed_items=["INVALID"]
        )
        mock_workflow.run = AsyncMock(return_value=mock_result_event)

        runner = FlowRunner[pd.DataFrame](mock_workflow)
        result = await runner.run(symbols=["INVALID"])

        # Verify FlowResultEvent captures error
        assert isinstance(result, FlowResultEvent)
        assert result.success is False
        assert result.data is None
        assert result.error_message == "Test workflow error"
        assert result.failed_items == ["INVALID"]
        assert result.metadata["execution_time"] is not None
        assert result.metadata["workflow"] == "Mock"

    @pytest.mark.asyncio
    async def test_flow_runner_workflow_exception(self):
        """Test FlowRunner handling workflow execution exception."""
        mock_workflow = Mock()
        mock_workflow.run = AsyncMock(
            side_effect=ValueError("Workflow execution failed")
        )

        runner = FlowRunner[pd.DataFrame](mock_workflow)
        result = await runner.run(test_param="value")

        # Verify exception is captured in FlowResultEvent
        assert isinstance(result, FlowResultEvent)
        assert result.success is False
        assert result.data is None
        assert result.error_message is not None
        assert "Workflow execution failed" in result.error_message
        assert result.metadata["execution_time"] is not None
        assert result.metadata["workflow"] == "Mock"
        assert result.metadata["error_type"] == "ValueError"
        assert result.metadata["kwargs"] == {"test_param": "value"}

    @pytest.mark.asyncio
    async def test_flow_runner_no_result_attribute(self):
        """Test FlowRunner handling workflow result without .result attribute."""
        mock_workflow = Mock()

        # Mock workflow returning FlowResultEvent with data
        mock_result_event = FlowResultEvent.success_result(
            data=pd.DataFrame({"C": [7, 8, 9]})
        )
        mock_workflow.run = AsyncMock(return_value=mock_result_event)

        runner = FlowRunner[pd.DataFrame](mock_workflow)
        result = await runner.run()

        # Verify FlowResultEvent
        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 3
        assert result.metadata["workflow"] == "Mock"

    def test_validate_provider_result_success(self):
        """Test _validate_provider_result with successful provider result."""
        mock_workflow = Mock()
        runner = FlowRunner[pd.DataFrame](mock_workflow)

        # Create mock successful provider result
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = pd.DataFrame({"A": [1, 2, 3]})

        # Test validation
        validated_data = runner._validate_provider_result(mock_result, "test_data")

        assert isinstance(validated_data, pd.DataFrame)
        assert len(validated_data) == 3

    def test_validate_provider_result_exception(self):
        """Test _validate_provider_result with exception."""
        mock_workflow = Mock()
        runner = FlowRunner[pd.DataFrame](mock_workflow)

        test_exception = ValueError("Test provider error")

        # Test exception handling
        with pytest.raises(ValueError) as exc_info:
            runner._validate_provider_result(test_exception, "test_data")

        assert str(exc_info.value) == "Test provider error"

    def test_validate_provider_result_failure(self):
        """Test _validate_provider_result with failed provider result."""
        mock_workflow = Mock()
        runner = FlowRunner[pd.DataFrame](mock_workflow)

        # Create mock failed provider result
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Provider fetch failed"

        # Test failure handling
        with pytest.raises(Exception) as exc_info:
            runner._validate_provider_result(mock_result, "test_data")

        assert "test_data data fetch failed: Provider fetch failed" in str(
            exc_info.value
        )

    def test_validate_provider_result_empty_data(self):
        """Test _validate_provider_result with empty DataFrame."""
        mock_workflow = Mock()
        runner = FlowRunner[pd.DataFrame](mock_workflow)

        # Create mock provider result with empty data
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = pd.DataFrame()  # Empty DataFrame

        # Test empty data handling
        with pytest.raises(Exception) as exc_info:
            runner._validate_provider_result(mock_result, "test_data")

        assert "No test_data data available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_provider_tasks_success(self):
        """Test process_provider_tasks with successful tasks."""

        # Create mock tasks and results
        async def mock_task_1():
            result = Mock()
            result.success = True
            result.data = pd.DataFrame({"A": [1, 2]})
            result.execution_time = 0.5
            return result

        async def mock_task_2():
            result = Mock()
            result.success = True
            result.data = pd.DataFrame({"B": [3, 4]})
            result.execution_time = 0.7
            return result

        tasks = {"task1": mock_task_1(), "task2": mock_task_2()}

        # Test processing
        processed = await FlowRunner.process_provider_tasks(tasks)

        # Verify results
        assert len(processed) == 2
        assert processed["task1"]["success"] is True
        assert isinstance(processed["task1"]["data"], pd.DataFrame)
        assert processed["task1"]["execution_time"] == 0.5
        assert processed["task2"]["success"] is True
        assert processed["task2"]["execution_time"] == 0.7

    @pytest.mark.asyncio
    async def test_process_provider_tasks_with_failures(self):
        """Test process_provider_tasks with mixed success/failure."""

        # Create mock tasks with one success, one failure
        async def mock_success_task():
            result = Mock()
            result.success = True
            result.data = pd.DataFrame({"A": [1, 2]})
            return result

        async def mock_failure_task():
            raise ValueError("Task failed")

        tasks = {"success": mock_success_task(), "failure": mock_failure_task()}

        # Test processing
        processed = await FlowRunner.process_provider_tasks(tasks)

        # Verify mixed results
        assert len(processed) == 2
        assert processed["success"]["success"] is True
        assert processed["failure"]["success"] is False
        assert "Task failed" in processed["failure"]["error"]

    @pytest.mark.asyncio
    async def test_create_from_provider_results_success(self):
        """Test _create_from_provider_results with successful tasks."""
        mock_workflow = Mock()
        runner = FlowRunner[pd.DataFrame](mock_workflow)

        # Create mock successful tasks
        async def mock_task():
            result = Mock()
            result.success = True
            result.data = pd.DataFrame({"Value": [100, 200]})
            return result

        tasks = {"AAPL": mock_task()}
        base_date = datetime(2024, 1, 1)

        # Test creation
        flow_result = await runner._create_from_provider_results(tasks, base_date)

        # Verify FlowResult
        assert flow_result.success is True
        assert isinstance(flow_result.data, pd.DataFrame)
        assert len(flow_result.data) == 2
        assert flow_result.successful_items == ["AAPL"]
        assert flow_result.failed_items == []
        assert flow_result.base_date == base_date
        assert flow_result.execution_time is not None

    @pytest.mark.asyncio
    async def test_create_from_provider_results_all_failed(self):
        """Test _create_from_provider_results with all tasks failing."""
        mock_workflow = Mock()
        runner = FlowRunner[pd.DataFrame](mock_workflow)

        # Create mock failing task
        async def mock_failed_task():
            raise ValueError("All tasks failed")

        tasks = {"INVALID": mock_failed_task()}

        # Test creation
        flow_result = await runner._create_from_provider_results(tasks)

        # Verify error FlowResult
        assert flow_result.success is False
        assert flow_result.data is None
        assert flow_result.error_message is not None
        assert "Failed to process all items" in flow_result.error_message
        assert flow_result.failed_items == ["INVALID"]


class TestFlowRunnerIntegration:
    """Test FlowRunner integration patterns."""

    @pytest.mark.asyncio
    async def test_flow_runner_workflow_pattern_example(self):
        """Test FlowRunner with a workflow pattern similar to existing codebase."""

        # Create a mock workflow with custom class name for proper testing
        workflow = Mock(spec=Workflow)
        workflow.__class__.__name__ = "MockMarketWorkflow"
        workflow.run = AsyncMock(
            return_value=FlowResultEvent.success_result(
                data=pd.DataFrame(
                    {
                        "Market_Cap": [100, 120, 110],
                        "GDP": [50, 55, 60],
                        "Ratio": [2.0, 2.18, 1.83],
                    }
                ),
                base_date=datetime(2024, 1, 1),
                metadata={
                    "trend_data": {"slope": 0.1, "r_squared": 0.85},
                    "was_adjusted": False,
                    "original_period": "5Y",
                    "actual_period": "5Y",
                    "data_points": 3,
                },
            )
        )

        # Test the FlowRunner pattern
        runner = FlowRunner[pd.DataFrame](workflow)

        # Execute using FlowRunner
        result = await runner.run(base_date=datetime(2024, 1, 1), original_period="5Y")

        # Verify FlowResultEvent structure
        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 3
        assert list(result.data.columns) == ["Market_Cap", "GDP", "Ratio"]
        assert result.metadata["execution_time"] is not None
        assert result.base_date == datetime(2024, 1, 1)

        # Verify metadata includes workflow information
        assert result.metadata["workflow"] == "MockMarketWorkflow"
        assert result.metadata["trend_data"]["slope"] == 0.1
        assert result.metadata["was_adjusted"] is False
        assert result.metadata["data_points"] == 3

    def test_flow_runner_usage_demonstration(self):
        """Demonstrate how FlowRunner would be used in practice."""
        # This test shows the pattern but doesn't actually run workflows

        # Before FlowRunner (current pattern):
        # workflow = BuffetIndicatorWorkflow()
        # result = await workflow.run(
        #   base_date=base_date,
        #   original_period=original_period
        # )
        # buffet_data = result.get("data")
        # trend_data = result.get("trend_data")

        # After FlowRunner (new pattern):
        # runner = FlowRunner[pd.DataFrame](BuffetIndicatorWorkflow())
        # result = await runner.run(
        #   base_date=base_date,
        #   original_period=original_period
        # )
        # buffet_data = result.data  # Direct access to DataFrame
        # trend_data = result.metadata["original_result"]["trend_data"]  # Metadata

        # Verify the pattern is simple and clear
        mock_workflow = Mock()
        runner = FlowRunner[pd.DataFrame](mock_workflow)

        # The runner is ready to use
        assert runner.workflow is mock_workflow
        assert hasattr(runner, "run")
        assert hasattr(runner, "_validate_provider_result")
        assert hasattr(FlowRunner, "process_provider_tasks")
        assert hasattr(runner, "_create_from_provider_results")
