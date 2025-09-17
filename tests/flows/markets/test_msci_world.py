"""
Unit tests for MSCI World workflow and FlowRunner integration.
"""

import pandas as pd
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.markets.msci_world import (
    fetch_msci_world_data,
    MSCIWorldWorkflow,
)
from app.flows.base import FlowRunner, FlowResultEvent


@pytest.fixture
def sample_msci_data():
    """Create sample MSCI World Index data."""
    dates = pd.date_range(start="2020-01-01", periods=252, freq="D")
    msci_values = [2500 + (i % 30) * 10 + (i // 100) * 50 for i in range(252)]
    return pd.DataFrame(
        {
            "Open": [v - 5 for v in msci_values],
            "High": [v + 15 for v in msci_values],
            "Low": [v - 15 for v in msci_values],
            "Close": msci_values,
            "Adj Close": msci_values,
            "Volume": [100000 + i * 500 for i in range(252)],
        },
        index=dates,
    )


class TestMSCIWorldWorkflow:
    """Test the MSCIWorldWorkflow class."""

    def test_workflow_initialization(self):
        """Test that workflow initializes correctly."""
        workflow = MSCIWorldWorkflow()
        assert workflow is not None
        assert hasattr(workflow, "yahoo_provider")

    @pytest.mark.asyncio
    async def test_fetch_and_process_msci_data_success(self):
        """Test successful MSCI data fetching and processing step."""
        workflow = MSCIWorldWorkflow()

        msci_data = pd.DataFrame(
            {"Close": [2500.0, 2520.0, 2540.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        msci_result = MagicMock()
        msci_result.success = True
        msci_result.data = msci_data

        workflow.yahoo_provider.get_data = AsyncMock(return_value=msci_result)

        start_event = MagicMock()
        start_event.base_date = datetime(2020, 1, 1)

        result = await workflow.fetch_and_process_msci_data(start_event)

        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert result.data is not None
        assert not result.data.empty
        assert "MSCI_World" in result.data.columns


class TestFetchMSCIWorldData:
    """Test the fetch_msci_world_data function."""

    @pytest.mark.asyncio
    async def test_fetch_msci_world_data_success(self, sample_msci_data):
        """Test successful MSCI World data fetch and calculation."""
        msci_result = MagicMock()
        msci_result.success = True
        msci_result.data = sample_msci_data

        with patch(
            "app.flows.markets.msci_world.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=msci_result)
            mock_yahoo.return_value = mock_yahoo_instance

            base_date = datetime(2020, 1, 1)
            result = await fetch_msci_world_data(base_date)

            assert "data" in result
            assert "base_date" in result
            assert "latest_value" in result
            assert "data_points" in result

            data = result["data"]
            assert not data.empty
            assert "MSCI_World" in data.columns

            latest_value = result["latest_value"]
            assert isinstance(latest_value, (float, int, np.integer))
            assert 2000 <= latest_value <= 4000  # Reasonable MSCI range

            mock_yahoo_instance.get_data.assert_called_once_with("^990100-USD-STRD")

    @pytest.mark.asyncio
    async def test_msci_world_data_filtering_by_base_date(self, sample_msci_data):
        """Test that MSCI data is properly filtered by base_date."""
        msci_result = MagicMock()
        msci_result.success = True
        msci_result.data = sample_msci_data

        with patch(
            "app.flows.markets.msci_world.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=msci_result)
            mock_yahoo.return_value = mock_yahoo_instance

            base_date = datetime(2020, 6, 15)  # Mid-period
            result = await fetch_msci_world_data(base_date)

            data = result["data"]
            assert not data.empty

            # All data should be from base_date onwards
            assert data.index.min() >= pd.to_datetime(base_date.date())

            # Should be less than full dataset
            assert len(data) < 252

    @pytest.mark.asyncio
    async def test_msci_world_workflow_direct(self):
        """Test the MSCIWorldWorkflow class directly."""
        msci_data = pd.DataFrame(
            {
                "Close": [2500.0, 2520.0, 2540.0, 2560.0, 2580.0],
                "Volume": [100000] * 5,
            },
            index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
        )

        msci_result = MagicMock()
        msci_result.success = True
        msci_result.data = msci_data

        with patch(
            "app.flows.markets.msci_world.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=msci_result)
            mock_yahoo.return_value = mock_yahoo_instance

            workflow = MSCIWorldWorkflow()
            base_date = datetime(2020, 1, 1)

            result = await workflow.run(base_date=base_date)

            assert result.data is not None
            assert result.metadata is not None
            assert "latest_value" in result.metadata

            data = result.data
            assert not data.empty
            assert len(data) == 5
            assert "MSCI_World" in data.columns

            latest_value = result.metadata["latest_value"]
            assert latest_value == 2580.0


class TestMSCIWorldFlowRunner:
    """Test the MSCI World workflow with FlowRunner."""

    @pytest.mark.asyncio
    async def test_flowrunner_integration_success(self):
        """Test successful workflow execution with FlowRunner."""
        workflow = MSCIWorldWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        mock_result = FlowResultEvent.success_result(
            data=pd.DataFrame(
                {"MSCI_World": [2500.0, 2520.0, 2540.0]},
                index=pd.date_range("2020-01-01", periods=3, freq="D"),
            ),
            metadata={
                "latest_value": 2540.0,
                "data_points": 3,
            },
        )

        workflow.run = AsyncMock(return_value=mock_result)

        result = await runner.run(base_date=datetime(2020, 1, 1))

        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert result.data is not None
        assert "MSCI_World" in result.data.columns

    @pytest.mark.asyncio
    async def test_msci_provider_failure(self):
        """Test handling of MSCI provider failure."""
        msci_result = MagicMock()
        msci_result.success = False
        msci_result.error_message = "MSCI provider failed"

        with patch(
            "app.flows.markets.msci_world.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=msci_result)
            mock_yahoo.return_value = mock_yahoo_instance

            workflow = MSCIWorldWorkflow()
            try:
                await workflow.run(base_date=datetime(2020, 1, 1))
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "MSCI World data fetch failed" in str(e)

    @pytest.mark.asyncio
    async def test_msci_no_close_price_data(self):
        """Test handling when MSCI data has no Close price column."""
        msci_data = pd.DataFrame(
            {
                "Open": [2500.0, 2520.0, 2540.0],
                "High": [2550.0, 2570.0, 2590.0],
                "Low": [2480.0, 2500.0, 2520.0],
                "Volume": [100000] * 3,
            },
            index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
        )

        msci_result = MagicMock()
        msci_result.success = True
        msci_result.data = msci_data

        with patch(
            "app.flows.markets.msci_world.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=msci_result)
            mock_yahoo.return_value = mock_yahoo_instance

            workflow = MSCIWorldWorkflow()
            try:
                await workflow.run(base_date=datetime(2020, 1, 1))
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "No Close price data available for MSCI World" in str(e)
