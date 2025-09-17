"""
Unit tests for Bloomberg Commodity workflow and FlowRunner integration.
"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.markets.bloomberg_commodity import (
    fetch_bloomberg_commodity_data,
    BloombergCommodityWorkflow,
)
from app.flows.base import FlowResultEvent


@pytest.fixture
def sample_bcom_data():
    """Create sample Bloomberg Commodity Index data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    bcom_values = [80 + (i % 40) * 2 + (i // 100) * 3 for i in range(365)]
    return pd.DataFrame(
        {
            "Open": [v - 2 for v in bcom_values],
            "High": [v + 3 for v in bcom_values],
            "Low": [v - 3 for v in bcom_values],
            "Close": bcom_values,
            "Adj Close": bcom_values,
            "Volume": [50000 + i * 100 for i in range(365)],
        },
        index=dates,
    )


class TestBloombergCommodityWorkflow:
    """Test the BloombergCommodityWorkflow class."""

    def test_workflow_initialization(self):
        """Test that workflow initializes correctly."""
        workflow = BloombergCommodityWorkflow()
        assert workflow is not None
        assert hasattr(workflow, "yahoo_provider")

    @pytest.mark.asyncio
    async def test_fetch_and_process_bcom_data_success(self):
        """Test successful BCOM data fetching and processing step."""
        workflow = BloombergCommodityWorkflow()

        bcom_data = pd.DataFrame(
            {"Close": [80.0, 82.0, 84.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        bcom_result = MagicMock()
        bcom_result.success = True
        bcom_result.data = bcom_data

        workflow.yahoo_provider.get_data = AsyncMock(return_value=bcom_result)

        start_event = MagicMock()
        start_event.base_date = datetime(2020, 1, 1)

        result = await workflow.fetch_and_process_bcom_data(start_event)

        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert result.data is not None
        assert not result.data.empty
        assert "BCOM" in result.data.columns
        assert "BCOM_MA50" in result.data.columns
        assert "BCOM_MA200" in result.data.columns


class TestFetchBloombergCommodityData:
    """Test the fetch_bloomberg_commodity_data function."""

    @pytest.mark.asyncio
    async def test_fetch_bloomberg_commodity_data_success(self, sample_bcom_data):
        """Test successful Bloomberg Commodity data fetch and calculation."""
        bcom_result = MagicMock()
        bcom_result.success = True
        bcom_result.data = sample_bcom_data

        with patch(
            "app.flows.markets.bloomberg_commodity.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
            mock_yahoo.return_value = mock_yahoo_instance

            base_date = datetime(2020, 1, 1)
            result = await fetch_bloomberg_commodity_data(base_date)

            assert "data" in result
            assert "base_date" in result
            assert "latest_value" in result
            assert "data_points" in result

            data = result["data"]
            assert not data.empty
            assert "BCOM" in data.columns
            assert "BCOM_MA50" in data.columns
            assert "BCOM_MA200" in data.columns

            latest_value = result["latest_value"]
            assert isinstance(latest_value, (float, int)) or hasattr(latest_value, 'dtype')
            assert 50 <= latest_value <= 200  # Reasonable BCOM range

            mock_yahoo_instance.get_data.assert_called_once_with("^BCOM")

    @pytest.mark.asyncio
    async def test_bloomberg_commodity_moving_averages_calculation(
        self, sample_bcom_data
    ):
        """Test the accuracy of BCOM moving averages calculation."""
        bcom_result = MagicMock()
        bcom_result.success = True
        bcom_result.data = sample_bcom_data

        with patch(
            "app.flows.markets.bloomberg_commodity.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
            mock_yahoo.return_value = mock_yahoo_instance

            base_date = datetime(2020, 1, 1)
            result = await fetch_bloomberg_commodity_data(base_date)

            data = result["data"]
            assert "BCOM_MA50" in data.columns
            assert "BCOM_MA200" in data.columns

            # Verify moving average calculation accuracy for a full year of data
            ma50_values = data["BCOM_MA50"]
            ma200_values = data["BCOM_MA200"]

            # Check that we have reasonable MA values
            ma50_non_null = ma50_values.dropna()
            ma200_non_null = ma200_values.dropna()

            if len(data) >= 50:
                assert not ma50_non_null.empty
                # MA50 should be calculated after 50 periods
                assert len(ma50_non_null) == len(data) - 49

            if len(data) >= 200:
                assert not ma200_non_null.empty
                # MA200 should be calculated after 200 periods
                assert len(ma200_non_null) == len(data) - 199

    @pytest.mark.asyncio
    async def test_bloomberg_commodity_workflow_direct(self):
        """Test the BloombergCommodityWorkflow class directly."""
        bcom_data = pd.DataFrame(
            {
                "Close": [80.0, 82.0, 84.0, 86.0, 88.0],
                "Volume": [50000] * 5,
            },
            index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
        )

        bcom_result = MagicMock()
        bcom_result.success = True
        bcom_result.data = bcom_data

        with patch(
            "app.flows.markets.bloomberg_commodity.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
            mock_yahoo.return_value = mock_yahoo_instance

            workflow = BloombergCommodityWorkflow()
            base_date = datetime(2020, 1, 1)

            result = await workflow.run(base_date=base_date)

            assert result.data is not None
            assert result.metadata is not None
            assert "latest_value" in result.metadata

            data = result.data
            assert not data.empty
            assert len(data) == 5
            assert "BCOM" in data.columns

            latest_value = result.metadata["latest_value"]
            assert latest_value == 88.0
