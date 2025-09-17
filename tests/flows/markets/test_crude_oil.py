"""
Unit tests for Crude Oil workflow and FlowRunner integration.
"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.markets.crude_oil import (
    fetch_crude_oil_data,
    CrudeOilWorkflow,
)


@pytest.fixture
def sample_wti_data():
    """Create sample WTI crude oil data."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    wti_values = [60 + (i % 20) * 2 + (i // 50) * 5 for i in range(100)]
    return pd.DataFrame(
        {
            "Open": [v - 1 for v in wti_values],
            "High": [v + 2 for v in wti_values],
            "Low": [v - 2 for v in wti_values],
            "Close": wti_values,
            "Adj Close": wti_values,
            "Volume": [100000 + i * 1000 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def sample_brent_data():
    """Create sample Brent crude oil data."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    brent_values = [65 + (i % 20) * 2 + (i // 50) * 5 for i in range(100)]
    return pd.DataFrame(
        {
            "Open": [v - 1 for v in brent_values],
            "High": [v + 2 for v in brent_values],
            "Low": [v - 2 for v in brent_values],
            "Close": brent_values,
            "Adj Close": brent_values,
            "Volume": [80000 + i * 800 for i in range(100)],
        },
        index=dates,
    )


class TestCrudeOilWorkflow:
    """Test the CrudeOilWorkflow class."""

    def test_workflow_initialization(self):
        """Test that workflow initializes correctly."""
        workflow = CrudeOilWorkflow()
        assert workflow is not None
        assert hasattr(workflow, "yahoo_provider")

    @pytest.mark.asyncio
    async def test_initiate_crude_oil_fetch(self):
        """Test the dispatch step that sends parallel events."""
        workflow = CrudeOilWorkflow()

        ctx = MagicMock()
        ctx.store.set = AsyncMock()
        ctx.send_event = MagicMock()

        start_event = MagicMock()
        start_event.base_date = datetime(2020, 1, 1)

        result = await workflow.initiate_crude_oil_fetch(ctx, start_event)

        ctx.store.set.assert_any_call("base_date", start_event.base_date)
        assert ctx.send_event.call_count == 2  # WTI and Brent events
        assert result is not None


class TestFetchCrudeOilData:
    """Test the fetch_crude_oil_data function."""

    @pytest.mark.asyncio
    async def test_fetch_crude_oil_data_success(
        self, sample_wti_data, sample_brent_data
    ):
        """Test successful crude oil data fetch and calculation."""
        wti_result = MagicMock()
        wti_result.success = True
        wti_result.data = sample_wti_data

        brent_result = MagicMock()
        brent_result.success = True
        brent_result.data = sample_brent_data

        with patch(
            "app.flows.markets.crude_oil.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()

            def mock_get_data(query):
                if query == "CL=F":
                    return wti_result
                elif query == "BZ=F":
                    return brent_result
                else:
                    raise ValueError(f"Unexpected query: {query}")

            mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
            mock_yahoo.return_value = mock_yahoo_instance

            base_date = datetime(2020, 1, 1)
            result = await fetch_crude_oil_data(base_date)

            assert "data" in result
            assert "base_date" in result
            assert "latest_wti" in result
            assert "latest_brent" in result
            assert "data_points" in result

            data = result["data"]
            assert not data.empty
            assert "WTI" in data.columns
            assert "Brent" in data.columns

            latest_wti = result["latest_wti"]
            latest_brent = result["latest_brent"]
            assert isinstance(latest_wti, (float, int)) or hasattr(latest_wti, 'dtype')
            assert isinstance(latest_brent, (float, int)) or hasattr(latest_brent, 'dtype')
            assert 20 <= latest_wti <= 150  # Reasonable WTI range
            assert 20 <= latest_brent <= 150  # Reasonable Brent range

    @pytest.mark.asyncio
    async def test_crude_oil_workflow_direct(self):
        """Test the CrudeOilWorkflow class directly."""
        wti_data = pd.DataFrame(
            {
                "Close": [60.0, 62.0, 64.0, 66.0, 68.0],
                "Volume": [100000] * 5,
            },
            index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
        )

        brent_data = pd.DataFrame(
            {
                "Close": [65.0, 67.0, 69.0, 71.0, 73.0],
                "Volume": [80000] * 5,
            },
            index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
        )

        wti_result = MagicMock()
        wti_result.success = True
        wti_result.data = wti_data

        brent_result = MagicMock()
        brent_result.success = True
        brent_result.data = brent_data

        with patch(
            "app.flows.markets.crude_oil.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()

            def mock_get_data(query):
                if query == "CL=F":
                    return wti_result
                elif query == "BZ=F":
                    return brent_result
                else:
                    raise ValueError(f"Unexpected query: {query}")

            mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
            mock_yahoo.return_value = mock_yahoo_instance

            workflow = CrudeOilWorkflow()
            base_date = datetime(2020, 1, 1)

            result = await workflow.run(base_date=base_date)

            assert result.data is not None
            assert result.metadata is not None
            assert "latest_wti" in result.metadata
            assert "latest_brent" in result.metadata

            data = result.data
            assert not data.empty
            assert len(data) == 5
            assert "WTI" in data.columns
            assert "Brent" in data.columns

            latest_wti = result.metadata["latest_wti"]
            latest_brent = result.metadata["latest_brent"]
            assert latest_wti == 68.0
            assert latest_brent == 73.0
