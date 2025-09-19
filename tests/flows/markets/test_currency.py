"""
Unit tests for Currency workflow and FlowRunner integration.
"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.markets.currency import (
    fetch_currency_data,
    CurrencyWorkflow,
)
from app.flows.base import FlowRunner, FlowResultEvent


@pytest.fixture
def sample_usdeur_data():
    """Create sample USD/EUR data."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    # Create realistic EUR=X values (USD/EUR rate, typically 0.8-1.2 range)
    usdeur_values = [0.85 + (i % 50) * 0.005 for i in range(100)]
    return pd.DataFrame(
        {
            "Open": [v - 0.01 for v in usdeur_values],
            "High": [v + 0.02 for v in usdeur_values],
            "Low": [v - 0.02 for v in usdeur_values],
            "Close": usdeur_values,
            "Adj Close": usdeur_values,
            "Volume": [1000000 + i * 1000 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def sample_gbpeur_data():
    """Create sample GBP/EUR data."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    # Create realistic GBPEUR=X values (GBP/EUR rate, typically 1.0-1.3 range)
    gbpeur_values = [1.1 + (i % 30) * 0.005 for i in range(100)]
    return pd.DataFrame(
        {
            "Open": [v - 0.01 for v in gbpeur_values],
            "High": [v + 0.02 for v in gbpeur_values],
            "Low": [v - 0.02 for v in gbpeur_values],
            "Close": gbpeur_values,
            "Adj Close": gbpeur_values,
            "Volume": [500000 + i * 500 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def mock_provider_result():
    """Create a mock provider result."""
    result = MagicMock()
    result.success = True
    result.error_message = None
    return result


class TestCurrencyWorkflow:
    """Test the CurrencyWorkflow class."""

    def test_workflow_initialization(self):
        """Test that workflow initializes correctly."""
        workflow = CurrencyWorkflow()
        assert workflow is not None
        assert hasattr(workflow, "yahoo_provider")

    @pytest.mark.asyncio
    async def test_initiate_currency_fetch(self):
        """Test the dispatch step that sends parallel events."""
        workflow = CurrencyWorkflow()

        # Create mock context
        ctx = MagicMock()
        ctx.store.set = AsyncMock()
        ctx.send_event = MagicMock()

        # Create start event
        start_event = MagicMock()
        start_event.base_date = datetime(2020, 1, 1)

        # Execute the step
        result = await workflow.initiate_currency_fetch(ctx, start_event)

        # Verify context operations
        ctx.store.set.assert_any_call("base_date", start_event.base_date)
        assert ctx.send_event.call_count == 2  # USD/EUR and GBP/EUR events

        # Verify return event
        assert result is not None

    @pytest.mark.asyncio
    async def test_fetch_usdeur_data_success(self):
        """Test successful USD/EUR data fetching step."""
        workflow = CurrencyWorkflow()

        # Create mock USD/EUR data
        usdeur_data = pd.DataFrame(
            {"Close": [0.85, 0.86, 0.87]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        # Mock provider result
        usdeur_result = MagicMock()
        usdeur_result.success = True
        usdeur_result.data = usdeur_data

        # Mock the provider method
        workflow.yahoo_provider.get_data = AsyncMock(return_value=usdeur_result)

        # Create fetch event
        fetch_event = MagicMock()
        fetch_event.base_date = datetime(2020, 1, 1)

        # Execute the step
        result = await workflow.fetch_usdeur_data(fetch_event)

        # Verify result
        assert result.data is not None
        assert not result.data.empty
        assert result.error is None

    @pytest.mark.asyncio
    async def test_fetch_gbpeur_data_success(self):
        """Test successful GBP/EUR data fetching step."""
        workflow = CurrencyWorkflow()

        # Create mock GBP/EUR data
        gbpeur_data = pd.DataFrame(
            {"Close": [1.15, 1.16, 1.17]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        # Mock provider result
        gbpeur_result = MagicMock()
        gbpeur_result.success = True
        gbpeur_result.data = gbpeur_data

        # Mock the provider method
        workflow.yahoo_provider.get_data = AsyncMock(return_value=gbpeur_result)

        # Create fetch event
        fetch_event = MagicMock()
        fetch_event.base_date = datetime(2020, 1, 1)

        # Execute the step
        result = await workflow.fetch_gbpeur_data(fetch_event)

        # Verify result
        assert result.data is not None
        assert not result.data.empty
        assert result.error is None

    @pytest.mark.asyncio
    async def test_fetch_usdeur_data_provider_failure(self):
        """Test handling of USD/EUR provider failure."""
        workflow = CurrencyWorkflow()

        # Mock provider result with failure
        usdeur_result = MagicMock()
        usdeur_result.success = False
        usdeur_result.error_message = "USD/EUR provider failed"

        # Mock the provider method
        workflow.yahoo_provider.get_data = AsyncMock(return_value=usdeur_result)

        # Create fetch event
        fetch_event = MagicMock()
        fetch_event.base_date = datetime(2020, 1, 1)

        # Execute the step
        result = await workflow.fetch_usdeur_data(fetch_event)

        # Verify error handling
        assert result.data is None
        assert result.error is not None
        assert "USD/EUR provider failed" in result.error


class TestCurrencyFlowRunner:
    """Test the Currency workflow with FlowRunner."""

    @pytest.mark.asyncio
    async def test_flowrunner_integration_success(self):
        """Test successful workflow execution with FlowRunner."""
        workflow = CurrencyWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Mock the workflow to return a FlowResultEvent
        mock_result = FlowResultEvent.success_result(
            data=pd.DataFrame(
                {
                    "USD_EUR": [0.85, 0.86],
                    "GBP_EUR": [1.15, 1.16],
                }
            ),
            metadata={
                "latest_usdeur": 0.86,
                "latest_gbpeur": 1.16,
                "data_points": 2,
            },
        )

        # Mock workflow.run to return the mock result
        workflow.run = AsyncMock(return_value=mock_result)

        # Execute workflow through FlowRunner
        result = await runner.run(base_date=datetime(2020, 1, 1))

        # Verify FlowResultEvent structure
        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert result.data is not None
        assert not result.data.empty
        assert "USD_EUR" in result.data.columns
        assert "GBP_EUR" in result.data.columns

    @pytest.mark.asyncio
    async def test_flowrunner_integration_workflow_exception(self):
        """Test FlowRunner handling of Exception."""
        workflow = CurrencyWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Mock workflow to raise Exception
        workflow.run = AsyncMock(side_effect=Exception("Test currency workflow error"))

        # Execute and expect FlowResultEvent with error
        result = await runner.run(base_date=datetime(2020, 1, 1))

        assert isinstance(result, FlowResultEvent)
        assert result.success is False
        assert result.data is None
        assert (
            result.error_message is not None
            and "Test currency workflow error" in result.error_message
        )


class TestFetchCurrencyData:
    """Test the fetch_currency_data function."""

    @pytest.mark.asyncio
    async def test_fetch_currency_data_success(
        self, sample_usdeur_data, sample_gbpeur_data
    ):
        """Test successful currency data fetch and calculation."""
        # Setup mock provider results
        usdeur_result = MagicMock()
        usdeur_result.success = True
        usdeur_result.data = sample_usdeur_data

        gbpeur_result = MagicMock()
        gbpeur_result.success = True
        gbpeur_result.data = sample_gbpeur_data

        # Mock the provider
        with patch(
            "app.flows.markets.currency.create_yahoo_history_provider"
        ) as mock_yahoo:
            # Setup provider mock
            mock_yahoo_instance = AsyncMock()

            # Mock different calls for USD/EUR and GBP/EUR
            def mock_get_data(query):
                if query == "EUR=X":
                    return usdeur_result
                elif query == "GBPEUR=X":
                    return gbpeur_result
                else:
                    raise ValueError(f"Unexpected query: {query}")

            mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the function
            base_date = datetime(2020, 1, 1)
            result = await fetch_currency_data(base_date)

            # Verify results
            assert "data" in result
            assert "base_date" in result
            assert "latest_usdeur" in result
            assert "latest_gbpeur" in result
            assert "data_points" in result

            data = result["data"]
            assert not data.empty
            assert "USD_EUR" in data.columns
            assert "GBP_EUR" in data.columns

            # Verify exchange rates are within reasonable ranges
            latest_usdeur = result["latest_usdeur"]
            latest_gbpeur = result["latest_gbpeur"]
            assert isinstance(latest_usdeur, (float, int)) or hasattr(
                latest_usdeur, "dtype"
            )
            assert isinstance(latest_gbpeur, (float, int)) or hasattr(
                latest_gbpeur, "dtype"
            )
            assert 0.5 <= latest_usdeur <= 1.5  # Reasonable USD/EUR range
            assert 0.8 <= latest_gbpeur <= 1.5  # Reasonable GBP/EUR range

    @pytest.mark.asyncio
    async def test_fetch_currency_data_usdeur_error(self):
        """Test handling of USD/EUR provider errors."""
        # Test the error handling within the workflow
        workflow = CurrencyWorkflow()

        # Mock the Yahoo provider to fail for USD/EUR
        with patch.object(workflow, "yahoo_provider") as mock_yahoo:
            mock_yahoo.get_data = AsyncMock(
                side_effect=Exception("Yahoo USD/EUR API error")
            )

            # Test the workflow directly
            try:
                await workflow.run(base_date=datetime(2020, 1, 1))
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "Yahoo USD/EUR API error" in str(e)

    @pytest.mark.asyncio
    async def test_currency_data_empty_handling(self):
        """Test handling of empty currency data."""
        # Setup mock provider result with empty data
        usdeur_result = MagicMock()
        usdeur_result.success = True
        usdeur_result.data = pd.DataFrame()  # Empty DataFrame

        gbpeur_result = MagicMock()
        gbpeur_result.success = True
        gbpeur_result.data = pd.DataFrame()  # Empty DataFrame

        # Mock the provider
        with patch(
            "app.flows.markets.currency.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()

            def mock_get_data(query):
                if query == "EUR=X":
                    return usdeur_result
                elif query == "GBPEUR=X":
                    return gbpeur_result
                else:
                    raise ValueError(f"Unexpected query: {query}")

            mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the workflow directly
            workflow = CurrencyWorkflow()
            try:
                await workflow.run(base_date=datetime(2020, 1, 1))
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "USD/EUR data fetch failed" in str(e)

    @pytest.mark.asyncio
    async def test_currency_provider_failure(self):
        """Test handling of currency provider failure."""
        # Setup mock provider result with failure
        usdeur_result = MagicMock()
        usdeur_result.success = False
        usdeur_result.error_message = "USD/EUR provider failed"

        # Mock the provider
        with patch(
            "app.flows.markets.currency.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=usdeur_result)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the workflow directly
            workflow = CurrencyWorkflow()
            try:
                await workflow.run(base_date=datetime(2020, 1, 1))
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "USD/EUR data fetch failed" in str(e)

    @pytest.mark.asyncio
    async def test_currency_data_filtering_by_base_date(
        self, sample_usdeur_data, sample_gbpeur_data
    ):
        """Test that currency data is properly filtered by base_date."""
        # Setup mock provider results
        usdeur_result = MagicMock()
        usdeur_result.success = True
        usdeur_result.data = sample_usdeur_data

        gbpeur_result = MagicMock()
        gbpeur_result.success = True
        gbpeur_result.data = sample_gbpeur_data

        # Mock the provider
        with patch(
            "app.flows.markets.currency.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()

            def mock_get_data(query):
                if query == "EUR=X":
                    return usdeur_result
                elif query == "GBPEUR=X":
                    return gbpeur_result
                else:
                    raise ValueError(f"Unexpected query: {query}")

            mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test with different base dates
            base_date = datetime(2020, 2, 15)  # Mid-period
            result = await fetch_currency_data(base_date)

            data = result["data"]
            assert not data.empty

            # All data should be from base_date onwards
            assert data.index.min() >= pd.to_datetime(base_date.date())

            # Should be less than full dataset
            assert len(data) < 100  # Less than the full 100 days

    @pytest.mark.asyncio
    async def test_currency_workflow_direct(self):
        """Test the CurrencyWorkflow class directly."""
        # Create sample currency data
        usdeur_data = pd.DataFrame(
            {
                "Close": [0.85, 0.86, 0.87, 0.88, 0.89],
                "Volume": [1000000] * 5,
            },
            index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
        )

        gbpeur_data = pd.DataFrame(
            {
                "Close": [1.15, 1.16, 1.17, 1.18, 1.19],
                "Volume": [500000] * 5,
            },
            index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
        )

        # Mock provider results
        usdeur_result = MagicMock()
        usdeur_result.success = True
        usdeur_result.data = usdeur_data

        gbpeur_result = MagicMock()
        gbpeur_result.success = True
        gbpeur_result.data = gbpeur_data

        # Mock the provider at the class level
        with patch(
            "app.flows.markets.currency.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()

            def mock_get_data(query):
                if query == "EUR=X":
                    return usdeur_result
                elif query == "GBPEUR=X":
                    return gbpeur_result
                else:
                    raise ValueError(f"Unexpected query: {query}")

            mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the workflow directly
            workflow = CurrencyWorkflow()
            base_date = datetime(2020, 1, 1)

            result = await workflow.run(base_date=base_date)

            # Verify results
            assert result.data is not None
            assert result.metadata is not None
            assert "latest_usdeur" in result.metadata
            assert "latest_gbpeur" in result.metadata

            data = result.data
            assert not data.empty
            assert len(data) == 5  # Should have 5 days of data
            assert "USD_EUR" in data.columns
            assert "GBP_EUR" in data.columns

            # Verify latest rates
            latest_usdeur = result.metadata["latest_usdeur"]
            latest_gbpeur = result.metadata["latest_gbpeur"]
            assert latest_usdeur == 0.89  # Last value in USD/EUR data
            assert latest_gbpeur == 1.19  # Last value in GBP/EUR data

    @pytest.mark.asyncio
    async def test_currency_no_close_price_data(self):
        """Test handling when currency data has no Close price column."""
        # Create currency data without Close price
        usdeur_data = pd.DataFrame(
            {
                "Open": [0.85, 0.86, 0.87],
                "High": [0.86, 0.87, 0.88],
                "Low": [0.84, 0.85, 0.86],
                "Volume": [1000000] * 3,
            },
            index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
        )

        # Mock provider result
        usdeur_result = MagicMock()
        usdeur_result.success = True
        usdeur_result.data = usdeur_data

        # Mock the provider
        with patch(
            "app.flows.markets.currency.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=usdeur_result)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the workflow directly
            workflow = CurrencyWorkflow()
            try:
                await workflow.run(base_date=datetime(2020, 1, 1))
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "No Close price data in USD/EUR data" in str(e)
