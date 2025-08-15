"""
Unit tests for compare workflow functions.

Tests the new workflow functions in app/flows/compare.py with mocked data
to avoid external API dependencies.
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.compare import (
    fetch_raw_ticker_data,
    fetch_returns_data,
    fetch_volatility_data,
    fetch_volume_data,
    fetch_rsi_data,
    CompareDataWorkflow,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "Open": [100 + i for i in range(100)],
        "High": [105 + i for i in range(100)],
        "Low": [95 + i for i in range(100)],
        "Close": [102 + i for i in range(100)],
        "Adj Close": [102 + i for i in range(100)],
        "Volume": [1000000 + i * 1000 for i in range(100)],
    }, index=dates)


@pytest.fixture
def mock_provider_result():
    """Create a mock provider result."""
    result = MagicMock()
    result.success = True
    return result


class TestFetchRawTickerData:
    """Test fetch_raw_ticker_data function."""

    @pytest.mark.asyncio
    async def test_fetch_raw_ticker_data_success(self, sample_ohlcv_data, mock_provider_result):
        """Test successful data fetching."""
        mock_provider_result.data = sample_ohlcv_data
        
        with patch("app.flows.compare.create_yahoo_history_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.get_data.return_value = mock_provider_result
            mock_create.return_value = mock_provider
            
            # Test with cache disabled to avoid aiocache in tests
            with patch("app.flows.cache.settings") as mock_settings:
                mock_settings.FLOW_CACHE_ENABLED = False
                
                result = await fetch_raw_ticker_data(
                    ["AAPL", "MSFT"], 
                    datetime(2024, 1, 1)
                )
                
                assert "AAPL" in result
                assert "MSFT" in result
                assert isinstance(result["AAPL"], pd.DataFrame)
                assert len(result["AAPL"]) == 100

    @pytest.mark.asyncio
    async def test_fetch_raw_ticker_data_empty_tickers(self):
        """Test with empty ticker list."""
        result = await fetch_raw_ticker_data([], datetime(2024, 1, 1))
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_raw_ticker_data_provider_failure(self):
        """Test handling of provider failures."""
        with patch("app.flows.compare.create_yahoo_history_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.get_data.side_effect = Exception("API Error")
            mock_create.return_value = mock_provider
            
            with patch("app.flows.cache.settings") as mock_settings:
                mock_settings.FLOW_CACHE_ENABLED = False
                
                result = await fetch_raw_ticker_data(
                    ["INVALID"], 
                    datetime(2024, 1, 1)
                )
                
                assert result == {}


class TestFetchReturnsData:
    """Test fetch_returns_data function."""

    @pytest.mark.asyncio
    async def test_fetch_returns_data_success(self, sample_ohlcv_data):
        """Test successful returns calculation."""
        mock_raw_data = {"AAPL": sample_ohlcv_data}
        
        with patch("app.flows.compare.fetch_raw_ticker_data") as mock_fetch:
            mock_fetch.return_value = mock_raw_data
            
            result = await fetch_returns_data(
                ["AAPL"], 
                datetime(2024, 1, 1)
            )
            
            assert result["successful_tickers"] == ["AAPL"]
            assert len(result["failed_tickers"]) == 0
            assert isinstance(result["data"], pd.DataFrame)
            assert "AAPL" in result["data"].columns

    @pytest.mark.asyncio
    async def test_fetch_returns_data_empty_tickers(self):
        """Test with empty ticker list."""
        result = await fetch_returns_data([], datetime(2024, 1, 1))
        
        assert result["successful_tickers"] == []
        assert result["failed_tickers"] == []
        assert isinstance(result["data"], pd.DataFrame)
        assert result["data"].empty

    @pytest.mark.asyncio
    async def test_fetch_returns_data_no_raw_data(self):
        """Test when no raw data is available."""
        with patch("app.flows.compare.fetch_raw_ticker_data") as mock_fetch:
            mock_fetch.return_value = {}
            
            result = await fetch_returns_data(
                ["AAPL"], 
                datetime(2024, 1, 1)
            )
            
            assert result["successful_tickers"] == []
            assert result["failed_tickers"] == ["AAPL"]

    @pytest.mark.asyncio
    async def test_fetch_returns_data_missing_close_column(self):
        """Test handling of data without Close column."""
        # Create data without Close column
        bad_data = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Volume": [1000, 1100, 1200],
        }, index=pd.date_range("2024-01-01", periods=3, freq="D"))
        
        mock_raw_data = {"AAPL": bad_data}
        
        with patch("app.flows.compare.fetch_raw_ticker_data") as mock_fetch:
            mock_fetch.return_value = mock_raw_data
            
            result = await fetch_returns_data(
                ["AAPL"], 
                datetime(2024, 1, 1)
            )
            
            assert result["successful_tickers"] == []
            assert result["failed_tickers"] == ["AAPL"]


class TestFetchVolatilityData:
    """Test fetch_volatility_data function."""

    @pytest.mark.asyncio
    async def test_fetch_volatility_data_success(self, sample_ohlcv_data):
        """Test successful volatility calculation."""
        mock_raw_data = {"AAPL": sample_ohlcv_data}
        
        with patch("app.flows.compare.fetch_raw_ticker_data") as mock_fetch:
            mock_fetch.return_value = mock_raw_data
            
            # Mock the calculate_volatility function
            mock_vol_data = pd.DataFrame({
                "Volatility": [0.2, 0.3, 0.25]
            }, index=pd.date_range("2024-01-01", periods=3, freq="D"))
            
            with patch("app.flows.compare.calculate_volatility") as mock_calc:
                mock_calc.return_value = mock_vol_data
                
                result = await fetch_volatility_data(
                    ["AAPL"], 
                    datetime(2024, 1, 1)
                )
                
                assert result["successful_tickers"] == ["AAPL"]
                assert len(result["failed_tickers"]) == 0
                assert isinstance(result["data"], pd.DataFrame)
                assert "AAPL" in result["data"].columns

    @pytest.mark.asyncio
    async def test_fetch_volatility_data_calculation_failure(self, sample_ohlcv_data):
        """Test handling of volatility calculation failure."""
        mock_raw_data = {"AAPL": sample_ohlcv_data}
        
        with patch("app.flows.compare.fetch_raw_ticker_data") as mock_fetch:
            mock_fetch.return_value = mock_raw_data
            
            with patch("app.flows.compare.calculate_volatility") as mock_calc:
                mock_calc.return_value = pd.DataFrame()  # Empty result
                
                result = await fetch_volatility_data(
                    ["AAPL"], 
                    datetime(2024, 1, 1)
                )
                
                assert result["successful_tickers"] == []
                assert result["failed_tickers"] == ["AAPL"]


class TestFetchVolumeData:
    """Test fetch_volume_data function."""

    @pytest.mark.asyncio
    async def test_fetch_volume_data_success(self, sample_ohlcv_data):
        """Test successful volume extraction."""
        mock_raw_data = {"AAPL": sample_ohlcv_data}
        
        with patch("app.flows.compare.fetch_raw_ticker_data") as mock_fetch:
            mock_fetch.return_value = mock_raw_data
            
            result = await fetch_volume_data(
                ["AAPL"], 
                datetime(2024, 1, 1)
            )
            
            assert result["successful_tickers"] == ["AAPL"]
            assert len(result["failed_tickers"]) == 0
            assert isinstance(result["data"], pd.DataFrame)
            assert "AAPL" in result["data"].columns

    @pytest.mark.asyncio
    async def test_fetch_volume_data_missing_volume_column(self):
        """Test handling of data without Volume column."""
        # Create data without Volume column
        bad_data = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [102, 103, 104],
        }, index=pd.date_range("2024-01-01", periods=3, freq="D"))
        
        mock_raw_data = {"AAPL": bad_data}
        
        with patch("app.flows.compare.fetch_raw_ticker_data") as mock_fetch:
            mock_fetch.return_value = mock_raw_data
            
            result = await fetch_volume_data(
                ["AAPL"], 
                datetime(2024, 1, 1)
            )
            
            assert result["successful_tickers"] == []
            assert result["failed_tickers"] == ["AAPL"]


class TestFetchRSIData:
    """Test fetch_rsi_data function."""

    @pytest.mark.asyncio
    async def test_fetch_rsi_data_success(self, sample_ohlcv_data):
        """Test successful RSI calculation."""
        mock_raw_data = {"AAPL": sample_ohlcv_data}
        
        with patch("app.flows.compare.fetch_raw_ticker_data") as mock_fetch:
            mock_fetch.return_value = mock_raw_data
            
            # Mock the calculate_rsi function
            mock_rsi_data = pd.DataFrame({
                "RSI": [30, 70, 50]
            }, index=pd.date_range("2024-01-01", periods=3, freq="D"))
            
            with patch("app.flows.compare.calculate_rsi") as mock_calc:
                mock_calc.return_value = mock_rsi_data
                
                result = await fetch_rsi_data(
                    ["AAPL"], 
                    datetime(2024, 1, 1)
                )
                
                assert result["successful_tickers"] == ["AAPL"]
                assert len(result["failed_tickers"]) == 0
                assert isinstance(result["data"], pd.DataFrame)
                assert "AAPL" in result["data"].columns


class TestCompareDataWorkflow:
    """Test CompareDataWorkflow class."""

    def test_workflow_initialization(self):
        """Test workflow can be initialized."""
        with patch("app.flows.compare.create_yahoo_history_provider") as mock_create:
            mock_create.return_value = MagicMock()
            workflow = CompareDataWorkflow()
            assert workflow.yahoo_history is not None

    @pytest.mark.asyncio
    async def test_workflow_fetch_ticker_data_step(self, sample_ohlcv_data, mock_provider_result):
        """Test the fetch_ticker_data step."""
        mock_provider_result.data = sample_ohlcv_data
        
        with patch("app.flows.compare.create_yahoo_history_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.get_data.return_value = mock_provider_result
            mock_create.return_value = mock_provider
            
            workflow = CompareDataWorkflow()
            
            # Create mock start event
            start_event = MagicMock()
            start_event.tickers = ["AAPL", "MSFT"]
            start_event.base_date = datetime(2024, 1, 1)
            
            result = await workflow.fetch_ticker_data(start_event)
            
            assert result.tickers == ["AAPL", "MSFT"]
            assert result.base_date == datetime(2024, 1, 1)
            assert "AAPL" in result.results
            assert "MSFT" in result.results