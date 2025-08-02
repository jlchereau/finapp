"""
Unit tests for the TickerModel class.
"""

import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import pytest

from app.models.ticker import TickerModel


class TestTickerModel:
    """Test cases for TickerModel class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.ticker_model = TickerModel(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init_creates_data_directory(self):
        """Test that TickerModel creates the data directory on initialization."""
        new_temp_dir = tempfile.mkdtemp()
        shutil.rmtree(new_temp_dir)  # Remove it so we can test creation
        
        model = TickerModel(data_dir=new_temp_dir)
        
        assert Path(new_temp_dir).exists()
        assert model.data_dir == Path(new_temp_dir)
        
        # Clean up
        shutil.rmtree(new_temp_dir)
    
    def test_init_with_existing_directory(self):
        """Test that TickerModel works with existing data directory."""
        model = TickerModel(data_dir=self.temp_dir)
        
        assert model.data_dir == Path(self.temp_dir)
        assert Path(self.temp_dir).exists()
    
    def test_get_cache_path(self):
        """Test cache path generation."""
        ticker = "AAPL"
        date = "20240101"
        
        cache_path = self.ticker_model._get_cache_path(ticker, date)
        expected_path = Path(self.temp_dir) / "20240101.AAPL.parquet"
        
        assert cache_path == expected_path
    
    def test_get_cache_path_lowercase_ticker(self):
        """Test cache path generation with lowercase ticker."""
        ticker = "aapl"
        date = "20240101"
        
        cache_path = self.ticker_model._get_cache_path(ticker, date)
        expected_path = Path(self.temp_dir) / "20240101.AAPL.parquet"
        
        assert cache_path == expected_path
    
    def test_is_cache_valid_no_file(self):
        """Test cache validation when no cache file exists."""
        result = self.ticker_model._is_cache_valid("AAPL", "20240101")
        assert result is False
    
    def test_is_cache_valid_old_file(self):
        """Test cache validation with old cache file."""
        ticker = "AAPL"
        date = datetime.now().strftime('%Y%m%d')
        cache_path = self.ticker_model._get_cache_path(ticker, date)
        
        # Create a cache file with old timestamp
        cache_path.touch()
        yesterday = datetime.now() - timedelta(days=1)
        os.utime(cache_path, (yesterday.timestamp(), yesterday.timestamp()))
        
        result = self.ticker_model._is_cache_valid(ticker, date)
        assert result is False
    
    def test_is_cache_valid_current_file(self):
        """Test cache validation with current cache file."""
        ticker = "AAPL"
        date = datetime.now().strftime('%Y%m%d')
        cache_path = self.ticker_model._get_cache_path(ticker, date)
        
        # Create a cache file with current timestamp
        cache_path.touch()
        
        result = self.ticker_model._is_cache_valid(ticker, date)
        assert result is True
    
    @patch('app.models.ticker.yf.Ticker')
    def test_get_ticker_data_from_yfinance(self, mock_yf_ticker):
        """Test fetching ticker data from yfinance when no cache exists."""
        # Mock yfinance data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_yf_ticker.return_value = mock_ticker_instance
        
        result = self.ticker_model.get_ticker_data("AAPL", "1y")
        
        assert result is not None
        assert len(result) == 3
        assert 'Close' in result.columns
        mock_yf_ticker.assert_called_once_with("AAPL")
        mock_ticker_instance.history.assert_called_once_with(period="1y")
    
    @patch('app.models.ticker.yf.Ticker')
    def test_get_ticker_data_empty_response(self, mock_yf_ticker):
        """Test handling of empty response from yfinance."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_yf_ticker.return_value = mock_ticker_instance
        
        result = self.ticker_model.get_ticker_data("INVALID", "1y")
        
        assert result is None
    
    @patch('app.models.ticker.yf.Ticker')
    def test_get_ticker_data_yfinance_exception(self, mock_yf_ticker):
        """Test handling of yfinance exceptions."""
        mock_yf_ticker.side_effect = Exception("Network error")
        
        result = self.ticker_model.get_ticker_data("AAPL", "1y")
        
        assert result is None
    
    def test_get_ticker_data_from_cache(self):
        """Test loading ticker data from cache."""
        # Create mock cached data
        mock_data = pd.DataFrame({
            'Open': [100, 101],
            'Close': [100.5, 101.5]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        today = datetime.now().strftime('%Y%m%d')
        cache_path = self.ticker_model._get_cache_path("AAPL", today)
        mock_data.to_parquet(cache_path)
        
        result = self.ticker_model.get_ticker_data("AAPL", "1y")
        
        assert result is not None
        assert len(result) == 2
        # Compare values since index might have different freq attribute
        pd.testing.assert_frame_equal(result.reset_index(drop=True), mock_data.reset_index(drop=True))
    
    @patch('app.models.ticker.yf.Ticker')
    def test_get_ticker_info_success(self, mock_yf_ticker):
        """Test successful ticker info retrieval."""
        mock_info = {
            'symbol': 'AAPL',
            'longName': 'Apple Inc.',
            'trailingPE': 25.5,
            'priceToBook': 5.2
        }
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_yf_ticker.return_value = mock_ticker_instance
        
        result = self.ticker_model.get_ticker_info("AAPL")
        
        assert result == mock_info
        mock_yf_ticker.assert_called_once_with("AAPL")
    
    @patch('app.models.ticker.yf.Ticker')
    def test_get_ticker_info_exception(self, mock_yf_ticker):
        """Test handling of exceptions during ticker info retrieval."""
        mock_yf_ticker.side_effect = Exception("API error")
        
        result = self.ticker_model.get_ticker_info("AAPL")
        
        assert result is None
    
    @patch.object(TickerModel, 'get_ticker_data')
    def test_get_price_data_empty_tickers(self, mock_get_ticker_data):
        """Test get_price_data with empty ticker list."""
        result = self.ticker_model.get_price_data([])
        
        assert result.empty
        mock_get_ticker_data.assert_not_called()
    
    @patch.object(TickerModel, 'get_ticker_data')
    def test_get_price_data_single_ticker(self, mock_get_ticker_data):
        """Test get_price_data with single ticker."""
        mock_data = pd.DataFrame({
            'Close': [100, 110, 120]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_get_ticker_data.return_value = mock_data
        
        result = self.ticker_model.get_price_data(["AAPL"])
        
        expected = pd.DataFrame({
            'AAPL': [100.0, 110.0, 120.0]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        assert not result.empty
        pd.testing.assert_frame_equal(result, expected)
    
    @patch.object(TickerModel, 'get_ticker_data')
    def test_get_price_data_multiple_tickers(self, mock_get_ticker_data):
        """Test get_price_data with multiple tickers."""
        def mock_data_side_effect(ticker):
            if ticker == "AAPL":
                return pd.DataFrame({
                    'Close': [100, 110, 120]
                }, index=pd.date_range('2024-01-01', periods=3))
            elif ticker == "MSFT":
                return pd.DataFrame({
                    'Close': [200, 210, 220]
                }, index=pd.date_range('2024-01-01', periods=3))
            return None
        
        mock_get_ticker_data.side_effect = mock_data_side_effect
        
        result = self.ticker_model.get_price_data(["AAPL", "MSFT"])
        
        expected = pd.DataFrame({
            'AAPL': [100.0, 110.0, 120.0],
            'MSFT': [100.0, 105.0, 110.0]  # Normalized to base 100
        }, index=pd.date_range('2024-01-01', periods=3))
        
        assert not result.empty
        assert len(result.columns) == 2
        # Check that values are normalized to base 100
        assert result.iloc[0]['AAPL'] == 100.0
        assert result.iloc[0]['MSFT'] == 100.0
    
    @patch.object(TickerModel, 'get_ticker_data')
    def test_get_price_data_with_base_date(self, mock_get_ticker_data):
        """Test get_price_data with specific base date."""
        mock_data = pd.DataFrame({
            'Close': [100, 110, 120]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_get_ticker_data.return_value = mock_data
        
        result = self.ticker_model.get_price_data(["AAPL"], base_date="2024-01-02")
        
        # Should normalize to the second day's value (110)
        expected_values = [100/110*100, 110/110*100, 120/110*100]
        
        assert not result.empty
        assert abs(result.iloc[1]['AAPL'] - 100.0) < 0.001  # Second day should be 100
    
    @patch.object(TickerModel, 'get_ticker_info')
    def test_calculate_graham_metrics_success(self, mock_get_ticker_info):
        """Test successful Graham metrics calculation."""
        mock_info = {
            'trailingPE': 12.5,
            'priceToBook': 1.2,
            'debtToEquity': 30.0,  # 30% = 0.3 ratio
            'currentRatio': 2.5,
            'dividendYield': 0.025  # 2.5%
        }
        
        mock_get_ticker_info.return_value = mock_info
        
        result = self.ticker_model.calculate_graham_metrics("AAPL")
        
        assert 'PE Ratio' in result
        assert result['PE Ratio']['value'] == 12.5
        assert result['PE Ratio']['passes'] is True
        
        assert 'Price to Book' in result
        assert result['Price to Book']['value'] == 1.2
        assert result['Price to Book']['passes'] is True
        
        assert 'Debt to Equity' in result
        assert result['Debt to Equity']['value'] == 0.3
        assert result['Debt to Equity']['passes'] is True
        
        assert 'Current Ratio' in result
        assert result['Current Ratio']['value'] == 2.5
        assert result['Current Ratio']['passes'] is True
        
        assert 'Dividend Yield' in result
        assert result['Dividend Yield']['value'] == 2.5
        assert result['Dividend Yield']['passes'] is True
    
    @patch.object(TickerModel, 'get_ticker_info')
    def test_calculate_graham_metrics_failing_criteria(self, mock_get_ticker_info):
        """Test Graham metrics calculation with failing criteria."""
        mock_info = {
            'trailingPE': 25.0,  # > 15, should fail
            'priceToBook': 2.0,  # > 1.5, should fail
            'debtToEquity': 75.0,  # 75% = 0.75 ratio > 0.5, should fail
            'currentRatio': 1.5,  # < 2, should fail
            'dividendYield': 0.0  # No dividend, should fail
        }
        
        mock_get_ticker_info.return_value = mock_info
        
        result = self.ticker_model.calculate_graham_metrics("AAPL")
        
        assert result['PE Ratio']['passes'] is False
        assert result['Price to Book']['passes'] is False
        assert result['Debt to Equity']['passes'] is False
        assert result['Current Ratio']['passes'] is False
        assert result['Dividend Yield']['passes'] is False
    
    @patch.object(TickerModel, 'get_ticker_info')
    def test_calculate_graham_metrics_missing_data(self, mock_get_ticker_info):
        """Test Graham metrics calculation with missing data."""
        mock_info = {}  # Empty info dict is falsy, so returns empty result
        
        mock_get_ticker_info.return_value = mock_info
        
        result = self.ticker_model.calculate_graham_metrics("AAPL")
        
        # Empty dict is falsy, so method returns empty dict immediately
        assert result == {}
    
    @patch.object(TickerModel, 'get_ticker_info')
    def test_calculate_graham_metrics_partial_data(self, mock_get_ticker_info):
        """Test Graham metrics calculation with partial data (non-empty dict but missing keys)."""
        mock_info = {'symbol': 'AAPL'}  # Non-empty dict but missing financial metrics
        
        mock_get_ticker_info.return_value = mock_info
        
        result = self.ticker_model.calculate_graham_metrics("AAPL")
        
        # Should create metrics with default values from .get() calls
        assert 'PE Ratio' in result
        assert result['PE Ratio']['value'] == 0
        assert result['PE Ratio']['passes'] is False
        
        assert 'Price to Book' in result
        assert result['Price to Book']['value'] == 0
        assert result['Price to Book']['passes'] is False
        
        assert 'Current Ratio' in result
        assert result['Current Ratio']['value'] == 0
        assert result['Current Ratio']['passes'] is False
        
        assert 'Dividend Yield' in result
        assert result['Dividend Yield']['value'] == 0
        assert result['Dividend Yield']['passes'] is False
    
    @patch.object(TickerModel, 'get_ticker_info')
    def test_calculate_graham_metrics_no_info(self, mock_get_ticker_info):
        """Test Graham metrics calculation when ticker info is unavailable."""
        mock_get_ticker_info.return_value = None
        
        result = self.ticker_model.calculate_graham_metrics("INVALID")
        
        assert result == {}


class TestTickerModelIntegration:
    """Integration tests for TickerModel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ticker_model = TickerModel(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('app.models.ticker.yf.Ticker')
    def test_caching_workflow(self, mock_yf_ticker):
        """Test the complete caching workflow."""
        # Mock yfinance data
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_yf_ticker.return_value = mock_ticker_instance
        
        # First call should fetch from yfinance and cache
        result1 = self.ticker_model.get_ticker_data("AAPL", "1y")
        
        # Verify cache file was created
        today = datetime.now().strftime('%Y%m%d')
        cache_path = self.ticker_model._get_cache_path("AAPL", today)
        assert cache_path.exists()
        
        # Second call should load from cache (reset mock to verify)
        mock_yf_ticker.reset_mock()
        result2 = self.ticker_model.get_ticker_data("AAPL", "1y")
        
        # yfinance should not be called again
        mock_yf_ticker.assert_not_called()
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1.reset_index(drop=True), result2.reset_index(drop=True))