"""
Ticker data model for financial data fetching and caching.

Handles yfinance integration with parquet file caching in ./data/ directory.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import yfinance as yf


class TickerModel:
    """Model for fetching and caching ticker data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, ticker: str, date: str) -> Path:
        """Get cache file path for ticker data."""
        return self.data_dir / f"{date}.{ticker.upper()}.parquet"
    
    def _is_cache_valid(self, ticker: str, date: str) -> bool:
        """Check if cached data exists and is recent."""
        cache_path = self._get_cache_path(ticker, date)
        if not cache_path.exists():
            return False
        
        # Check if cache is from today (for daily data refresh)
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_time.date() == datetime.now().date()
    
    def get_ticker_data(self, ticker: str, period: str = "10y") -> Optional[pd.DataFrame]:
        """
        Get ticker data with caching.
        
        Args:
            ticker: Stock ticker symbol
            period: Period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        today = datetime.now().strftime('%Y%m%d')
        cache_path = self._get_cache_path(ticker, today)
        
        # Try to load from cache first
        if self._is_cache_valid(ticker, today):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass  # Fall through to fetch new data
        
        # Fetch from yfinance
        try:
            yf_ticker = yf.Ticker(ticker)
            data = yf_ticker.history(period=period)
            
            if data.empty:
                return None
            
            # Cache the data
            data.to_parquet(cache_path)
            return data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get ticker info/fundamentals from yfinance.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with ticker info or None if error
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            return yf_ticker.info
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return None
    
    def get_price_data(self, tickers: List[str], base_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get price data for multiple tickers, normalized to base 100.
        
        Args:
            tickers: List of ticker symbols
            base_date: Date to normalize to 100 (YYYY-MM-DD), defaults to earliest common date
            
        Returns:
            DataFrame with normalized prices for each ticker
        """
        if not tickers:
            return pd.DataFrame()
        
        all_data = {}
        for ticker in tickers:
            data = self.get_ticker_data(ticker)
            if data is not None and not data.empty:
                all_data[ticker] = data['Close']
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all ticker data
        combined = pd.DataFrame(all_data)
        combined = combined.dropna()
        
        if combined.empty:
            return pd.DataFrame()
        
        # Normalize to base 100
        if base_date:
            try:
                base_date_parsed = pd.to_datetime(base_date)
                if base_date_parsed in combined.index:
                    base_values = combined.loc[base_date_parsed]
                    normalized = (combined / base_values) * 100
                else:
                    # Use closest date if exact date not found
                    closest_idx = combined.index.get_indexer([base_date_parsed], method='nearest')[0]
                    base_values = combined.iloc[closest_idx]
                    normalized = (combined / base_values) * 100
            except Exception:
                # Fall back to first date if base_date parsing fails
                base_values = combined.iloc[0]
                normalized = (combined / base_values) * 100
        else:
            # Use first date as base
            base_values = combined.iloc[0]
            normalized = (combined / base_values) * 100
        
        return normalized
    
    def calculate_graham_metrics(self, ticker: str) -> Dict[str, Any]:
        """
        Calculate Graham's defensive investor criteria.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with metrics and pass/fail status
        """
        info = self.get_ticker_info(ticker)
        if not info:
            return {}
        
        metrics = {}
        
        try:
            # PE Ratio (should be < 15 for defensive investors)
            pe_ratio = info.get('trailingPE', 0)
            metrics['PE Ratio'] = {
                'value': pe_ratio,
                'criterion': '< 15',
                'passes': pe_ratio < 15 if pe_ratio else False
            }
            
            # Price to Book (should be < 1.5)
            pb_ratio = info.get('priceToBook', 0)
            metrics['Price to Book'] = {
                'value': pb_ratio,
                'criterion': '< 1.5',
                'passes': pb_ratio < 1.5 if pb_ratio else False
            }
            
            # Debt to Equity (should be < 0.5)
            debt_to_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
            metrics['Debt to Equity'] = {
                'value': debt_to_equity,
                'criterion': '< 0.5',
                'passes': debt_to_equity < 0.5
            }
            
            # Current Ratio (should be > 2)
            current_ratio = info.get('currentRatio', 0)
            metrics['Current Ratio'] = {
                'value': current_ratio,
                'criterion': '> 2',
                'passes': current_ratio > 2 if current_ratio else False
            }
            
            # Dividend Yield (should exist for defensive stocks)
            dividend_yield = info.get('dividendYield', 0)
            metrics['Dividend Yield'] = {
                'value': dividend_yield * 100 if dividend_yield else 0,
                'criterion': '> 0%',
                'passes': dividend_yield > 0 if dividend_yield else False
            }
            
        except Exception as e:
            print(f"Error calculating Graham metrics for {ticker}: {e}")
        
        return metrics


# Global instance for use across the app
ticker_model = TickerModel()