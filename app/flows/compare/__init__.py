"""
Compare flows module.

This module provides LlamaIndex workflows for financial data comparison
across multiple tickers, including returns, volatility, volume, and RSI analysis.
"""

from .time_series import TimeSeriesWorkflow, fetch_time_series_data
from .metrics import MetricsWorkflow
from .returns import ReturnsWorkflow, fetch_returns_data
from .volatility import VolatilityWorkflow, fetch_volatility_data
from .volume import VolumeWorkflow, fetch_volume_data
from .rsi import RSIWorkflow, fetch_rsi_data

__all__ = [
    "TimeSeriesWorkflow",
    "fetch_time_series_data",
    "MetricsWorkflow",
    "ReturnsWorkflow",
    "fetch_returns_data",
    "VolatilityWorkflow",
    "fetch_volatility_data",
    "VolumeWorkflow",
    "fetch_volume_data",
    "RSIWorkflow",
    "fetch_rsi_data",
]
