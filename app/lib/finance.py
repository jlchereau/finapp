"""
A collection of finance functions for the computation of key financial metrics
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_returns(
    data: pd.DataFrame, base_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Calculate percentage returns from price data.

    Args:
        data: DataFrame with price data (must have 'Close' or 'Adj Close' column)
        base_date: Optional base date to normalize from (if None, uses first row)

    Returns:
        DataFrame with percentage returns normalized to base date
    """
    if data.empty:
        return pd.DataFrame()

    # Get close prices
    if "Close" in data.columns:
        prices = data["Close"]
    elif "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        raise ValueError("No Close price data found in DataFrame")

    # Filter to base date if provided
    if base_date is not None:
        prices = prices[prices.index >= base_date]

    if prices.empty:
        return pd.DataFrame()

    # Calculate percentage returns from first value
    first_price = prices.iloc[0]
    returns = ((prices / first_price) - 1) * 100

    return returns.to_frame("Returns")


def calculate_volatility(
    data: pd.DataFrame, window: int = 30, annualize: bool = True
) -> pd.DataFrame:
    """
    Calculate rolling volatility from price data.

    Args:
        data: DataFrame with price data (must have 'Close' or 'Adj Close' column)
        window: Rolling window size in days (default: 30)
        annualize: Whether to annualize volatility (default: True)

    Returns:
        DataFrame with rolling volatility values
    """
    if data.empty:
        return pd.DataFrame()

    # Get close prices
    if "Close" in data.columns:
        prices = data["Close"]
    elif "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        raise ValueError("No Close price data found in DataFrame")

    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()

    # Calculate rolling standard deviation
    rolling_std = daily_returns.rolling(window=window).std()

    # Annualize if requested (multiply by sqrt of trading days per year)
    if annualize:
        rolling_std = rolling_std * np.sqrt(252)
        rolling_std = rolling_std * 100  # Convert to percentage
    else:
        rolling_std = rolling_std * 100  # Convert to percentage

    return rolling_std.to_frame("Volatility")


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        data: DataFrame with price data (must have 'Close' or 'Adj Close' column)
        window: Period for RSI calculation (default: 14)

    Returns:
        DataFrame with RSI values (0-100 scale)
    """
    if data.empty:
        return pd.DataFrame()

    # Get close prices
    if "Close" in data.columns:
        prices = data["Close"]
    elif "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        raise ValueError("No Close price data found in DataFrame")

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)

    # Calculate rolling averages
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi.to_frame("RSI")


def calculate_volume_metrics(data: pd.DataFrame, ma_window: int = 20) -> pd.DataFrame:
    """
    Calculate volume metrics including raw volume and moving average.

    Args:
        data: DataFrame with volume data (must have 'Volume' column)
        ma_window: Window for moving average calculation (default: 20)

    Returns:
        DataFrame with volume and volume moving average
    """
    if data.empty:
        return pd.DataFrame()

    if "Volume" not in data.columns:
        raise ValueError("No Volume data found in DataFrame")

    volume = data["Volume"]
    volume_ma = volume.rolling(window=ma_window).mean()

    result = pd.DataFrame({"Volume": volume, "Volume_MA": volume_ma})

    return result
