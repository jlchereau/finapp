"""
A collection of finance functions for the computation of key financial metrics
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from app.lib.logger import logger


def get_close_prices(data: pd.DataFrame, ticker: str) -> pd.Series | None:
    """
    Extract close prices from OHLCV data with fallback to Adj Close.

    Args:
        data: OHLCV DataFrame
        ticker: Ticker symbol for logging

    Returns:
        Close price Series or None if not available
    """
    if "Close" in data.columns:
        return data["Close"].dropna()
    elif "Adj Close" in data.columns:
        return data["Adj Close"].dropna()
    else:
        logger.warning(f"No Close price data for {ticker}")
        return None


def calculate_returns(
    data: pd.DataFrame, base_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Calculate percentage returns from price data.

    IMPORTANT: For workflow usage, pass base_date=None and filter the result afterwards.
    This ensures calculations are done on full historical data before period filtering.

    Args:
        data: DataFrame with price data (must have 'Close' or 'Adj Close' column)
        base_date: Optional base date to normalize from (if None, uses first row)
                  For workflows: pass None and filter result with
                  ensure_minimum_data_points()

    Returns:
        DataFrame with percentage returns normalized to base date
    """
    if data.empty:
        return pd.DataFrame()

    # Get close prices using centralized function
    prices = get_close_prices(data, "unknown")
    if prices is None:
        raise ValueError("No Close price data found in DataFrame")

    # Filter to base date if provided (legacy behavior for backward compatibility)
    if base_date is not None:
        prices = prices[prices.index >= base_date]

    if len(prices) == 0:
        return pd.DataFrame()

    # Calculate percentage returns from first value
    # Ensure we're working with a pandas Series (not numpy array)
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices, index=data.index)

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

    # Get close prices using centralized function
    prices = get_close_prices(data, "unknown")
    if prices is None:
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

    # Get close prices using centralized function
    prices = get_close_prices(data, "unknown")
    if prices is None:
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

    # Convert to DataFrame if it's a Series
    if isinstance(rsi, pd.Series):
        return rsi.to_frame("RSI")
    else:
        # Handle case where rsi might be an array
        return pd.DataFrame({"RSI": rsi}, index=data.index[window:])


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


def calculate_exponential_trend(
    data: pd.DataFrame, column: str
) -> Optional[Dict[str, Any]]:
    """
    Calculate exponential trend lines with confidence intervals for a time series.

    Uses log-linear regression to fit an exponential trend and calculates
    standard deviation bands at 1 and 2 sigma levels.

    Args:
        data: DataFrame with datetime index and the specified column
        column: Name of the column to analyze

    Returns:
        Dictionary with trend data containing:
        - dates: Original datetime index
        - trend: Exponential trend line values
        - plus1_std: +1 standard deviation band
        - minus1_std: -1 standard deviation band
        - plus2_std: +2 standard deviation band
        - minus2_std: -2 standard deviation band

        Returns None if calculation fails or insufficient data
    """
    if data.empty or len(data) < 2:
        return None

    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Prepare data for regression using fractional years for smooth curves
    # Convert dates to fractional years to avoid stair-step pattern in quarterly data
    def date_to_fractional_year(date):
        """Convert datetime to fractional year (e.g., 2020.25 for Q2 2020)."""
        year = date.year
        # Calculate fraction of year: (day_of_year - 1) / days_in_year
        start_of_year = pd.Timestamp(year, 1, 1)
        days_in_year = 366 if pd.Timestamp(year, 12, 31).day_of_year == 366 else 365
        day_of_year = (date - start_of_year).days + 1
        return year + (day_of_year - 1) / days_in_year

    # Convert index to fractional years for smooth regression
    fractional_years = np.array([date_to_fractional_year(date) for date in data.index])
    X = fractional_years.reshape(-1, 1)
    y = data[column].values

    if len(y) < 2:
        return None

    # Handle zeros/negatives
    # Convert to numpy array to avoid ExtensionArray comparison issues
    y_array = np.asarray(y)
    if np.any(np.isnan(y_array) | (y_array <= 0)):
        raise ValueError(f"Column '{column}' contains non-positive values")

    y_log = np.log(y_array)

    # Fit linear regression on log-transformed data
    model = LinearRegression()
    model.fit(X, y_log)

    # Generate predictions
    y_log_pred = model.predict(X)
    y_pred = np.exp(y_log_pred)

    # Calculate residuals and standard deviation
    residuals = y_log - y_log_pred
    std = residuals.std()

    # Generate trend lines with confidence intervals
    y_plus1 = np.exp(y_log_pred + std)
    y_minus1 = np.exp(y_log_pred - std)
    y_plus2 = np.exp(y_log_pred + 2 * std)
    y_minus2 = np.exp(y_log_pred - 2 * std)

    return {
        "dates": data.index,
        "trend": y_pred,
        "plus1_std": y_plus1,
        "minus1_std": y_minus1,
        "plus2_std": y_plus2,
        "minus2_std": y_minus2,
    }
