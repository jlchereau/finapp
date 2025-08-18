"""
Fred provider module
This module provides functionality to fetch data from the FRED API.

See:
- https://fred.stlouisfed.org/docs/api/fred/
"""

import httpx
import pandas as pd
from pandas import DataFrame

from app.lib.settings import settings
from .base import (
    BaseProvider,
    ProviderType,
    ProviderConfig,
    NonRetriableProviderException,
    RetriableProviderException,
)
from .cache import cache


class FredSeriesProvider(BaseProvider[DataFrame]):
    """
    Fred series provider class.
    This class provides functionality to fetch data from the FRED API.
    See https://fred.stlouisfed.org/docs/api/fred/series_observations.html
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.FRED_SERIES

    @cache
    async def _fetch_data(self, query: str | None, **kwargs) -> DataFrame:
        """
        Fetch economic time series data from the FRED API.

        Args:
            query: FRED series ID (e.g., 'GDPC1' for Real GDP, 'CPIAUCSL' for CPI)
            **kwargs: Additional parameters:
                - observation_start: Start date in YYYY-MM-DD format
                - observation_end: End date in YYYY-MM-DD format
                - frequency: Data frequency (a, q, m, w, d)
                - aggregation_method: How to aggregate (avg, sum, eop)
                - units: Units transformation
                - limit: Maximum number of observations (default: 100000)

        Returns:
            DataFrame with DatetimeIndex and 'value' column containing the time
            series data

        Raises:
            ValueError: If series_id is invalid or no data is found
            Exception: For FRED API errors or HTTP issues
        """
        self.logger.debug(f"FredSeriesProvider._fetch_data called for query: {query}")

        # Validate API key
        if not settings.FRED_API_KEY:
            self.logger.error("FRED_API_KEY not configured in settings")
            raise ValueError("FRED_API_KEY is required but not configured")

        # Validate series_id
        if not query:
            self.logger.error("Series ID (query) is required for FRED provider")
            raise ValueError("Series ID must be provided")

        series_id = query.upper().strip()
        self.logger.debug(f"Normalized series ID: {series_id}")

        # Build request parameters
        params = {
            "series_id": series_id,
            "api_key": settings.FRED_API_KEY,
            "file_type": "json",
        }

        # Add optional parameters
        if observation_start := kwargs.get("observation_start"):
            params["observation_start"] = observation_start
        if observation_end := kwargs.get("observation_end"):
            params["observation_end"] = observation_end
        if frequency := kwargs.get("frequency"):
            params["frequency"] = frequency
        if aggregation_method := kwargs.get("aggregation_method"):
            params["aggregation_method"] = aggregation_method
        if units := kwargs.get("units"):
            params["units"] = units
        if limit := kwargs.get("limit", 100000):
            params["limit"] = limit

        self.logger.debug(f"FRED API request params: {params}")

        try:
            # Make HTTP request to FRED API
            url = "https://api.stlouisfed.org/fred/series/observations"

            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                self.logger.info(f"Calling FRED API for series {series_id}")
                response = await client.get(url, params=params)
                response.raise_for_status()
                json_data = response.json()

            # Check for FRED API errors
            if "error_code" in json_data and "error_message" in json_data:
                error_msg = json_data["error_message"]
                self.logger.error(f"FRED API error for {series_id}: {error_msg}")
                if (
                    "Bad Request" in error_msg
                    or "series does not exist" in error_msg.lower()
                ):
                    raise NonRetriableProviderException(
                        f"Invalid series ID: {error_msg}"
                    )
                else:
                    raise RetriableProviderException(f"FRED API error: {error_msg}")

            # Extract observations
            observations = json_data.get("observations", [])
            if not observations:
                self.logger.warning(f"No observations returned for series {series_id}")
                raise ValueError(f"No data found for series ID: {series_id}")

            self.logger.debug(
                f"Retrieved {len(observations)} observations for {series_id}"
            )

            # Convert to DataFrame
            data_dict = {}
            for obs in observations:
                date_str = obs.get("date")
                value_str = obs.get("value")

                if date_str and value_str != ".":  # FRED uses "." for missing values
                    try:
                        date = pd.to_datetime(date_str)
                        value = float(value_str)
                        data_dict[date] = value
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"Skipping invalid data point: date={date_str}, "
                            f"value={value_str}, error={e}"
                        )
                        continue

            if not data_dict:
                self.logger.warning(
                    f"No valid data points found for series {series_id}"
                )
                raise ValueError(
                    f"No valid data points found for series ID: {series_id}"
                )

            # Create DataFrame with DatetimeIndex
            df = pd.DataFrame.from_dict(data_dict, orient="index", columns=["value"])
            df.index.name = "date"
            df = df.sort_index()  # Ensure chronological order

            self.logger.debug(
                f"Successfully processed {len(df)} data points for {series_id}"
            )
            return df

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                self.logger.error(f"Bad request for series {series_id}: {e}")
                raise NonRetriableProviderException(
                    f"Invalid request for series {series_id}: {e}"
                ) from e
            elif e.response.status_code == 403:
                self.logger.error(f"Authentication failed for FRED API: {e}")
                raise NonRetriableProviderException("Invalid FRED API key") from e
            else:
                self.logger.warning(f"HTTP error for series {series_id}: {e}")
                raise RetriableProviderException(f"HTTP error: {e}") from e
        except httpx.RequestError as e:
            self.logger.warning(f"Network error for series {series_id}: {e}")
            raise RetriableProviderException(f"Network error: {e}") from e
        except ValueError as e:
            # Non-retriable errors (e.g., empty data, invalid series)
            self.logger.error(f"Non-retriable error for series {series_id}: {e}")
            raise NonRetriableProviderException(str(e)) from e
        except Exception as e:
            # Other errors are retriable
            self.logger.warning(
                f"Retriable error for series {series_id}: {type(e).__name__}: {e}"
            )
            raise RetriableProviderException(str(e)) from e


# Factory function for easy provider creation
def create_fred_series_provider(
    timeout: float = 30.0,
    retries: int = 3,
) -> FredSeriesProvider:
    """
    Factory function to create a FRED Series provider with custom settings.

    Args:
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Configured FredSeriesProvider instance
    """
    config = ProviderConfig(timeout=timeout, retries=retries)
    return FredSeriesProvider(config)
