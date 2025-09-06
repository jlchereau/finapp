# pylint: disable=broad-exception-raised
"""
LlamaIndex workflow for market page data collection.

This workflow handles fetching economic data for market indicators
like the Buffet Indicator using FredSeriesProvider and YahooHistoryProvider.
"""

from typing import Dict, Any
from datetime import datetime

import pandas as pd
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from app.providers.fred import create_fred_series_provider
from app.providers.yahoo import create_yahoo_history_provider
from app.flows.helpers import (
    process_multiple_provider_results,
    validate_single_provider_task,
)
from app.lib.logger import logger
from app.lib.exceptions import WorkflowException
from app.flows.cache import apply_flow_cache


class VIXEvent(Event):
    """Event emitted when VIX data is fetched."""

    vix_data: pd.DataFrame
    base_date: datetime


class YieldCurveEvent(Event):
    """Event emitted when yield curve data is fetched."""

    yield_curve_data: pd.DataFrame
    base_date: datetime


class CurrencyEvent(Event):
    """Event emitted when currency data is fetched."""

    usdeur_data: pd.DataFrame
    gbpeur_data: pd.DataFrame
    base_date: datetime


class PreciousMetalsEvent(Event):
    """Event emitted when precious metals data is fetched."""

    gold_data: pd.DataFrame
    base_date: datetime


class CryptoCurrencyEvent(Event):
    """Event emitted when cryptocurrency data is fetched."""

    bitcoin_data: pd.DataFrame
    ethereum_data: pd.DataFrame
    base_date: datetime


class CrudeOilEvent(Event):
    """Event emitted when crude oil data is fetched."""

    wti_data: pd.DataFrame
    brent_data: pd.DataFrame
    base_date: datetime


class BloombergCommodityEvent(Event):
    """Event emitted when Bloomberg Commodity Index data is fetched."""

    bcom_data: pd.DataFrame
    base_date: datetime


class MSCIWorldEvent(Event):
    """Event emitted when MSCI World Index data is fetched."""

    msci_data: pd.DataFrame
    base_date: datetime


class VIXWorkflow(Workflow):
    """
    Workflow that fetches VIX (volatility index) data.

    The VIX is a measure of market volatility, often called the "fear gauge".
    We use the ^VIX ticker from Yahoo Finance to get historical VIX data.

    This workflow:
    - Fetches daily VIX data from Yahoo Finance
    - Calculates historical mean for reference
    - Filters data based on base_date for display
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_vix_data(self, ev: StartEvent) -> VIXEvent:
        """
        Fetch VIX data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            VIXEvent with VIX data
        """
        base_date = ev.base_date

        logger.debug(f"VIXWorkflow: Fetching VIX data from {base_date}")

        # Fetch VIX data using helper function
        vix_result = await self.yahoo_provider.get_data("^VIX")
        vix_data = validate_single_provider_task(vix_result, "VIX")

        return VIXEvent(vix_data=vix_data, base_date=base_date)

    @step
    async def process_vix_data(self, ev: VIXEvent) -> StopEvent:
        """
        Process VIX data and calculate statistics.

        Args:
            ev: VIXEvent with VIX data

        Returns:
            StopEvent with processed VIX data and statistics
        """
        vix_data = ev.vix_data
        base_date = ev.base_date

        logger.debug("VIXWorkflow: Processing VIX data")

        try:
            # Extract close prices from VIX data
            if "Close" in vix_data.columns:
                vix_close = vix_data["Close"].dropna()
            elif "Adj Close" in vix_data.columns:
                vix_close = vix_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in VIX data")
                raise Exception("No Close price data available for VIX")

            if vix_close.empty:
                logger.error("No VIX close price data available")
                raise Exception("No VIX data available")

            # Normalize timezone to ensure proper filtering
            if vix_close.index.tz is not None:
                vix_close_naive = vix_close.tz_localize(None)
            else:
                vix_close_naive = vix_close

            # Calculate historical mean from full dataset
            historical_mean = float(vix_close_naive.mean())

            # Calculate 50-day moving average on full dataset
            moving_avg_50 = vix_close_naive.rolling(window=50, min_periods=1).mean()

            # Create complete result DataFrame
            result_df = pd.DataFrame(
                {
                    "VIX": vix_close_naive,
                    "VIX_MA50": moving_avg_50,
                },
                index=vix_close_naive.index,
            )

            # Filter by base_date for display purposes
            base_date_pd = pd.to_datetime(base_date.date())
            display_data = result_df[result_df.index >= base_date_pd]

            if display_data.empty:
                logger.warning(f"No VIX data after base_date {base_date} for display")
                # Return empty result but don't error - this is just a display filter
                display_data = pd.DataFrame(columns=["VIX", "VIX_MA50"])

            logger.info(
                f"VIX processing completed: {len(display_data)} data points "
                f"from {base_date}, historical mean: {historical_mean:.2f}"
            )

            return StopEvent(
                result={
                    "data": display_data,
                    "base_date": base_date,
                    "historical_mean": historical_mean,
                    "latest_value": (
                        display_data["VIX"].iloc[-1] if not display_data.empty else None
                    ),
                    "data_points": len(display_data),
                }
            )

        except Exception as e:
            logger.error(f"Error processing VIX data: {e}")
            # Re-raise as WorkflowException for better handling
            raise WorkflowException(
                workflow="VIXWorkflow",
                step="process_vix_data",
                message=f"VIX data processing failed: {e}",
                user_message="Failed to process VIX data. Please try again later.",
                context={"base_date": str(base_date)},
            ) from e


class YieldCurveWorkflow(Workflow):
    """
    Workflow that fetches US Treasury yield curve data.

    The yield curve shows interest rates across different maturities,
    from 1-month to 30-year Treasury securities. This workflow:
    - Fetches multiple Treasury series from FRED API concurrently
    - Combines data into structured DataFrame with maturity columns
    - Applies period filtering with minimum data points guarantee
    """

    def __init__(self):
        """Initialize workflow with FRED provider."""
        super().__init__()
        # Create provider
        self.fred_provider = create_fred_series_provider(timeout=30.0, retries=2)

    @step
    async def fetch_yield_data(self, ev: StartEvent) -> YieldCurveEvent:
        """
        Fetch yield curve data from FRED API.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            YieldCurveEvent with yield curve data
        """
        base_date = ev.base_date

        logger.debug(f"YieldCurveWorkflow: Fetching yield curve data from {base_date}")

        try:
            # Fetch yield curve data using the new method
            yield_data = await self.fred_provider.fetch_yield_curve_data(
                query=None,  # Not used for yield curve data
                observation_start=base_date.strftime("%Y-%m-%d") if base_date else None,
            )

            if yield_data.empty:
                logger.error("Empty yield curve data returned")
                raise Exception("No yield curve data available")

            logger.debug(
                f"Yield curve data: {len(yield_data)} rows, "
                f"{len(yield_data.columns)} maturities, "
                f"range: {yield_data.index.min()} to {yield_data.index.max()}"
            )

            return YieldCurveEvent(yield_curve_data=yield_data, base_date=base_date)

        except Exception as e:
            logger.error(f"Failed to fetch yield curve data: {e}")
            raise

    @step
    async def process_yield_data(self, ev: YieldCurveEvent) -> StopEvent:
        """
        Process yield curve data and apply filtering.

        Args:
            ev: YieldCurveEvent with yield curve data

        Returns:
            StopEvent with processed yield curve data
        """
        yield_data = ev.yield_curve_data
        base_date = ev.base_date

        logger.debug("YieldCurveWorkflow: Processing yield curve data")

        try:
            # Apply date filtering if base_date is provided
            if base_date:
                base_date_pd = pd.to_datetime(base_date.date())
                display_data = yield_data[yield_data.index >= base_date_pd]
            else:
                display_data = yield_data

            if display_data.empty:
                logger.warning(f"No yield curve data after base_date {base_date}")
                # Return empty result but don't error
                display_data = pd.DataFrame()

            logger.info(
                f"YieldCurve completed: {len(display_data)} trading days "
                f"from {base_date}"
            )

            return StopEvent(
                result={
                    "data": display_data,
                    "base_date": base_date,
                    "latest_date": (
                        display_data.index[-1] if not display_data.empty else None
                    ),
                    "latest_yields": (
                        display_data.iloc[-1].to_dict()
                        if not display_data.empty
                        else {}
                    ),
                    "data_points": len(display_data),
                    "maturities": (
                        list(display_data.columns) if not display_data.empty else []
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Error processing yield curve data: {e}")
            # Re-raise as WorkflowException for better handling
            raise WorkflowException(
                workflow="YieldCurveWorkflow",
                step="process_yield_data",
                message=f"Yield curve data processing failed: {e}",
                user_message=(
                    "Failed to process yield curve data. Please try again later."
                ),
                context={"base_date": str(base_date)},
            ) from e


class CurrencyWorkflow(Workflow):
    """
    Workflow that fetches currency exchange rate data.

    This workflow fetches EUR/USD and EUR/GBP exchange rates from Yahoo Finance
    to display major currency pairs against the Euro.

    The workflow:
    - Fetches EURUSD=X and EURGBP=X data from Yahoo Finance in parallel
    - Normalizes and aligns the data on common dates
    - Applies base_date filtering for display
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_currency_data(self, ev: StartEvent) -> CurrencyEvent:
        """
        Fetch currency exchange rate data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            CurrencyEvent with EUR/USD and EUR/GBP data
        """
        base_date = ev.base_date

        logger.debug(f"CurrencyWorkflow: Fetching currency data from {base_date}")

        # Create tasks for parallel execution
        tasks = {
            "USD_EUR": self.yahoo_provider.get_data("EUR=X"),  # USD/EUR
            "GBP_EUR": self.yahoo_provider.get_data("GBPEUR=X"),  # GBP/EUR
        }

        # Execute tasks in parallel using helper function
        results = await process_multiple_provider_results(tasks)

        # Check for failures and extract data
        if not results["USD_EUR"]["success"]:
            raise Exception(f"USD/EUR data fetch failed: {results['USD_EUR']['error']}")
        if not results["GBP_EUR"]["success"]:
            raise Exception(f"GBP/EUR data fetch failed: {results['GBP_EUR']['error']}")

        usdeur_data = results["USD_EUR"]["data"]
        gbpeur_data = results["GBP_EUR"]["data"]

        return CurrencyEvent(
            usdeur_data=usdeur_data, gbpeur_data=gbpeur_data, base_date=base_date
        )

    @step
    async def process_currency_data(self, ev: CurrencyEvent) -> StopEvent:
        """
        Process and combine currency data.

        Args:
            ev: CurrencyEvent with USD/EUR and GBP/EUR data

        Returns:
            StopEvent with processed currency data
        """
        usdeur_data = ev.usdeur_data
        gbpeur_data = ev.gbpeur_data
        base_date = ev.base_date

        logger.debug("CurrencyWorkflow: Processing currency data")

        try:
            # Extract close prices from USD/EUR data
            if "Close" in usdeur_data.columns:
                usdeur_close = usdeur_data["Close"].dropna()
            elif "Adj Close" in usdeur_data.columns:
                usdeur_close = usdeur_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in USD/EUR data")
                raise Exception("No Close price data available for USD/EUR")

            if usdeur_close.empty:
                logger.error("No USD/EUR close price data available")
                raise Exception("No USD/EUR data available")

            # Extract close prices from GBP/EUR data
            if "Close" in gbpeur_data.columns:
                gbpeur_close = gbpeur_data["Close"].dropna()
            elif "Adj Close" in gbpeur_data.columns:
                gbpeur_close = gbpeur_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in GBP/EUR data")
                raise Exception("No Close price data available for GBP/EUR")

            if gbpeur_close.empty:
                logger.error("No GBP/EUR close price data available")
                raise Exception("No GBP/EUR data available")

            # Normalize timezones to ensure proper alignment
            if usdeur_close.index.tz is not None:
                usdeur_close_naive = usdeur_close.tz_localize(None)
            else:
                usdeur_close_naive = usdeur_close

            if gbpeur_close.index.tz is not None:
                gbpeur_close_naive = gbpeur_close.tz_localize(None)
            else:
                gbpeur_close_naive = gbpeur_close

            # Create combined DataFrame with aligned dates
            # Use outer join to get all available dates from both series
            result_df = pd.DataFrame(
                {
                    "USD_EUR": usdeur_close_naive,
                    "GBP_EUR": gbpeur_close_naive,
                }
            )

            # Forward fill missing values for alignment
            # (currency markets may have different holidays)
            result_df = result_df.ffill().dropna()

            if result_df.empty:
                logger.error("No overlapping currency data after alignment")
                raise Exception("No overlapping currency data available")

            # Filter by base_date for display purposes
            base_date_pd = pd.to_datetime(base_date.date())
            display_data = result_df[result_df.index >= base_date_pd]

            if display_data.empty:
                logger.warning(
                    f"No currency data after base_date {base_date} for display"
                )
                # Return empty result but don't error - this is just a display filter
                display_data = pd.DataFrame(columns=["USD_EUR", "GBP_EUR"])

            logger.info(
                f"Currency processing completed: {len(display_data)} data points "
                f"from {base_date}"
            )

            return StopEvent(
                result={
                    "data": display_data,
                    "base_date": base_date,
                    "latest_usdeur": (
                        display_data["USD_EUR"].iloc[-1]
                        if not display_data.empty
                        else None
                    ),
                    "latest_gbpeur": (
                        display_data["GBP_EUR"].iloc[-1]
                        if not display_data.empty
                        else None
                    ),
                    "data_points": len(display_data),
                }
            )

        except Exception as e:
            logger.error(f"Error processing currency data: {e}")
            # Re-raise as WorkflowException for better handling
            raise WorkflowException(
                workflow="CurrencyWorkflow",
                step="process_currency_data",
                message=f"Currency data processing failed: {e}",
                user_message="Failed to process currency data. Please try again later.",
                context={"base_date": str(base_date)},
            ) from e


class PreciousMetalsWorkflow(Workflow):
    """
    Workflow that fetches precious metals (Gold Futures) data.

    This workflow fetches gold futures data from Yahoo Finance using the GC=F ticker
    (COMEX Gold Futures). Gold is a key commodity and safe-haven asset.

    The workflow:
    - Fetches daily gold futures data from Yahoo Finance
    - Calculates 50-day and 200-day moving averages for trend analysis
    - Applies base_date filtering for display
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_gold_data(self, ev: StartEvent) -> PreciousMetalsEvent:
        """
        Fetch gold futures data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            PreciousMetalsEvent with gold data
        """
        base_date = ev.base_date

        logger.debug(f"PreciousMetalsWorkflow: Fetching gold data from {base_date}")

        # Fetch Gold Futures data (COMEX) using helper function
        gold_result = await self.yahoo_provider.get_data("GC=F")
        gold_data = validate_single_provider_task(gold_result, "Gold")

        return PreciousMetalsEvent(gold_data=gold_data, base_date=base_date)

    @step
    async def process_gold_data(self, ev: PreciousMetalsEvent) -> StopEvent:
        """
        Process gold data and calculate statistics.

        Args:
            ev: PreciousMetalsEvent with gold data

        Returns:
            StopEvent with processed gold data and statistics
        """
        gold_data = ev.gold_data
        base_date = ev.base_date

        logger.debug("PreciousMetalsWorkflow: Processing gold data")

        try:
            # Extract close prices from gold data
            if "Close" in gold_data.columns:
                gold_close = gold_data["Close"].dropna()
            elif "Adj Close" in gold_data.columns:
                gold_close = gold_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in gold data")
                raise Exception("No Close price data available for gold")

            if gold_close.empty:
                logger.error("No gold close price data available")
                raise Exception("No gold data available")

            # Normalize timezone to ensure proper filtering
            if gold_close.index.tz is not None:
                gold_close_naive = gold_close.tz_localize(None)
            else:
                gold_close_naive = gold_close

            # Calculate moving averages on full dataset
            moving_avg_50 = gold_close_naive.rolling(window=50, min_periods=50).mean()
            moving_avg_200 = gold_close_naive.rolling(
                window=200, min_periods=200
            ).mean()

            # Create complete result DataFrame
            result_df = pd.DataFrame(
                {
                    "Gold": gold_close_naive,
                    "Gold_MA50": moving_avg_50,
                    "Gold_MA200": moving_avg_200,
                },
                index=gold_close_naive.index,
            )

            # Filter by base_date for display purposes
            base_date_pd = pd.to_datetime(base_date.date())
            display_data = result_df[result_df.index >= base_date_pd]

            if display_data.empty:
                logger.warning(f"No gold data after base_date {base_date} for display")
                # Return empty result but don't error - this is just a display filter
                display_data = pd.DataFrame(columns=["Gold", "Gold_MA50", "Gold_MA200"])

            logger.info(
                f"Gold processing completed: {len(display_data)} data points "
                f"from {base_date}"
            )

            return StopEvent(
                result={
                    "data": display_data,
                    "base_date": base_date,
                    "latest_value": (
                        display_data["Gold"].iloc[-1]
                        if not display_data.empty
                        else None
                    ),
                    "data_points": len(display_data),
                }
            )

        except Exception as e:
            logger.error(f"Error processing gold data: {e}")
            # Re-raise as WorkflowException for better handling
            raise WorkflowException(
                workflow="PreciousMetalsWorkflow",
                step="process_gold_data",
                message=f"Gold data processing failed: {e}",
                user_message="Failed to process gold data. Please try again later.",
                context={"base_date": str(base_date)},
            ) from e


class CryptoCurrencyWorkflow(Workflow):
    """
    Workflow that fetches cryptocurrency (Bitcoin and Ethereum) data.
    This workflow fetches Bitcoin and Ethereum price data from Yahoo Finance using
    BTC-USD and ETH-USD tickers. These are major cryptocurrencies that provide
    insight into the digital asset market.
    The workflow:
    - Fetches daily Bitcoin and Ethereum data from Yahoo Finance in parallel
    - Applies base_date filtering for display
    - No moving averages or trend lines per user requirements
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_crypto_data(self, ev: StartEvent) -> CryptoCurrencyEvent:
        """
        Fetch Bitcoin and Ethereum data from Yahoo Finance in parallel.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            CryptoCurrencyEvent with Bitcoin and Ethereum data
        """
        base_date = ev.base_date
        logger.debug(f"CryptoCurrencyWorkflow: Fetching crypto data from {base_date}")

        # Fetch Bitcoin and Ethereum data in parallel using helper function
        tasks = {
            "Bitcoin": self.yahoo_provider.get_data("BTC-USD"),
            "Ethereum": self.yahoo_provider.get_data("ETH-USD"),
        }

        # Execute tasks in parallel using helper function
        results = await process_multiple_provider_results(tasks)

        # Extract data from results, checking for failures
        if not results["Bitcoin"]["success"]:
            raise Exception(f"Bitcoin data fetch failed: {results['Bitcoin']['error']}")
        if not results["Ethereum"]["success"]:
            raise Exception(
                f"Ethereum data fetch failed: {results['Ethereum']['error']}"
            )

        bitcoin_data = results["Bitcoin"]["data"]
        ethereum_data = results["Ethereum"]["data"]

        return CryptoCurrencyEvent(
            bitcoin_data=bitcoin_data, ethereum_data=ethereum_data, base_date=base_date
        )

    @step
    async def process_crypto_data(self, ev: CryptoCurrencyEvent) -> StopEvent:
        """
        Process cryptocurrency data and prepare for display.

        Args:
            ev: CryptoCurrencyEvent with Bitcoin and Ethereum data

        Returns:
            StopEvent with processed crypto data
        """
        bitcoin_data = ev.bitcoin_data
        ethereum_data = ev.ethereum_data
        base_date = ev.base_date

        logger.debug("CryptoCurrencyWorkflow: Processing crypto data")

        try:
            # Extract close prices from Bitcoin data
            if "Close" in bitcoin_data.columns:
                bitcoin_close = bitcoin_data["Close"].dropna()
            elif "Adj Close" in bitcoin_data.columns:
                bitcoin_close = bitcoin_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in Bitcoin data")
                raise Exception("No Close price data available for Bitcoin")

            if bitcoin_close.empty:
                logger.error("No Bitcoin close price data available")
                raise Exception("No Bitcoin data available")

            # Extract close prices from Ethereum data
            if "Close" in ethereum_data.columns:
                ethereum_close = ethereum_data["Close"].dropna()
            elif "Adj Close" in ethereum_data.columns:
                ethereum_close = ethereum_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in Ethereum data")
                raise Exception("No Close price data available for Ethereum")

            if ethereum_close.empty:
                logger.error("No Ethereum close price data available")
                raise Exception("No Ethereum data available")

            # Normalize timezone to ensure proper filtering
            if bitcoin_close.index.tz is not None:
                bitcoin_close_naive = bitcoin_close.tz_localize(None)
            else:
                bitcoin_close_naive = bitcoin_close

            if ethereum_close.index.tz is not None:
                ethereum_close_naive = ethereum_close.tz_localize(None)
            else:
                ethereum_close_naive = ethereum_close

            # Combine data into single DataFrame for comparison chart
            # Align data to common dates
            common_dates = bitcoin_close_naive.index.intersection(
                ethereum_close_naive.index
            )

            if common_dates.empty:
                logger.warning("No common dates between Bitcoin and Ethereum data")
                result_df = pd.DataFrame(columns=["BTC", "ETH"])
            else:
                result_df = pd.DataFrame(
                    {
                        "BTC": bitcoin_close_naive.reindex(common_dates),
                        "ETH": ethereum_close_naive.reindex(common_dates),
                    },
                    index=common_dates,
                )

            # Filter by base_date for display purposes
            base_date_pd = pd.to_datetime(base_date.date())
            display_data = result_df[result_df.index >= base_date_pd]

            if display_data.empty:
                logger.warning(
                    f"No crypto data after base_date {base_date} for display"
                )
                # Return empty result but don't error - this is just a display filter
                display_data = pd.DataFrame(columns=["BTC", "ETH"])

            logger.info(
                f"Crypto processing completed: {len(display_data)} data points "
                f"from {base_date}"
            )

            return StopEvent(
                result={
                    "data": display_data,
                    "base_date": base_date,
                    "latest_btc": (
                        display_data["BTC"].iloc[-1] if not display_data.empty else None
                    ),
                    "latest_eth": (
                        display_data["ETH"].iloc[-1] if not display_data.empty else None
                    ),
                    "data_points": len(display_data),
                }
            )

        except Exception as e:
            logger.error(f"Error processing crypto data: {e}")
            # Re-raise as WorkflowException for better handling
            raise WorkflowException(
                workflow="CryptoCurrencyWorkflow",
                step="process_crypto_data",
                message=f"Crypto data processing failed: {e}",
                user_message="Failed to process crypto data. Please try again later.",
                context={"base_date": str(base_date)},
            ) from e


class CrudeOilWorkflow(Workflow):
    """
    Workflow that fetches crude oil (WTI and Brent) price data.
    This workflow fetches crude oil price data from Yahoo Finance using
    CL=F and BZ=F tickers. These are the major crude oil benchmarks.
    The workflow:
    - Fetches daily WTI and Brent data from Yahoo Finance in parallel
    - Applies base_date filtering for display
    - No moving averages or trend lines per user requirements
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_crude_oil_data(self, ev: StartEvent) -> CrudeOilEvent:
        """
        Fetch WTI and Brent crude oil data from Yahoo Finance in parallel.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            CrudeOilEvent with WTI and Brent data
        """
        base_date = ev.base_date
        logger.debug(f"CrudeOilWorkflow: Fetching crude oil data from {base_date}")

        # Fetch WTI and Brent data in parallel using helper function
        tasks = {
            "WTI": self.yahoo_provider.get_data("CL=F"),  # WTI Crude Oil
            "Brent": self.yahoo_provider.get_data("BZ=F"),  # Brent Crude Oil
        }

        results = await process_multiple_provider_results(tasks)

        # Check for failures and extract data
        if not results["WTI"]["success"]:
            raise Exception(f"WTI data fetch failed: {results['WTI']['error']}")
        if not results["Brent"]["success"]:
            raise Exception(f"Brent data fetch failed: {results['Brent']['error']}")

        wti_data = results["WTI"]["data"]
        brent_data = results["Brent"]["data"]

        return CrudeOilEvent(
            wti_data=wti_data, brent_data=brent_data, base_date=base_date
        )

    @step
    async def process_crude_oil_data(self, ev: CrudeOilEvent) -> StopEvent:
        """
        Process crude oil data and prepare for display.

        Args:
            ev: CrudeOilEvent with WTI and Brent data

        Returns:
            StopEvent with processed crude oil data
        """
        wti_data = ev.wti_data
        brent_data = ev.brent_data
        base_date = ev.base_date

        logger.debug("CrudeOilWorkflow: Processing crude oil data")

        try:
            # Extract close prices from WTI data
            if "Close" in wti_data.columns:
                wti_close = wti_data["Close"].dropna()
            elif "Adj Close" in wti_data.columns:
                wti_close = wti_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in WTI data")
                raise Exception("No Close price data available for WTI")

            if wti_close.empty:
                logger.error("No WTI close price data available")
                raise Exception("No WTI data available")

            # Extract close prices from Brent data
            if "Close" in brent_data.columns:
                brent_close = brent_data["Close"].dropna()
            elif "Adj Close" in brent_data.columns:
                brent_close = brent_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in Brent data")
                raise Exception("No Close price data available for Brent")

            if brent_close.empty:
                logger.error("No Brent close price data available")
                raise Exception("No Brent data available")

            # Normalize timezone to ensure proper filtering
            if wti_close.index.tz is not None:
                wti_close_naive = wti_close.tz_localize(None)
            else:
                wti_close_naive = wti_close

            if brent_close.index.tz is not None:
                brent_close_naive = brent_close.tz_localize(None)
            else:
                brent_close_naive = brent_close

            # Combine data into single DataFrame for comparison chart
            # Align data to common dates
            common_dates = wti_close_naive.index.intersection(brent_close_naive.index)

            if common_dates.empty:
                logger.warning("No common dates between WTI and Brent data")
                result_df = pd.DataFrame(columns=["WTI", "Brent"])
            else:
                result_df = pd.DataFrame(
                    {
                        "WTI": wti_close_naive.reindex(common_dates),
                        "Brent": brent_close_naive.reindex(common_dates),
                    },
                    index=common_dates,
                )

            # Filter by base_date for display purposes
            base_date_pd = pd.to_datetime(base_date.date())
            display_data = result_df[result_df.index >= base_date_pd]

            if display_data.empty:
                logger.warning(
                    f"No crude oil data after base_date {base_date} for display"
                )
                # Return empty result but don't error - this is just a display filter
                display_data = pd.DataFrame(columns=["WTI", "Brent"])

            logger.info(
                f"Crude oil processing completed: {len(display_data)} data points "
                f"from {base_date}"
            )

            return StopEvent(
                result={
                    "data": display_data,
                    "base_date": base_date,
                    "latest_wti": (
                        display_data["WTI"].iloc[-1] if not display_data.empty else None
                    ),
                    "latest_brent": (
                        display_data["Brent"].iloc[-1]
                        if not display_data.empty
                        else None
                    ),
                    "data_points": len(display_data),
                }
            )

        except Exception as e:
            logger.error(f"Error processing crude oil data: {e}")
            # Re-raise as WorkflowException for better handling
            raise WorkflowException(
                workflow="CrudeOilWorkflow",
                step="process_crude_oil_data",
                message=f"Crude oil data processing failed: {e}",
                user_message=(
                    "Failed to process crude oil data. Please try again later."
                ),
                context={"base_date": str(base_date)},
            ) from e


class BloombergCommodityWorkflow(Workflow):
    """
    Workflow that fetches Bloomberg Commodity Index (^BCOM) data.

    This workflow fetches the Bloomberg Commodity Index from Yahoo Finance using
    the ^BCOM ticker. The index tracks the performance of a diversified basket of
    commodity futures contracts.

    The workflow:
    - Fetches daily ^BCOM data from Yahoo Finance
    - Calculates 50-day and 200-day moving averages on full dataset
    - Applies base_date filtering for display
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_bcom_data(self, ev: StartEvent) -> BloombergCommodityEvent:
        """
        Fetch Bloomberg Commodity Index data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            BloombergCommodityEvent with ^BCOM data
        """
        base_date = ev.base_date
        logger.debug(
            f"BloombergCommodityWorkflow: Fetching ^BCOM data from {base_date}"
        )

        # Fetch Bloomberg Commodity Index data using helper function
        bcom_result = await self.yahoo_provider.get_data("^BCOM")
        bcom_data = validate_single_provider_task(bcom_result, "^BCOM")

        return BloombergCommodityEvent(bcom_data=bcom_data, base_date=base_date)

    @step
    async def process_bcom_data(self, ev: BloombergCommodityEvent) -> StopEvent:
        """
        Process Bloomberg Commodity Index data and calculate moving averages.

        Args:
            ev: BloombergCommodityEvent with ^BCOM data

        Returns:
            StopEvent with processed ^BCOM data and moving averages
        """
        bcom_data = ev.bcom_data
        base_date = ev.base_date

        logger.debug("BloombergCommodityWorkflow: Processing ^BCOM data")

        try:
            # Extract close prices from ^BCOM data
            if "Close" in bcom_data.columns:
                bcom_close = bcom_data["Close"].dropna()
            elif "Adj Close" in bcom_data.columns:
                bcom_close = bcom_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in ^BCOM data")
                raise Exception("No Close price data available for ^BCOM")

            if bcom_close.empty:
                logger.error("No ^BCOM close price data available")
                raise Exception("No ^BCOM data available")

            # Normalize timezone to ensure proper filtering
            if bcom_close.index.tz is not None:
                bcom_close_naive = bcom_close.tz_localize(None)
            else:
                bcom_close_naive = bcom_close

            # Calculate moving averages on full dataset
            moving_avg_50 = bcom_close_naive.rolling(window=50, min_periods=50).mean()
            moving_avg_200 = bcom_close_naive.rolling(
                window=200, min_periods=200
            ).mean()

            # Create complete result DataFrame
            result_df = pd.DataFrame(
                {
                    "BCOM": bcom_close_naive,
                    "BCOM_MA50": moving_avg_50,
                    "BCOM_MA200": moving_avg_200,
                },
                index=bcom_close_naive.index,
            )

            # Filter by base_date for display purposes
            base_date_pd = pd.to_datetime(base_date.date())
            display_data = result_df[result_df.index >= base_date_pd]

            if display_data.empty:
                logger.warning(f"No ^BCOM data after base_date {base_date} for display")
                # Return empty result but don't error - this is just a display filter
                display_data = pd.DataFrame(columns=["BCOM", "BCOM_MA50", "BCOM_MA200"])

            logger.info(
                f"Bloomberg Commodity processing completed: {len(display_data)} "
                f"data points from {base_date}"
            )

            return StopEvent(
                result={
                    "data": display_data,
                    "base_date": base_date,
                    "latest_value": (
                        display_data["BCOM"].iloc[-1]
                        if not display_data.empty
                        else None
                    ),
                    "data_points": len(display_data),
                }
            )

        except Exception as e:
            logger.error(f"Error processing ^BCOM data: {e}")
            # Re-raise as WorkflowException for better handling
            raise WorkflowException(
                workflow="BloombergCommodityWorkflow",
                step="process_bcom_data",
                message=f"Bloomberg Commodity data processing failed: {e}",
                user_message=(
                    "Failed to process Bloomberg Commodity Index data. "
                    "Please try again later."
                ),
                context={"base_date": str(base_date)},
            ) from e


class MSCIWorldWorkflow(Workflow):
    """
    Workflow that fetches MSCI World Index data.

    This workflow fetches MSCI World Index data from Yahoo Finance using
    the ^990100-USD-STRD ticker. The MSCI World Index is a broad global
    equity index that captures large and mid cap representation across
    23 Developed Markets countries.

    The workflow:
    - Fetches daily MSCI World Index data from Yahoo Finance
    - Calculates 50-day and 200-day moving averages on full dataset
    - Calculates Bollinger Bands (20-day MA ± 2 std dev)
    - Applies base_date filtering for display
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_msci_data(self, ev: StartEvent) -> MSCIWorldEvent:
        """
        Fetch MSCI World Index data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            MSCIWorldEvent with MSCI World data
        """
        base_date = ev.base_date
        logger.debug(f"MSCIWorldWorkflow: Fetching MSCI World data from {base_date}")

        # Fetch MSCI World Index data using helper function
        msci_result = await self.yahoo_provider.get_data("^990100-USD-STRD")
        msci_data = validate_single_provider_task(msci_result, "MSCI World")

        return MSCIWorldEvent(msci_data=msci_data, base_date=base_date)

    @step
    async def process_msci_data(self, ev: MSCIWorldEvent) -> StopEvent:
        """
        Process MSCI World data and calculate technical indicators.

        Args:
            ev: MSCIWorldEvent with MSCI World data

        Returns:
            StopEvent with processed MSCI World data and indicators
        """
        msci_data = ev.msci_data
        base_date = ev.base_date

        logger.debug("MSCIWorldWorkflow: Processing MSCI World data")

        try:
            # Extract close prices from MSCI World data
            if "Close" in msci_data.columns:
                msci_close = msci_data["Close"].dropna()
            elif "Adj Close" in msci_data.columns:
                msci_close = msci_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in MSCI World data")
                raise Exception("No Close price data available for MSCI World")

            if msci_close.empty:
                logger.error("No MSCI World close price data available")
                raise Exception("No MSCI World data available")

            # Normalize timezone to ensure proper filtering
            if msci_close.index.tz is not None:
                msci_close_naive = msci_close.tz_localize(None)
            else:
                msci_close_naive = msci_close

            # Calculate moving averages on full dataset
            moving_avg_50 = msci_close_naive.rolling(window=50, min_periods=50).mean()
            moving_avg_200 = msci_close_naive.rolling(
                window=200, min_periods=200
            ).mean()

            # Calculate Bollinger Bands (20-day MA ± 2 standard deviations)
            rolling_mean = msci_close_naive.rolling(window=20, min_periods=20).mean()
            rolling_std = msci_close_naive.rolling(window=20, min_periods=20).std()
            bollinger_upper = rolling_mean + (rolling_std * 2)
            bollinger_lower = rolling_mean - (rolling_std * 2)

            # Create complete result DataFrame
            result_df = pd.DataFrame(
                {
                    "MSCI_World": msci_close_naive,
                    "MSCI_MA50": moving_avg_50,
                    "MSCI_MA200": moving_avg_200,
                    "MSCI_BB_Upper": bollinger_upper,
                    "MSCI_BB_Lower": bollinger_lower,
                    "MSCI_BB_Mid": rolling_mean,
                },
                index=msci_close_naive.index,
            )

            # Filter by base_date for display purposes
            base_date_pd = pd.to_datetime(base_date.date())
            display_data = result_df[result_df.index >= base_date_pd]

            if display_data.empty:
                logger.warning(
                    f"No MSCI World data after base_date {base_date} for display"
                )
                # Return empty result but don't error - this is just a display filter
                display_data = pd.DataFrame(
                    columns=[
                        "MSCI_World",
                        "MSCI_MA50",
                        "MSCI_MA200",
                        "MSCI_BB_Upper",
                        "MSCI_BB_Lower",
                        "MSCI_BB_Mid",
                    ]
                )

            logger.info(
                f"MSCI World processing completed: {len(display_data)} data points "
                f"from {base_date}"
            )

            return StopEvent(
                result={
                    "data": display_data,
                    "base_date": base_date,
                    "latest_value": (
                        display_data["MSCI_World"].iloc[-1]
                        if not display_data.empty
                        else None
                    ),
                    "data_points": len(display_data),
                }
            )

        except Exception as e:
            logger.error(f"Error processing MSCI World data: {e}")
            # Re-raise as WorkflowException for better handling
            raise WorkflowException(
                workflow="MSCIWorldWorkflow",
                step="process_msci_data",
                message=f"MSCI World data processing failed: {e}",
                user_message=(
                    "Failed to process MSCI World data. Please try again later."
                ),
                context={"base_date": str(base_date)},
            ) from e


@apply_flow_cache
async def fetch_yield_curve_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process US Treasury yield curve data.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with yield curve data
        - base_date: The base date used
        - latest_date: Date of most recent data
        - maturities: List of available maturities
        - data_points: Number of data points
    """
    try:
        logger.info(f"Starting yield curve data fetch from {base_date}")

        # Create and run workflow
        workflow = YieldCurveWorkflow()
        result = await workflow.run(base_date=base_date)

        logger.info("Yield curve workflow completed successfully")

        # Extract result data from workflow result
        if hasattr(result, "result"):
            return result.result
        else:
            logger.warning(
                "Yield curve workflow result missing .result attribute, "
                "returning directly"
            )
            return result

    except Exception as e:
        logger.error(f"Yield curve workflow failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_yield_curve_data",
            step="workflow_execution",
            message=f"Yield curve workflow execution failed: {e}",
            user_message=(
                "Failed to fetch yield curve data due to a system error. "
                "Please try again."
            ),
            context={"base_date": str(base_date)},
        ) from e


@apply_flow_cache
async def fetch_vix_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process VIX data.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with VIX values
        - base_date: The base date used
        - historical_mean: Historical mean VIX value
        - latest_value: Most recent VIX value
        - data_points: Number of data points
    """
    try:
        logger.info(f"Starting VIX data fetch from {base_date}")

        # Create and run workflow
        workflow = VIXWorkflow()
        result = await workflow.run(base_date=base_date)

        logger.info("VIX workflow completed successfully")

        # Extract result data from workflow result
        if hasattr(result, "result"):
            return result.result
        else:
            logger.warning(
                "VIX workflow result missing .result attribute, returning directly"
            )
            return result

    except Exception as e:
        logger.error(f"VIX workflow failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_vix_data",
            step="workflow_execution",
            message=f"VIX workflow execution failed: {e}",
            user_message=(
                "Failed to fetch VIX data due to a system error. Please try again."
            ),
            context={"base_date": str(base_date)},
        ) from e


@apply_flow_cache
async def fetch_currency_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process currency exchange rate data.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with USD/EUR and GBP/EUR rates
        - base_date: The base date used
        - latest_usdeur: Most recent USD/EUR rate
        - latest_gbpeur: Most recent GBP/EUR rate
        - data_points: Number of data points
    """
    try:
        logger.info(f"Starting currency data fetch from {base_date}")

        # Create and run workflow
        workflow = CurrencyWorkflow()
        result = await workflow.run(base_date=base_date)

        logger.info("Currency workflow completed successfully")

        # Extract result data from workflow result
        if hasattr(result, "result"):
            return result.result
        else:
            logger.warning(
                "Currency workflow result missing .result attribute, returning directly"
            )
            return result

    except Exception as e:
        logger.error(f"Currency workflow failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_currency_data",
            step="workflow_execution",
            message=f"Currency workflow execution failed: {e}",
            user_message=(
                "Failed to fetch currency data due to a system error. "
                "Please try again."
            ),
            context={"base_date": str(base_date)},
        ) from e


@apply_flow_cache
async def fetch_precious_metals_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process precious metals (Gold Futures) data.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with gold values and moving average
        - base_date: The base date used
        - latest_value: Most recent gold price
        - data_points: Number of data points
    """
    try:
        logger.info(f"Starting precious metals data fetch from {base_date}")

        # Create and run workflow
        workflow = PreciousMetalsWorkflow()
        result = await workflow.run(base_date=base_date)

        logger.info("Precious metals workflow completed successfully")

        # Extract result data from workflow result
        if hasattr(result, "result"):
            return result.result
        else:
            logger.warning(
                "Precious metals workflow result missing .result attribute, "
                "returning directly"
            )
            return result

    except Exception as e:
        logger.error(f"Precious metals workflow failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_precious_metals_data",
            step="workflow_execution",
            message=f"Precious metals workflow execution failed: {e}",
            user_message=(
                "Failed to fetch precious metals data due to a system error. "
                "Please try again."
            ),
            context={"base_date": str(base_date)},
        ) from e


@apply_flow_cache
async def fetch_crypto_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process cryptocurrency (Bitcoin and Ethereum) data.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with Bitcoin and Ethereum values
        - base_date: The base date used
        - latest_btc: Most recent Bitcoin price
        - latest_eth: Most recent Ethereum price
        - data_points: Number of data points
    """
    try:
        logger.info(f"Starting cryptocurrency data fetch from {base_date}")

        # Create and run workflow
        workflow = CryptoCurrencyWorkflow()
        result = await workflow.run(base_date=base_date)

        logger.info("Cryptocurrency workflow completed successfully")

        # Extract result data from workflow result
        if hasattr(result, "result"):
            return result.result
        else:
            logger.warning(
                "Cryptocurrency workflow result missing .result attribute, "
                "returning directly"
            )
            return result

    except Exception as e:
        logger.error(f"Cryptocurrency workflow failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_crypto_data",
            step="workflow_execution",
            message=f"Cryptocurrency workflow execution failed: {e}",
            user_message=(
                "Failed to fetch cryptocurrency data due to a system error. "
                "Please try again."
            ),
            context={"base_date": str(base_date)},
        ) from e


@apply_flow_cache
async def fetch_crude_oil_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process crude oil (WTI and Brent) data.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with WTI and Brent values
        - base_date: The base date used
        - latest_wti: Most recent WTI price
        - latest_brent: Most recent Brent price
        - data_points: Number of data points
    """
    try:
        logger.info(f"Starting crude oil data fetch from {base_date}")

        # Create and run workflow
        workflow = CrudeOilWorkflow()
        result = await workflow.run(base_date=base_date)

        logger.info("Crude oil workflow completed successfully")

        # Extract result data from workflow result
        if hasattr(result, "result"):
            return result.result
        else:
            logger.warning(
                "Crude oil workflow result missing .result attribute, "
                "returning directly"
            )
            return result

    except Exception as e:
        logger.error(f"Crude oil workflow failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_crude_oil_data",
            step="workflow_execution",
            message=f"Crude oil workflow execution failed: {e}",
            user_message=(
                "Failed to fetch crude oil data due to a system error. "
                "Please try again."
            ),
            context={"base_date": str(base_date)},
        ) from e


@apply_flow_cache
async def fetch_bloomberg_commodity_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process Bloomberg Commodity Index (^BCOM) data.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with BCOM values and moving averages
        - base_date: The base date used
        - latest_value: Most recent BCOM value
        - data_points: Number of data points
    """
    try:
        logger.info(f"Starting Bloomberg Commodity data fetch from {base_date}")

        # Create and run workflow
        workflow = BloombergCommodityWorkflow()
        result = await workflow.run(base_date=base_date)

        logger.info("Bloomberg Commodity workflow completed successfully")

        # Extract result data from workflow result
        if hasattr(result, "result"):
            return result.result
        else:
            logger.warning(
                "Bloomberg Commodity workflow result missing .result attribute, "
                "returning directly"
            )
            return result

    except Exception as e:
        logger.error(f"Bloomberg Commodity workflow failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_bloomberg_commodity_data",
            step="workflow_execution",
            message=f"Bloomberg Commodity workflow execution failed: {e}",
            user_message=(
                "Failed to fetch Bloomberg Commodity Index data due to a system error. "
                "Please try again."
            ),
            context={"base_date": str(base_date)},
        ) from e


@apply_flow_cache
async def fetch_msci_world_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process MSCI World Index data.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with MSCI World values, moving averages,
          and Bollinger bands
        - base_date: The base date used
        - latest_value: Most recent MSCI World value
        - data_points: Number of data points
    """
    try:
        logger.info(f"Starting MSCI World data fetch from {base_date}")

        # Create and run workflow
        workflow = MSCIWorldWorkflow()
        result = await workflow.run(base_date=base_date)

        logger.info("MSCI World workflow completed successfully")

        # Extract result data from workflow result
        if hasattr(result, "result"):
            return result.result
        else:
            logger.warning(
                "MSCI World workflow result missing .result attribute, "
                "returning directly"
            )
            return result

    except Exception as e:
        logger.error(f"MSCI World workflow failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_msci_world_data",
            step="workflow_execution",
            message=f"MSCI World workflow execution failed: {e}",
            user_message=(
                "Failed to fetch MSCI World data due to a system error. "
                "Please try again."
            ),
            context={"base_date": str(base_date)},
        ) from e
