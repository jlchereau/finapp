"""
LlamaIndex workflow for cryptocurrency (Bitcoin and Ethereum) data collection.

This workflow handles fetching cryptocurrency price data using
YahooHistoryProvider with FlowRunner architecture.
"""

from typing import Dict, Any
from datetime import datetime

import pandas as pd
from workflows import Workflow, step, Context
from workflows.events import Event, StartEvent

from app.providers.yahoo import create_yahoo_history_provider
from app.flows.base import FlowRunner, FlowResultEvent
from app.lib.logger import logger
from app.lib.exceptions import FlowException
from app.flows.cache import apply_flow_cache


class DispatchEvent(Event):
    """Event to initiate parallel fetching of crypto data."""


class FetchBitcoinEvent(Event):
    """Event to initiate fetching of Bitcoin data from Yahoo."""

    base_date: datetime


class FetchEthereumEvent(Event):
    """Event to initiate fetching of Ethereum data from Yahoo."""

    base_date: datetime


class BitcoinResultEvent(Event):
    """Event containing result of Bitcoin data fetch."""

    data: pd.DataFrame | None
    error: str | None


class EthereumResultEvent(Event):
    """Event containing result of Ethereum data fetch."""

    data: pd.DataFrame | None
    error: str | None


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
    async def initiate_crypto_fetch(
        self, ctx: Context, ev: StartEvent
    ) -> FetchBitcoinEvent | FetchEthereumEvent:
        """
        Step 1: Dispatch - Send parallel fetch events for Bitcoin and Ethereum data.
        LlamaIndex automatically executes the sent events in parallel.

        Args:
        ctx: Workflow context for storing state and sending events
        ev.base_date: Start date for data fetching

        Returns:
        FetchBitcoinEvent (dummy event for LlamaIndex validation)
        """
        base_date = ev.base_date

        logger.debug(
            f"CryptoCurrencyWorkflow: Dispatching parallel crypto data fetch "
            f"from {base_date}"
        )

        # Store shared state for later steps
        await ctx.store.set("base_date", base_date)

        # Send events for parallel execution - LlamaIndex handles the parallelism
        ctx.send_event(FetchBitcoinEvent(base_date=base_date))
        ctx.send_event(FetchEthereumEvent(base_date=base_date))

        # Return dummy event to satisfy return type - LlamaIndex processes sent events
        return FetchBitcoinEvent(base_date=base_date)

    @step
    async def fetch_bitcoin_data(self, ev: FetchBitcoinEvent) -> BitcoinResultEvent:
        """
        Step 2: Bitcoin Processing - Fetch Bitcoin data from Yahoo Finance.
        This step runs in parallel with Ethereum data fetching automatically via
        LlamaIndex.

        Args:
        ev: FetchBitcoinEvent with fetch details

        Returns:
        BitcoinResultEvent with fetch result
        """
        # Mark event parameter as intentionally unused
        _ = ev
        logger.debug("Fetching Bitcoin data from Yahoo Finance")

        # Use standard provider interface - fetch full historical data
        provider_result = await self.yahoo_provider.get_data(
            query="BTC-USD",  # Bitcoin price in USD
        )

        # Handle ProviderResult properly
        if provider_result.success and provider_result.data is not None:
            data_df = provider_result.data
            if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                logger.debug(
                    f"Successfully fetched {len(data_df)} Bitcoin observations"
                )
                return BitcoinResultEvent(
                    data=data_df,
                    error=None,
                )
            else:
                logger.warning("Empty Bitcoin data returned")
                return BitcoinResultEvent(
                    data=None,
                    error="Empty data returned",
                )
        else:
            # Provider failed
            error_msg = provider_result.error_message or "Provider failed"
            logger.warning(f"Bitcoin provider failed: {error_msg}")
            return BitcoinResultEvent(
                data=None,
                error=error_msg,
            )

    @step
    async def fetch_ethereum_data(self, ev: FetchEthereumEvent) -> EthereumResultEvent:
        """
        Step 3: Ethereum Processing - Fetch Ethereum data from Yahoo Finance.
        This step runs in parallel with Bitcoin data fetching automatically via
        LlamaIndex.

        Args:
        ev: FetchEthereumEvent with fetch details

        Returns:
        EthereumResultEvent with fetch result
        """
        # Mark event parameter as intentionally unused
        _ = ev
        logger.debug("Fetching Ethereum data from Yahoo Finance")

        # Use standard provider interface - fetch full historical data
        provider_result = await self.yahoo_provider.get_data(
            query="ETH-USD",  # Ethereum price in USD
        )

        # Handle ProviderResult properly
        if provider_result.success and provider_result.data is not None:
            data_df = provider_result.data
            if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                logger.debug(
                    f"Successfully fetched {len(data_df)} Ethereum observations"
                )
                return EthereumResultEvent(
                    data=data_df,
                    error=None,
                )
            else:
                logger.warning("Empty Ethereum data returned")
                return EthereumResultEvent(
                    data=None,
                    error="Empty data returned",
                )
        else:
            # Provider failed
            error_msg = provider_result.error_message or "Provider failed"
            logger.warning(f"Ethereum provider failed: {error_msg}")
            return EthereumResultEvent(
                data=None,
                error=error_msg,
            )

    @step
    async def combine_crypto_data(
        self, ctx: Context, ev: BitcoinResultEvent | EthereumResultEvent
    ) -> FlowResultEvent | None:
        """
        Step 4: Combine - Combine all parallel results and create crypto
        comparison data.
        Uses collector pattern to wait for both Bitcoin and Ethereum results.

        Args:
        ctx: Workflow context for collecting events and accessing stored state
        ev: Either BitcoinResultEvent or EthereumResultEvent from parallel
        fetches

        Returns:
        FlowResultEvent with processed crypto data when both results
        collected, None otherwise
        """
        # Get the stored state from the dispatch step
        base_date = await ctx.store.get("base_date")

        # Collect events until we have both results (Bitcoin + Ethereum = 2 events)
        results = ctx.collect_events(ev, [BitcoinResultEvent, EthereumResultEvent])
        if results is None:
            # Not all results collected yet
            return None

        logger.debug(
            f"CryptoCurrencyWorkflow: Combining {len(results)} crypto data "
            f"source results"
        )

        # Separate and validate the results
        bitcoin_result = None
        ethereum_result = None

        for result in results:
            if isinstance(result, BitcoinResultEvent):
                bitcoin_result = result
            elif isinstance(result, EthereumResultEvent):
                ethereum_result = result

        # Check that we have both results - raise exceptions that FlowRunner will catch
        if bitcoin_result is None:
            raise FlowException(
                workflow="CryptoCurrencyWorkflow",
                step="combine_crypto_data",
                message="Bitcoin result not received",
            )
        if ethereum_result is None:
            raise FlowException(
                workflow="CryptoCurrencyWorkflow",
                step="combine_crypto_data",
                message="Ethereum result not received",
            )

        # Check for errors in the results - both are required for crypto comparison
        if bitcoin_result.error is not None:
            raise FlowException(
                workflow="CryptoCurrencyWorkflow",
                step="combine_crypto_data",
                message=f"Bitcoin data fetch failed: {bitcoin_result.error}",
            )
        if ethereum_result.error is not None:
            raise FlowException(
                workflow="CryptoCurrencyWorkflow",
                step="combine_crypto_data",
                message=f"Ethereum data fetch failed: {ethereum_result.error}",
            )

        # Extract the data
        bitcoin_data = bitcoin_result.data
        ethereum_data = ethereum_result.data

        logger.debug("CryptoCurrencyWorkflow: Processing crypto data")

        # Validate data is not None first - let exceptions bubble up
        if bitcoin_data is None:
            raise FlowException(
                workflow="CryptoCurrencyWorkflow",
                step="combine_crypto_data",
                message="Bitcoin data is missing",
            )
        if ethereum_data is None:
            raise FlowException(
                workflow="CryptoCurrencyWorkflow",
                step="combine_crypto_data",
                message="Ethereum data is missing",
            )

        logger.debug(
            f"Processing Bitcoin data: {len(bitcoin_data)} rows, "
            f"Ethereum data: {len(ethereum_data)} rows"
        )

        # Extract close prices from Bitcoin data
        if "Close" in bitcoin_data.columns:
            bitcoin_close = bitcoin_data["Close"].dropna()
        elif "Adj Close" in bitcoin_data.columns:
            bitcoin_close = bitcoin_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="CryptoCurrencyWorkflow",
                step="combine_crypto_data",
                message="No Close price data available for Bitcoin",
            )

        if bitcoin_close.empty:
            raise FlowException(
                workflow="CryptoCurrencyWorkflow",
                step="combine_crypto_data",
                message="No Bitcoin close price data available",
            )

        # Extract close prices from Ethereum data
        if "Close" in ethereum_data.columns:
            ethereum_close = ethereum_data["Close"].dropna()
        elif "Adj Close" in ethereum_data.columns:
            ethereum_close = ethereum_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="CryptoCurrencyWorkflow",
                step="combine_crypto_data",
                message="No Close price data available for Ethereum",
            )

        if ethereum_close.empty:
            raise FlowException(
                workflow="CryptoCurrencyWorkflow",
                step="combine_crypto_data",
                message="No Ethereum close price data available",
            )

        # Normalize timezone to ensure proper filtering
        if hasattr(bitcoin_close.index, "tz") and bitcoin_close.index.tz is not None:
            bitcoin_close_naive = bitcoin_close.tz_localize(None)
        else:
            bitcoin_close_naive = bitcoin_close

        if hasattr(ethereum_close.index, "tz") and ethereum_close.index.tz is not None:
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
            logger.warning(f"No crypto data after base_date {base_date} for display")
            # Return empty result but don't error - this is just a display filter
            display_data = pd.DataFrame(columns=["BTC", "ETH"])

        logger.info(
            f"Crypto processing completed: {len(display_data)} data points "
            f"from {base_date}"
        )

        return FlowResultEvent.success_result(
            data=display_data,
            base_date=base_date,
            metadata={
                "latest_btc": (
                    display_data["BTC"].iloc[-1] if not display_data.empty else None
                ),
                "latest_eth": (
                    display_data["ETH"].iloc[-1] if not display_data.empty else None
                ),
                "data_points": len(display_data),
            },
        )


@apply_flow_cache
async def fetch_crypto_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process cryptocurrency (Bitcoin and Ethereum) data using FlowRunner.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with Bitcoin and Ethereum values
        - base_date: The base date used
        - latest_btc: Most recent Bitcoin price
        - latest_eth: Most recent Ethereum price
        - data_points: Number of data points

    Raises:
        Exception: If workflow fails (for backward compatibility)
    """
    logger.info(f"Starting cryptocurrency data fetch from {base_date}")

    # Create workflow and FlowRunner
    workflow = CryptoCurrencyWorkflow()
    runner = FlowRunner[pd.DataFrame](workflow)

    # Run workflow using FlowRunner
    result_event = await runner.run(base_date=base_date)

    if result_event.success:
        logger.info("Cryptocurrency workflow completed successfully")

        # Convert FlowResultEvent back to dictionary format for backward compatibility
        metadata = result_event.metadata or {}

        return {
            "data": result_event.data,
            "base_date": result_event.base_date,
            "latest_btc": metadata.get("latest_btc"),
            "latest_eth": metadata.get("latest_eth"),
            "data_points": metadata.get("data_points", 0),
        }

    else:
        # Handle error case - raise exception for backward compatibility
        error_msg = result_event.error_message or "Unknown error"

        logger.error(f"Cryptocurrency workflow failed: {error_msg}")

        # Raise a standard exception for backward compatibility
        raise FlowException(
            workflow="CryptoCurrencyWorkflow",
            step="fetch_crypto_data",
            message=f"Cryptocurrency workflow failed: {error_msg}",
        )
