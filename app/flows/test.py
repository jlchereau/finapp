"""
Enhanced financial data workflow using improved providers.

This workflow demonstrates how to use the new async provider architecture
with comprehensive error handling and parallel data fetching.
"""

import asyncio
from typing import Dict, Any, List, Optional
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from app.providers.yahoo import (
    create_yahoo_history_provider,
    create_yahoo_info_provider,
)
from app.providers.zacks import create_zacks_provider
from app.providers.base import ProviderResult


class DataFetchedEvent(Event):
    """Event emitted when all provider data is fetched."""

    ticker: str
    yahoo_history: Optional[ProviderResult] = None
    yahoo_info: Optional[ProviderResult] = None
    zacks_data: Optional[ProviderResult] = None


class FinancialDataWorkflow(Workflow):
    """
    Workflow that fetches comprehensive financial data for a ticker.

    This workflow demonstrates:
    - Parallel data fetching from multiple providers
    - Comprehensive error handling
    - Proper use of async/await
    - Structured result aggregation
    """

    def __init__(self):
        """Initialize workflow with configured providers."""
        super().__init__()

        # Create providers with custom configurations
        self.yahoo_history = create_yahoo_history_provider(
            period="1y", timeout=30.0, retries=2
        )

        self.yahoo_info = create_yahoo_info_provider(timeout=30.0, retries=2)

        self.zacks = create_zacks_provider(
            timeout=20.0, retries=1, rate_limit=0.5  # 0.5 requests per second
        )

    @step
    async def fetch_all_data(self, ev: StartEvent) -> DataFetchedEvent:
        """
        First step: Fetch data from all providers in parallel.

        This demonstrates how to use the new provider architecture
        with proper error handling and parallel execution.
        """
        ticker = ev.ticker

        # Create tasks for parallel execution
        tasks = {
            "yahoo_history": self.yahoo_history.get_data(ticker, period="6mo"),
            "yahoo_info": self.yahoo_info.get_data(ticker),
            "zacks_data": self.zacks.get_data(ticker),
        }

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Map results back to named fields
        result_dict = dict(zip(tasks.keys(), results))

        # Handle any exceptions (shouldn't happen with new error handling)
        for key, result in result_dict.items():
            if isinstance(result, Exception):
                # Create a failed ProviderResult for consistency
                from app.providers.base import ProviderType

                result_dict[key] = ProviderResult(
                    success=False,
                    error_message=f"Unexpected workflow error: {str(result)}",
                    error_code=type(result).__name__,
                    provider_type=ProviderType.CUSTOM,
                    ticker=ticker,
                )

        return DataFetchedEvent(
            ticker=ticker,
            yahoo_history=result_dict["yahoo_history"],
            yahoo_info=result_dict["yahoo_info"],
            zacks_data=result_dict["zacks_data"],
        )

    @step
    async def process_and_aggregate(self, ev: DataFetchedEvent) -> StopEvent:
        """
        Second step: Process and aggregate the fetched data.

        This step demonstrates how to work with ProviderResult objects
        and create comprehensive summaries.
        """
        ticker = ev.ticker

        # Create comprehensive result summary
        summary = {
            "ticker": ticker,
            "timestamp": ev.yahoo_info.timestamp if ev.yahoo_info else None,
            "data_sources": {},
            "aggregated_data": {},
            "errors": [],
            "warnings": [],
        }

        # Process Yahoo History data
        if ev.yahoo_history and ev.yahoo_history.success:
            df = ev.yahoo_history.data
            summary["data_sources"]["yahoo_history"] = {
                "status": "success",
                "rows": len(df),
                "columns": list(df.columns),
                "date_range": {
                    "start": str(df.index.min()),
                    "end": str(df.index.max()),
                },
                "execution_time": ev.yahoo_history.execution_time,
            }

            # Add some aggregated metrics
            latest_data = df.iloc[-1]
            summary["aggregated_data"]["latest_price"] = float(latest_data["Close"])
            summary["aggregated_data"]["latest_volume"] = int(latest_data["Volume"])
            summary["aggregated_data"]["price_change_1d"] = float(
                latest_data["Close"] - df.iloc[-2]["Close"]
            )

            # Calculate some basic statistics
            summary["aggregated_data"]["volatility_6mo"] = float(
                df["Close"].pct_change().std() * (252**0.5)  # Annualized
            )
            summary["aggregated_data"]["avg_volume_6mo"] = float(df["Volume"].mean())
        else:
            summary["data_sources"]["yahoo_history"] = {
                "status": "failed",
                "error": (
                    ev.yahoo_history.error_message if ev.yahoo_history else "No data"
                ),
            }
            if ev.yahoo_history:
                summary["errors"].append(
                    f"Yahoo History: {ev.yahoo_history.error_message}"
                )

        # Process Yahoo Info data
        if ev.yahoo_info and ev.yahoo_info.success:
            info_data = ev.yahoo_info.data
            summary["data_sources"]["yahoo_info"] = {
                "status": "success",
                "execution_time": ev.yahoo_info.execution_time,
            }

            # Add fundamental data to aggregated results
            if hasattr(info_data, "company_name") and info_data.company_name:
                summary["aggregated_data"]["company_name"] = info_data.company_name
            if hasattr(info_data, "market_cap") and info_data.market_cap:
                summary["aggregated_data"]["market_cap"] = info_data.market_cap
            if hasattr(info_data, "pe_ratio") and info_data.pe_ratio:
                summary["aggregated_data"]["pe_ratio"] = info_data.pe_ratio
            if hasattr(info_data, "sector") and info_data.sector:
                summary["aggregated_data"]["sector"] = info_data.sector
        else:
            summary["data_sources"]["yahoo_info"] = {
                "status": "failed",
                "error": ev.yahoo_info.error_message if ev.yahoo_info else "No data",
            }
            if ev.yahoo_info:
                summary["errors"].append(f"Yahoo Info: {ev.yahoo_info.error_message}")

        # Process Zacks data
        if ev.zacks_data and ev.zacks_data.success:
            zacks_data = ev.zacks_data.data
            summary["data_sources"]["zacks"] = {
                "status": "success",
                "execution_time": ev.zacks_data.execution_time,
            }

            # Add Zacks-specific data
            if hasattr(zacks_data, "zacks_rank") and zacks_data.zacks_rank:
                summary["aggregated_data"]["zacks_rank"] = zacks_data.zacks_rank
            if hasattr(zacks_data, "vgm_score") and zacks_data.vgm_score:
                summary["aggregated_data"]["vgm_score"] = zacks_data.vgm_score
        else:
            summary["data_sources"]["zacks"] = {
                "status": "failed",
                "error": ev.zacks_data.error_message if ev.zacks_data else "No data",
            }
            if ev.zacks_data:
                summary["warnings"].append(f"Zacks: {ev.zacks_data.error_message}")

        # Add data quality assessment
        successful_sources = sum(
            1
            for source in summary["data_sources"].values()
            if source.get("status") == "success"
        )
        summary["data_quality"] = {
            "successful_sources": successful_sources,
            "total_sources": len(summary["data_sources"]),
            "completeness_score": successful_sources / len(summary["data_sources"]),
        }

        # Calculate total execution time
        total_time = 0
        for result in [ev.yahoo_history, ev.yahoo_info, ev.zacks_data]:
            if result and result.execution_time:
                total_time = max(total_time, result.execution_time)

        summary["total_execution_time"] = total_time

        return StopEvent(result=summary)


async def fetch_financial_data(ticker: str) -> Dict[str, Any]:
    """
    Main function to fetch comprehensive financial data for a ticker.

    This is the function that should be called from Reflex background events.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with comprehensive financial data and metadata
    """
    try:
        workflow = FinancialDataWorkflow()
        start_event = StartEvent(ticker=ticker.upper().strip())
        handler = workflow.run(start_event=start_event)
        result = await handler

        # Extract result data
        if hasattr(result, "result"):
            return result.result
        else:
            return result

    except (ValueError, ConnectionError, RuntimeError) as e:
        return {
            "ticker": ticker,
            "error": f"Workflow execution failed: {str(e)}",
            "data_sources": {},
            "aggregated_data": {},
            "errors": [str(e)],
            "warnings": [],
            "data_quality": {
                "successful_sources": 0,
                "total_sources": 3,
                "completeness_score": 0.0,
            },
        }


# Convenience function for multiple tickers
async def fetch_multiple_tickers(tickers: List[str]) -> Dict[str, Any]:
    """
    Fetch financial data for multiple tickers in parallel.

    Args:
        tickers: List of ticker symbols

    Returns:
        Dictionary mapping tickers to their financial data
    """
    tasks = [fetch_financial_data(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = {}
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            output[ticker] = {
                "ticker": ticker,
                "error": f"Failed to process ticker: {str(result)}",
                "data_sources": {},
                "aggregated_data": {},
                "errors": [str(result)],
                "warnings": [],
                "data_quality": {
                    "successful_sources": 0,
                    "total_sources": 3,
                    "completeness_score": 0.0,
                },
            }
        else:
            output[ticker] = result

    return output
