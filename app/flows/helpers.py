"""
Reusable helper functions for LlamaIndex workflows.

This module provides common utilities for handling provider results,
asyncio.gather operations, and data validation across workflows.
"""

# pylint: disable=broad-exception-raised

import asyncio
import time
from typing import Dict, Any, Tuple
from datetime import datetime
from deprecated.classic import deprecated

import pandas as pd

from app.providers.base import ProviderResult
from app.flows.base import FlowResult
from app.lib.logger import logger


@deprecated(reason="use methods available in @app/flows/base.py")
def validate_provider_result(
    result: ProviderResult[pd.DataFrame] | BaseException, data_name: str
) -> pd.DataFrame:
    """
    Validate a single provider result with consistent error handling.

    Args:
        result: Provider result or exception to validate
        data_name: Name of the data source for error messages

    Returns:
        Validated DataFrame from the provider result

    Raises:
        Exception: If result is invalid or contains no data
    """
    # Check for exceptions first
    if isinstance(result, Exception):
        logger.error(f"Failed to fetch {data_name} data: {result}")
        raise result

    # Check provider success status
    if not (hasattr(result, "success") and result.success):
        error_msg = getattr(result, "error_message", f"Unknown {data_name} fetch error")
        logger.error(f"{data_name} provider failed: {error_msg}")
        raise Exception(f"{data_name} data fetch failed: {error_msg}")

    # Check data attribute and type
    if not (hasattr(result, "data") and isinstance(result.data, pd.DataFrame)):
        error_msg = getattr(result, "error_message", f"Unknown {data_name} data error")
        logger.error(f"{data_name} provider returned invalid data: {error_msg}")
        raise Exception(f"{data_name} data fetch failed: {error_msg}")

    data = result.data

    # Check for empty data
    if data.empty:
        logger.error(f"Empty {data_name} data returned")
        raise Exception(f"No {data_name.lower()} data available")

    logger.debug(
        f"{data_name} data: {len(data)} rows, "
        f"range: {data.index.min()} to {data.index.max()}"
    )

    return data


@deprecated(reason="use methods available in @app/flows/base.py")
async def process_multiple_provider_results(tasks: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process multiple provider tasks with consistent error handling.

    This function handles the common pattern used in compare.py and similar
    workflows where multiple tickers/data sources are fetched in parallel.

    Args:
        tasks: Dictionary mapping identifiers to provider tasks

    Returns:
        Dictionary mapping identifiers to result metadata:
        - success: bool indicating if fetch succeeded
        - data: DataFrame if successful
        - error: str error message if failed
        - execution_time: Optional execution time from provider
    """
    if not tasks:
        return {}

    # Execute all tasks in parallel with proper typing
    results: Tuple[ProviderResult[pd.DataFrame] | BaseException, ...] = (
        await asyncio.gather(*tasks.values(), return_exceptions=True)
    )

    # Process results with consistent error handling
    processed_results = {}

    for identifier, result in zip(tasks.keys(), results):
        try:
            # Validate the provider result
            data = validate_provider_result(result, identifier)

            # Store success result
            processed_results[identifier] = {
                "success": True,
                "data": data,
                "execution_time": getattr(result, "execution_time", None),
            }
            logger.debug(f"Successfully processed {identifier}: {len(data)} rows")

        except Exception as e:
            # Store failure result
            processed_results[identifier] = {
                "success": False,
                "error": str(e),
            }
            logger.warning(f"Failed to process {identifier}: {e}")

    return processed_results


@deprecated(reason="use methods available in @app/flows/base.py")
def validate_single_provider_task(
    result: ProviderResult[pd.DataFrame] | BaseException,
    data_name: str,
    check_empty: bool = True,
) -> pd.DataFrame:
    """
    Validate a single provider task result (simplified version).

    This is a streamlined version for single data source workflows
    like those in markets.py where we just need the DataFrame.

    Args:
        result: Provider result or exception to validate
        data_name: Name of the data source for error messages
        check_empty: Whether to check for empty DataFrame

    Returns:
        Validated DataFrame from the provider result

    Raises:
        Exception: If result is invalid or contains no data
    """
    # Check for exceptions first
    if isinstance(result, Exception):
        logger.error(f"Failed to fetch {data_name} data: {result}")
        raise result

    # Check provider success status
    if not (hasattr(result, "success") and result.success):
        error_msg = getattr(result, "error_message", f"Unknown {data_name} fetch error")
        logger.error(f"{data_name} provider failed: {error_msg}")
        raise Exception(f"{data_name} data fetch failed: {error_msg}")

    # Check data attribute and type
    if not (hasattr(result, "data") and isinstance(result.data, pd.DataFrame)):
        error_msg = getattr(result, "error_message", f"Unknown {data_name} data error")
        logger.error(f"{data_name} provider returned invalid data: {error_msg}")
        raise Exception(f"{data_name} data fetch failed: {error_msg}")

    data = result.data

    # Check for empty data if requested
    if check_empty and data.empty:
        logger.error(f"Empty {data_name} data returned")
        raise Exception(f"No {data_name.lower()} data available")

    if not data.empty:
        logger.debug(
            f"{data_name} data: {len(data)} rows, "
            f"range: {data.index.min()} to {data.index.max()}"
        )

    return data


@deprecated(reason="use methods available in @app/flows/base.py")
async def create_flow_result_from_provider_results(
    tasks: Dict[str, Any],
    base_date: datetime | None = None,
    start_time: float | None = None,
) -> FlowResult[pd.DataFrame]:
    """
    Create a FlowResult from multiple provider tasks with consistent error handling.

    This is a higher-level wrapper around process_multiple_provider_results that
    returns a standardized FlowResult instead of a raw dictionary.

    Args:
        tasks: Dictionary mapping identifiers to provider tasks
        base_date: Base date used for data filtering
        start_time: Start time for execution time calculation

    Returns:
        FlowResult containing combined DataFrame or error information
    """
    execution_start = start_time or time.time()

    try:
        # Use existing provider results processing
        processed_results = await process_multiple_provider_results(tasks)

        if not processed_results:
            return FlowResult.success_result(
                data=pd.DataFrame(),
                base_date=base_date,
                execution_time=time.time() - execution_start,
                successful_items=[],
                failed_items=[],
                metadata={"message": "No tasks provided"},
            )

        # Separate successful and failed results
        successful_items = [
            identifier
            for identifier, result in processed_results.items()
            if result["success"]
        ]
        failed_items = [
            identifier
            for identifier, result in processed_results.items()
            if not result["success"]
        ]

        # If all tasks failed, return error result
        if not successful_items:
            error_messages = [
                f"{item}: {processed_results[item]['error']}" for item in failed_items
            ]
            return FlowResult.error_result(
                error_message=(
                    f"Failed to process all items: {'; '.join(error_messages)}"
                ),
                base_date=base_date,
                execution_time=time.time() - execution_start,
                failed_items=failed_items,
                metadata={"processed_results": processed_results},
            )

        # Combine successful data into single DataFrame
        combined_data = pd.DataFrame()
        metadata = {"processed_results": processed_results}

        for identifier in successful_items:
            result_data = processed_results[identifier]["data"]
            if not result_data.empty:
                # Add data with identifier as column name or suffix
                if combined_data.empty:
                    combined_data = result_data.copy()
                else:
                    combined_data = pd.concat([combined_data, result_data], axis=1)

        return FlowResult.success_result(
            data=combined_data,
            base_date=base_date,
            execution_time=time.time() - execution_start,
            successful_items=successful_items,
            failed_items=failed_items,
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"Error creating FlowResult from provider results: {e}")
        return FlowResult.error_result(
            error_message=f"Flow execution failed: {str(e)}",
            base_date=base_date,
            execution_time=time.time() - execution_start,
            failed_items=list(tasks.keys()) if tasks else [],
        )
