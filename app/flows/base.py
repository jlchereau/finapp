"""
Base classes and utilities for LlamaIndex workflows.

This module provides standardized result classes and common utilities
for all workflow operations in the application.
"""

import asyncio
import time
from deprecated.classic import deprecated
from datetime import datetime
from typing import Any, Dict, Tuple

from workflows import Workflow
from workflows.events import StopEvent
from pydantic import BaseModel, Field
import pandas as pd

from app.providers.base import ProviderResult
from app.lib.logger import logger
from app.lib.exceptions import FlowException


class FlowResult[T](BaseModel):
    """
    Standardized result wrapper for all workflow operations.

    Provides consistent structure and metadata across all flows,
    similar to ProviderResult for data providers.

    Type parameter T can be:
    - pd.DataFrame for data flows
    - Pydantic models for structured results
    - Any other result type
    """

    model_config = {"arbitrary_types_allowed": True}

    success: bool = Field(description="Whether the flow execution was successful")
    data: T | None = Field(
        default=None, description="Main result data (DataFrame or Pydantic model)"
    )
    error_message: str | None = Field(
        default=None, description="Error message if flow failed"
    )
    base_date: datetime | None = Field(
        default=None, description="Base date used for data filtering"
    )
    execution_time: float | None = Field(
        default=None, description="Flow execution time in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the flow was executed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional flow-specific data and metrics"
    )

    # Success indicators for multi-ticker/multi-item flows
    successful_items: list[str] = Field(
        default_factory=list,
        description="Successfully processed items (tickers, symbols, etc.)",
    )
    failed_items: list[str] = Field(
        default_factory=list, description="Failed items that could not be processed"
    )

    @classmethod
    def success_result(
        cls,
        data: T,
        base_date: datetime | None = None,
        execution_time: float | None = None,
        metadata: dict[str, Any] | None = None,
        successful_items: list[str] | None = None,
        failed_items: list[str] | None = None,
    ) -> "FlowResult[T]":
        """
        Create a successful FlowResult.

        Args:
            data: The result data
            base_date: Base date used for filtering
            execution_time: Execution time in seconds
            metadata: Additional flow-specific data
            successful_items: Successfully processed items
            failed_items: Failed items

        Returns:
            FlowResult with success=True
        """
        return cls(
            success=True,
            data=data,
            base_date=base_date,
            execution_time=execution_time,
            metadata=metadata or {},
            successful_items=successful_items or [],
            failed_items=failed_items or [],
        )

    @classmethod
    def error_result(
        cls,
        error_message: str,
        base_date: datetime | None = None,
        execution_time: float | None = None,
        metadata: dict[str, Any] | None = None,
        failed_items: list[str] | None = None,
    ) -> "FlowResult[T]":
        """
        Create a failed FlowResult.

        Args:
            error_message: Error description
            base_date: Base date used for filtering
            execution_time: Execution time in seconds
            metadata: Additional flow-specific data
            failed_items: Items that failed to process

        Returns:
            FlowResult with success=False
        """
        return cls(
            success=False,
            data=None,
            error_message=error_message,
            base_date=base_date,
            execution_time=execution_time,
            metadata=metadata or {},
            successful_items=[],
            failed_items=failed_items or [],
        )


class FlowResultEvent(StopEvent):
    """
    Unified workflow result event.

    This replaces FlowResult for cleaner LlamaIndex workflow patterns.
    Can represent both success and error states in a single event type.
    """

    success: bool = Field(description="Whether the workflow execution was successful")
    data: pd.DataFrame | Any | None = Field(
        default=None, description="Main result data (DataFrame or other)"
    )
    error_message: str | None = Field(
        default=None, description="Error message if workflow failed"
    )
    base_date: datetime | None = Field(
        default=None, description="Base date used for data filtering"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional flow-specific data and metrics"
    )

    # Keep these for backward compatibility
    successful_items: list[str] = Field(
        default_factory=list, description="Successfully processed items"
    )
    failed_items: list[str] = Field(default_factory=list, description="Failed items")

    @classmethod
    def success_result(
        cls,
        data: Any,
        base_date: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        successful_items: list[str] | None = None,
        failed_items: list[str] | None = None,
    ) -> "FlowResultEvent":
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            base_date=base_date,
            metadata=metadata or {},
            successful_items=successful_items or [],
            failed_items=failed_items or [],
        )

    @classmethod
    def error_result(
        cls,
        error_message: str,
        base_date: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        failed_items: list[str] | None = None,
    ) -> "FlowResultEvent":
        """Create an error result."""
        return cls(
            success=False,
            error_message=error_message,
            base_date=base_date,
            metadata=metadata or {},
            successful_items=[],
            failed_items=failed_items or [],
        )


class FlowRunner[T]:
    """
    Standardized runner for LlamaIndex workflows.

    Provides consistent workflow execution with automatic FlowResult wrapping,
    timing, error handling, and integration of helper utilities.

    Type parameter T should match the expected workflow result type:
    - pd.DataFrame for data workflows
    - Pydantic models for structured results
    - Any other result type
    """

    def __init__(self, workflow: Workflow):
        """
        Initialize FlowRunner with a LlamaIndex workflow instance.

        Args:
            workflow: LlamaIndex Workflow instance to execute
        """
        self.workflow: Workflow = workflow
        self._start_time: float | None = None

    async def run(self, **kwargs) -> FlowResultEvent:
        """
        Execute the workflow and return a standardized StopEvent.

        Args:
            **kwargs: Parameters to pass to workflow.run()

        Returns:
            FlowResultEvent with execution metadata
        """
        self._start_time = time.time()

        try:
            logger.debug(f"Starting workflow {self.workflow.__class__.__name__}")

            # Execute the workflow - it should return FlowResultEvent
            result_event = await self.workflow.run(**kwargs)
            execution_time = time.time() - self._start_time

            # Add execution time to the event metadata
            result_event.metadata["execution_time"] = execution_time
            result_event.metadata["workflow"] = self.workflow.__class__.__name__

            return result_event

        except Exception as e:
            execution_time = time.time() - self._start_time
            logger.error(f"Workflow {self.workflow.__class__.__name__} failed: {e}")

            return FlowResultEvent.error_result(
                error_message=str(e),
                metadata={
                    "execution_time": execution_time,
                    "workflow": self.workflow.__class__.__name__,
                    "error_type": type(e).__name__,
                    "kwargs": kwargs,
                },
            )

    def _convert_dict_result(
        self, result_data: dict, execution_time: float, **kwargs
    ) -> FlowResult[T]:
        """
        Convert legacy dictionary result to FlowResult format.

        Args:
            result_data: Dictionary result from workflow
            execution_time: Execution time in seconds
            **kwargs: Original workflow parameters

        Returns:
            FlowResult[T] with converted data
        """
        # Check if this is an error result
        if result_data.get("error"):
            return FlowResult.error_result(
                error_message=result_data.get("error", "Unknown workflow error"),
                execution_time=execution_time,
                failed_items=result_data.get("failed_tickers", []),
                metadata={
                    "workflow": self.workflow.__class__.__name__,
                    "original_result": result_data,
                },
            )

        # Check if data is missing or empty (but only if there's no explicit error)
        data = result_data.get("data")
        if data is None or (hasattr(data, "empty") and data.empty):
            return FlowResult.error_result(
                error_message="No data returned from workflow",
                execution_time=execution_time,
                failed_items=result_data.get("failed_tickers", []),
                metadata={
                    "workflow": self.workflow.__class__.__name__,
                    "original_result": result_data,
                },
            )

        # Success result - extract standard fields
        return FlowResult.success_result(
            data=data,
            base_date=kwargs.get("base_date"),
            execution_time=execution_time,
            successful_items=result_data.get("successful_tickers", []),
            failed_items=result_data.get("failed_tickers", []),
            metadata={
                "workflow": self.workflow.__class__.__name__,
                "original_result": result_data,
                **{
                    k: v
                    for k, v in result_data.items()
                    if k not in ["data", "successful_tickers", "failed_tickers"]
                },
            },
        )

    @staticmethod
    def _validate_provider_result(
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
            FlowException: If result is invalid or contains no data
        """
        # Check for exceptions first
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch {data_name} data: {result}")
            raise result

        # Check provider success status
        if not (hasattr(result, "success") and result.success):
            error_msg = getattr(
                result, "error_message", f"Unknown {data_name} fetch error"
            )
            logger.error(f"{data_name} provider failed: {error_msg}")
            raise FlowException(
                workflow="FlowRunner",
                step="validate_provider_result",
                message=f"{data_name} data fetch failed: {error_msg}",
            )

        # Check data attribute and type
        if not (hasattr(result, "data") and isinstance(result.data, pd.DataFrame)):
            error_msg = getattr(
                result, "error_message", f"Unknown {data_name} data error"
            )
            logger.error(f"{data_name} provider returned invalid data: {error_msg}")
            raise FlowException(
                workflow="FlowRunner",
                step="validate_provider_result",
                message=f"{data_name} data fetch failed: {error_msg}",
            )

        data = result.data

        # Check for empty data
        if data.empty:
            logger.error(f"Empty {data_name} data returned")
            raise FlowException(
                workflow="FlowRunner",
                step="validate_provider_result",
                message=f"No {data_name.lower()} data available",
            )

        logger.debug(
            f"{data_name} data: {len(data)} rows, "
            f"range: {data.index.min()} to {data.index.max()}"
        )

        return data

    @deprecated(
        reason="This is a bad use of llama-index workflows which have the "
        "ability to handle parallel tasks"
    )
    @staticmethod
    async def process_provider_tasks(tasks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multiple provider tasks with consistent error handling.

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
                data = FlowRunner._validate_provider_result(result, identifier)

                # Store success result
                processed_results[identifier] = {
                    "success": True,
                    "data": data,
                    "execution_time": getattr(result, "execution_time", None),
                }
                logger.debug(f"Successfully processed {identifier}: {len(data)} rows")

            except (
                FlowException,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
            ) as e:
                # Store failure result
                processed_results[identifier] = {
                    "success": False,
                    "error": str(e),
                }
                logger.warning(f"Failed to process {identifier}: {e}")

        return processed_results

    @staticmethod
    def _validate_single_provider_task(
        result: ProviderResult[pd.DataFrame] | BaseException,
        data_name: str,
        check_empty: bool = True,
    ) -> pd.DataFrame:
        """
        Validate a single provider task result (simplified version).

        Args:
            result: Provider result or exception to validate
            data_name: Name of the data source for error messages
            check_empty: Whether to check for empty DataFrame

        Returns:
            Validated DataFrame from the provider result

        Raises:
            FlowException: If result is invalid or contains no data
        """
        # Check for exceptions first
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch {data_name} data: {result}")
            raise result

        # Check provider success status
        if not (hasattr(result, "success") and result.success):
            error_msg = getattr(
                result, "error_message", f"Unknown {data_name} fetch error"
            )
            logger.error(f"{data_name} provider failed: {error_msg}")
            raise FlowException(
                workflow="FlowRunner",
                step="validate_single_provider_task",
                message=f"{data_name} data fetch failed: {error_msg}",
            )

        # Check data attribute and type
        if not (hasattr(result, "data") and isinstance(result.data, pd.DataFrame)):
            error_msg = getattr(
                result, "error_message", f"Unknown {data_name} data error"
            )
            logger.error(f"{data_name} provider returned invalid data: {error_msg}")
            raise FlowException(
                workflow="FlowRunner",
                step="validate_single_provider_task",
                message=f"{data_name} data fetch failed: {error_msg}",
            )

        data = result.data

        # Check for empty data if requested
        if check_empty and data.empty:
            logger.error(f"Empty {data_name} data returned")
            raise FlowException(
                workflow="FlowRunner",
                step="validate_single_provider_task",
                message=f"No {data_name.lower()} data available",
            )

        if not data.empty:
            logger.debug(
                f"{data_name} data: {len(data)} rows, "
                f"range: {data.index.min()} to {data.index.max()}"
            )

        return data

    @staticmethod
    async def _create_from_provider_results(
        tasks: Dict[str, Any],
        base_date: datetime | None = None,
        start_time: float | None = None,
    ) -> FlowResult[pd.DataFrame]:
        """
        Create a FlowResult from multiple provider tasks with consistent error handling.

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
            processed_results = await FlowRunner.process_provider_tasks(tasks)

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
                    f"{item}: {processed_results[item]['error']}"
                    for item in failed_items
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

        except (FlowException, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Error creating FlowResult from provider results: {e}")
            return FlowResult.error_result(
                error_message=f"Flow execution failed: {str(e)}",
                base_date=base_date,
                execution_time=time.time() - execution_start,
                failed_items=list(tasks.keys()) if tasks else [],
            )
