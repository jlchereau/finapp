"""
Base classes and utilities for LlamaIndex workflows.

This module provides standardized result classes and common utilities
for all workflow operations in the application.
"""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


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
