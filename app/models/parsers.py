"""
Parsers for different data formats and sources.

This module provides parsing capabilities for various data formats:
- JSON parsing with JMESPath expressions
- Future support for XML, CSV, and other formats
- Thread-safe model caching
- Extensible parser framework
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from threading import Lock
import jmespath
from pydantic import BaseModel, create_model, Field, ValidationError
from jmespath.exceptions import JMESPathError

# Parsers for different data formats and sources.
MODEL_CACHE: dict[str, Type[BaseModel]] = {}
CACHE_LOCK = Lock()


class ParseError(Exception):
    """Custom exception for parsing errors."""

    # Custom exception for parsing errors


class ParserConfig(BaseModel):
    """Configuration for parsers."""

    name: str = Field(description="Name of the generated model")
    fields: dict[str, dict[str, Any]] = Field(
        description="Field definitions with expressions and defaults"
    )
    strict_mode: bool = Field(
        default=False, description="Whether to raise errors on missing fields"
    )
    default_value: Any = Field(
        default=None, description="Default value for missing fields"
    )


class BaseParser(ABC):
    """
    Abstract base class for all parsers.
    Provides common functionality and enforces interface.
    """

    def __init__(self, config: dict[str, Any] | ParserConfig) -> None:
        """Initialize parser with configuration."""
        if isinstance(config, dict):
            self.config = ParserConfig(**config)
        else:
            self.config = config

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    async def parse_async(self, data: Any) -> BaseModel:
        """Asynchronously parse data into a Pydantic model."""
        raise NotImplementedError()

    def parse(self, data: Any) -> BaseModel:
        """Synchronously parse data into a Pydantic model using asyncio.run."""
        return asyncio.run(self.parse_async(data))


class PydanticJSONParser(BaseParser):
    """
    Parser for JSON data using Pydantic models.
    The model is generated dynamically based on the provided configuration.

    Features:
    - Dynamic model generation with caching
    - JMESPath expressions for data extraction
    - Thread-safe operations
    - Async/await support
    - Comprehensive error handling
    """

    def __init__(self, config: dict[str, Any] | ParserConfig) -> None:
        """Initialize JSON parser with configuration."""
        super().__init__(config)
        self.model = self._get_or_create_model()

    def _get_or_create_model(self) -> Type[BaseModel]:
        """
        Get or create the Pydantic model with thread safety.
        Uses global cache to avoid recreating models.
        """
        with CACHE_LOCK:
            if self.config.name in MODEL_CACHE:
                return MODEL_CACHE[self.config.name]

            fields = {}
            for field_name, conf in self.config.fields.items():
                default_value = conf.get("default", self.config.default_value)
                field_type = conf.get("type", Any)

                # Handle optional types
                if default_value is not None or not self.config.strict_mode:
                    field_type = Optional[field_type]

                fields[field_name] = (field_type, Field(default=default_value))

            model = create_model(self.config.name, **fields)
            MODEL_CACHE[self.config.name] = model
            self.logger.info("Created and cached model: %s", self.config.name)
            return model

    async def parse_async(self, data: Any) -> BaseModel:
        """
        Asynchronously create instance of the Pydantic model from data.

        Args:
            json_data: Dictionary containing the JSON data to parse

        Returns:
            Instance of the dynamically created Pydantic model

        Raises:
            ParseError: If parsing fails and strict_mode is enabled
        """
        if not data:
            if self.config.strict_mode:
                raise ParseError("No data provided and strict mode is enabled")
            return self.model()

        try:
            extracted_data: dict[str, Any] = {}
            missing_fields: list[str] = []
            for key, conf in self.config.fields.items():
                expr = conf.get("expr")
                default = conf.get("default", self.config.default_value)

                if expr:
                    try:
                        value = jmespath.search(expr, data)
                    except JMESPathError as e:
                        self.logger.warning(
                            "JMESPath error for field '%s' with expr '%s': %s",
                            key,
                            expr,
                            e,
                        )
                        value = None
                else:
                    # Direct field access if no expression provided
                    value = data.get(key)

                if value is None:
                    if self.config.strict_mode and default is None:
                        missing_fields.append(key)
                    value = default

                extracted_data[key] = value

            if missing_fields and self.config.strict_mode:
                msg = "Missing required fields in strict mode: " f"{missing_fields}"
                raise ParseError(msg)

            return self.model(**extracted_data)

        except ValidationError as e:
            error_msg = f"Failed to parse data with {self.config.name}: {e}"
            self.logger.error("%s", error_msg)
            if self.config.strict_mode:
                raise ParseError(error_msg) from e
            # Return empty model if not in strict mode
            return self.model()


class PydanticMultiSourceParser(BaseParser):
    """
    Parser that can handle multiple data sources and formats.
    Useful for providers that combine data from different sources.
    """

    def __init__(self, parsers: list[BaseParser]) -> None:
        """
        Initialize with a list of sub-parsers.

        Args:
            parsers: List of parser instances to use
        """
        # Create a combined config for logging
        combined_config = ParserConfig(name="MultiSourceParser", fields={})
        super().__init__(combined_config)
        self.parsers = parsers

    async def parse_async(self, data: Any) -> list[BaseModel]:
        """
        Parse multiple data sources concurrently.

        Args:
            data_sources: List of data to parse (one per parser)

        Returns:
            List of parsed Pydantic models
        """
        if not isinstance(data, list) or len(data) != len(self.parsers):
            msg = (
                f"Data sources count ({len(data)}) must match parsers count "
                f"({len(self.parsers)})"
            )
            raise ParseError(msg)

        # Parse all sources concurrently
        # Launch parse tasks concurrently
        tasks = []
        for parser, src in zip(self.parsers, data):
            tasks.append(parser.parse_async(src))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error("Parser %d failed: %s", i, result)
                if self.parsers[i].config.strict_mode:
                    raise result

        # Filter out exceptions if not in strict mode
        # Filter out exceptions if not in strict mode
        filtered: list[BaseModel] = [
            r for r in results if not isinstance(r, Exception)
        ]  # type: ignore
        return filtered


# Factory function for easy parser creation
def create_json_parser(
    name: str, field_mappings: dict[str, str], strict_mode: bool = False
) -> PydanticJSONParser:
    """
    Factory function to create a JSON parser with simple field mappings.

    Args:
        name: Name for the generated model
        field_mappings: Dictionary mapping field names to JMESPath expressions
        strict_mode: Whether to enable strict mode

    Returns:
        Configured PydanticJSONParser instance
    """
    fields = {}
    for field_name, expr in field_mappings.items():
        fields[field_name] = {"expr": expr, "default": None}

    config = ParserConfig(name=name, fields=fields, strict_mode=strict_mode)

    return PydanticJSONParser(config)


# For future consideration:
# class PydanticXMLParser(BaseParser)
# class PydanticCSVParser(BaseParser)
# class PydanticHTMLParser(BaseParser)
