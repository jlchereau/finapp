#!/usr/bin/env pytest
# pylint: disable=attribute-defined-outside-init,no-member
# flake8: noqa
# type: ignore
"""
Unit tests for the parsers module.
Tests JSON parsing, model generation, and multi-source parsing.
"""

import pytest

from app.models.parsers import (
    PydanticJSONParser,
    PydanticMultiSourceParser,
    ParserConfig,
    ParseError,
    create_json_parser,
    MODEL_CACHE,
    CACHE_LOCK,
)


class TestParserConfig:
    """Test cases for ParserConfig."""

    def test_parser_config_defaults(self):
        """Test ParserConfig with default values."""
        config = ParserConfig(
            name="TestModel", fields={"price": {"expr": "price", "default": None}}
        )

        assert config.name == "TestModel"
        assert config.fields == {"price": {"expr": "price", "default": None}}
        assert config.strict_mode is False
        assert config.default_value is None

    def test_parser_config_custom_values(self):
        """Test ParserConfig with custom values."""
        fields = {
            "price": {"expr": "current_price", "default": 0.0},
            "volume": {"expr": "trading_volume", "default": 0},
        }
        config = ParserConfig(
            name="CustomModel", fields=fields, strict_mode=True, default_value="N/A"
        )

        assert config.name == "CustomModel"
        assert config.fields == fields
        assert config.strict_mode is True
        assert config.default_value == "N/A"


class TestPydanticJSONParser:
    """Test cases for PydanticJSONParser."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear cache before each test
        with CACHE_LOCK:
            MODEL_CACHE.clear()

        self.simple_config = ParserConfig(
            name="SimpleTestModel",
            fields={
                "price": {"expr": "price", "default": 0.0},
                "volume": {"expr": "volume", "default": 0},
                "symbol": {"expr": "symbol", "default": "UNKNOWN"},
            },
        )

        self.jmespath_config = ParserConfig(
            name="JMESPathTestModel",
            fields={
                "current_price": {"expr": "quote.price", "default": None},
                "market_cap": {"expr": "fundamentals.marketCap", "default": None},
                "currency": {"expr": "quote.currency", "default": "USD"},
            },
        )

    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = PydanticJSONParser(self.simple_config)

        assert parser.config.name == "SimpleTestModel"
        assert hasattr(parser, "model")
        assert hasattr(parser, "logger")

    def test_parser_initialization_with_dict(self):
        """Test parser initialization with dictionary config."""
        config_dict = {
            "name": "DictTestModel",
            "fields": {"price": {"expr": "price", "default": 0.0}},
        }
        parser = PydanticJSONParser(config_dict)

        assert parser.config.name == "DictTestModel"

    def test_model_caching(self):
        """Test that models are cached properly."""
        # Create first parser
        parser1 = PydanticJSONParser(self.simple_config)
        model1 = parser1.model

        # Create second parser with same config
        parser2 = PydanticJSONParser(self.simple_config)
        model2 = parser2.model

        # Should be the same cached model
        assert model1 is model2
        assert "SimpleTestModel" in MODEL_CACHE

    @pytest.mark.asyncio
    async def test_parse_simple_data(self):
        """Test parsing simple JSON data."""
        parser = PydanticJSONParser(self.simple_config)
        data = {"price": 150.50, "volume": 1000000, "symbol": "AAPL"}

        result = await parser.parse_async(data)

        assert result.price == 150.50
        assert result.volume == 1000000
        assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_parse_with_jmespath(self):
        """Test parsing with JMESPath expressions."""
        parser = PydanticJSONParser(self.jmespath_config)
        data = {
            "quote": {"price": 150.50, "currency": "USD"},
            "fundamentals": {"marketCap": 2500000000000},
        }

        result = await parser.parse_async(data)

        assert result.current_price == 150.50
        assert result.market_cap == 2500000000000
        assert result.currency == "USD"

    @pytest.mark.asyncio
    async def test_parse_missing_fields_non_strict(self):
        """Test parsing with missing fields in non-strict mode."""
        parser = PydanticJSONParser(self.simple_config)
        data = {"price": 150.50}  # Missing volume and symbol

        result = await parser.parse_async(data)

        assert result.price == 150.50
        assert result.volume == 0  # Default value
        assert result.symbol == "UNKNOWN"  # Default value

    @pytest.mark.asyncio
    async def test_parse_missing_fields_strict_mode(self):
        """Test parsing with missing fields in strict mode."""
        strict_config = ParserConfig(
            name="StrictTestModel",
            fields={
                "price": {"expr": "price", "default": None},
                "required_field": {"expr": "required", "default": None},
            },
            strict_mode=True,
        )
        parser = PydanticJSONParser(strict_config)
        data = {"price": 150.50}  # Missing required_field

        with pytest.raises(ParseError) as exc_info:
            await parser.parse_async(data)

        assert "Missing required fields" in str(exc_info.value)
        assert "required_field" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_empty_data_non_strict(self):
        """Test parsing empty data in non-strict mode."""
        parser = PydanticJSONParser(self.simple_config)

        result = await parser.parse_async({})

        assert result.price == 0.0
        assert result.volume == 0
        assert result.symbol == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_strict_mode_no_data(self):
        """Test strict mode with no data provided."""
        strict_config = ParserConfig(
            name="StrictEmptyTestModel",
            fields={"price": {"expr": "price", "default": None}},
            strict_mode=True,
        )
        parser = PydanticJSONParser(strict_config)

        with pytest.raises(ParseError) as exc_info:
            await parser.parse_async({})

        assert "No data provided and strict mode is enabled" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_invalid_jmespath(self):
        """Test handling of invalid JMESPath expressions."""
        invalid_config = ParserConfig(
            name="InvalidJMESPathModel",
            fields={"price": {"expr": "invalid[[[expression", "default": 0.0}},
        )
        parser = PydanticJSONParser(invalid_config)
        data = {"price": 150.50}

        # Should not raise exception in non-strict mode
        result = await parser.parse_async(data)
        assert result.price == 0.0  # Should use default

    def test_parse_sync_wrapper(self):
        """Test synchronous parse wrapper."""
        parser = PydanticJSONParser(self.simple_config)
        data = {"price": 150.50, "volume": 1000000, "symbol": "AAPL"}

        result = parser.parse(data)

        assert getattr(result, "price") == 150.50
        assert getattr(result, "volume") == 1000000
        assert getattr(result, "symbol") == "AAPL"

    @pytest.mark.asyncio
    async def test_parse_none_data(self):
        """Test parsing None data."""
        parser = PydanticJSONParser(self.simple_config)

        result = await parser.parse_async(None)

        # Should return model with defaults
        assert result.price == 0.0
        assert result.volume == 0
        assert result.symbol == "UNKNOWN"


class TestPydanticMultiSourceParser:
    """Test cases for PydanticMultiSourceParser."""

    def setup_method(self):
        """Set up test fixtures."""
        with CACHE_LOCK:
            MODEL_CACHE.clear()

        self.parser1 = PydanticJSONParser(
            ParserConfig(
                name="Source1Model", fields={"price": {"expr": "price", "default": 0.0}}
            )
        )

        self.parser2 = PydanticJSONParser(
            ParserConfig(
                name="Source2Model", fields={"volume": {"expr": "volume", "default": 0}}
            )
        )

    def test_multi_source_parser_initialization(self):
        """Test multi-source parser initialization."""
        parser = PydanticMultiSourceParser([self.parser1, self.parser2])

        assert len(parser.parsers) == 2
        assert parser.config.name == "MultiSourceParser"

    @pytest.mark.asyncio
    async def test_parse_multiple_sources(self):
        """Test parsing multiple data sources."""
        parser = PydanticMultiSourceParser([self.parser1, self.parser2])
        data_sources = [{"price": 150.50}, {"volume": 1000000}]

        results = await parser.parse_async(data_sources)

        assert len(results) == 2
        assert results[0].price == 150.50
        assert results[1].volume == 1000000

    @pytest.mark.asyncio
    async def test_parse_mismatched_sources_count(self):
        """Test error when data sources don't match parsers count."""
        parser = PydanticMultiSourceParser([self.parser1, self.parser2])
        data_sources = [{"price": 150.50}]  # Only one source, need two

        with pytest.raises(ParseError) as exc_info:
            await parser.parse_async(data_sources)

        assert "Data sources count" in str(exc_info.value)
        assert "must match parsers count" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_with_parser_exception_strict(self):
        """Test handling parser exceptions when parser is in strict mode."""
        # Create a strict parser that will fail
        strict_parser = PydanticJSONParser(
            ParserConfig(
                name="FailingModel",
                fields={"required": {"expr": "required", "default": None}},
                strict_mode=True,
            )
        )

        # Multi-source parser with a strict parser
        parser = PydanticMultiSourceParser([self.parser1, strict_parser])
        data_sources = [{"price": 150.50}, {}]  # This will fail the strict parser

        # Should raise the exception from the strict parser
        with pytest.raises(ParseError) as exc_info:
            await parser.parse_async(data_sources)

        assert "No data provided and strict mode is enabled" in str(exc_info.value)


class TestCreateJsonParser:
    """Test cases for create_json_parser factory function."""

    def test_create_simple_parser(self):
        """Test creating a simple JSON parser."""
        field_mappings = {
            "price": "quote.price",
            "volume": "quote.volume",
            "symbol": "symbol",
        }

        parser = create_json_parser("FactoryTestModel", field_mappings)

        assert parser.config.name == "FactoryTestModel"
        assert len(parser.config.fields) == 3
        assert parser.config.fields["price"]["expr"] == "quote.price"
        assert parser.config.strict_mode is False

    def test_create_strict_parser(self):
        """Test creating a strict JSON parser."""
        field_mappings = {"price": "price"}

        parser = create_json_parser(
            "StrictFactoryTestModel", field_mappings, strict_mode=True
        )

        assert parser.config.strict_mode is True

    @pytest.mark.asyncio
    async def test_factory_parser_functionality(self):
        """Test that factory-created parser works correctly."""
        field_mappings = {
            "current_price": "quote.price",
            "trading_volume": "quote.volume",
        }

        parser = create_json_parser("FactoryFunctionalTest", field_mappings)
        data = {"quote": {"price": 150.50, "volume": 1000000}}

        result = await parser.parse_async(data)

        assert getattr(result, "current_price") == 150.50
        assert getattr(result, "trading_volume") == 1000000


class TestParseError:
    """Test cases for ParseError exception."""

    def test_parse_error_creation(self):
        """Test ParseError exception creation."""
        error = ParseError("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_parse_error_with_chaining(self):
        """Test ParseError with exception chaining."""
        original_error = ValueError("Original error")
        try:
            raise ParseError("Parse failed") from original_error
        except ParseError as parse_error:
            assert str(parse_error) == "Parse failed"
            assert parse_error.__cause__ is original_error
