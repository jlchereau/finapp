---
applyTo: "app/models/**/*.py"
---

# Improved Data Providers and Parsers Design

This document outlines the enhanced design for data providers and parsers in the FinApp project, addressing the requirements for extensibility, async/sync compatibility, multithreading, error reporting, and future caching capabilities.

## Key Improvements

### 1. **Async-First Architecture**
- All providers now support `async`/`await` for seamless integration with llama-index workflows
- Thread pool execution for blocking operations (like yfinance calls)
- Proper semaphore-based concurrency control
- Non-blocking I/O operations

### 2. **Comprehensive Error Handling**
- `ProviderResult<T>` wrapper provides standardized error reporting
- Retry logic with exponential backoff
- Timeout handling
- Detailed error categorization and metadata
- Exception chaining for better debugging

### 3. **Extensible Provider Framework**
- Abstract `BaseProvider<T>` class with generic typing
- Factory functions for easy provider creation
- Provider-specific configuration through `ProviderConfig`
- Plugin-style architecture for adding new data sources

### 4. **Enhanced Parser System**
- Thread-safe model caching
- Async parsing capabilities
- Multi-source parser support
- JMESPath expressions for flexible data extraction
- Strict/non-strict parsing modes

### 5. **Future-Ready Caching**
- Infrastructure prepared for caching implementation
- Cache TTL and invalidation hooks
- Provider-level cache configuration
- Metadata tracking for cache management

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Reflex Page   │────│ Workflow Engine  │────│  Data Provider  │
│ (Background     │    │ (llama-index)    │    │  (Yahoo/Zacks)  │
│  Event)         │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                      │
         │              ┌────────▼────────┐             │
         │              │ ProviderResult  │             │
         │              │   - success     │             │
         │              │   - data        │             │
         │              │   - errors      │             │
         │              │   - metadata    │             │
         │              └─────────────────┘             │
         │                                              │
         └──────────────── Error Handling ──────────────┘
```

## Usage Examples

### Basic Provider Usage

```python
from app.models.yahoo import create_yahoo_info_provider

# Create configured provider
provider = create_yahoo_info_provider(timeout=30.0, retries=3)

# Async usage (recommended for workflows)
result = await provider.get_data("AAPL")
if result.success:
    data = result.data
    print(f"Company: {data.company_name}")
else:
    print(f"Error: {result.error_message}")

# Sync usage (for non-async contexts)
result = provider.get_data_sync("AAPL")
```

### Workflow Integration

```python
from app.flows.financial_data import fetch_financial_data

# In a Reflex background event
@rx.event(background=True)
async def fetch_ticker_data(self):
    result = await fetch_financial_data("AAPL")
    # Result includes data from multiple providers with error handling
```

### Custom Provider Creation

```python
from app.models.base import BaseProvider, ProviderType

class CustomProvider(BaseProvider[DataFrame]):
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.CUSTOM
    
    async def _fetch_data(self, ticker: str, **kwargs) -> DataFrame:
        # Your custom data fetching logic
        pass
```

## Configuration Options

### Provider Configuration

```python
config = ProviderConfig(
    timeout=30.0,           # Request timeout
    retries=3,              # Retry attempts
    retry_delay=1.0,        # Delay between retries
    cache_enabled=True,     # Enable caching (future)
    cache_ttl=3600,         # Cache TTL in seconds
    rate_limit=1.0,         # Requests per second
    extra_config={          # Provider-specific settings
        "period": "1y",
        "interval": "1d"
    }
)
```

### Parser Configuration

```python
config = ParserConfig(
    name="CustomModel",
    fields={
        "price": {"expr": "regularMarketPrice", "default": None},
        "volume": {"expr": "volume", "default": 0}
    },
    strict_mode=False,      # Raise errors on missing fields
    default_value=None      # Global default for missing values
)
```

## Error Handling Strategy

### Provider-Level Errors
- Network timeouts and connection errors
- API rate limiting and quota exceeded
- Invalid ticker symbols
- Malformed response data

### Parser-Level Errors
- Invalid JSON/data format
- Missing required fields (in strict mode)
- JMESPath expression errors
- Type conversion failures

### Workflow-Level Errors
- Provider initialization failures
- Concurrent execution limits
- Resource exhaustion
- Unexpected exceptions

## Performance Characteristics

### Concurrency
- **Provider-Level Concurrency**: Semaphore-based limiting (default: 10, but should be configurable per provider) to manage the rate of requests to external APIs and prevent rate-limiting errors. This is a form of "API politeness."
- **Workflow-Level Concurrency**: Orchestrated by the workflow engine in `app/flows` (e.g., LlamaIndex's `num_workers`). This layer decides *what* to run in parallel, while the provider layer manages *how* to execute its specific calls safely.
- **Thread Pool for Blocking Operations**: Ensures that synchronous, blocking calls (like the `yfinance` library) do not halt the entire async event loop.
- **Async/await Throughout**: Maximizes I/O efficiency.

### Memory Management
- **Model Caching**: The `PydanticJSONParser` uses a global, thread-safe cache to avoid recompiling Pydantic models, reducing object creation and memory churn.
- **Streaming Data Processing**: For large datasets, consider using streaming responses where possible.
- **Garbage Collection Friendly**: The design avoids circular references and manages object lifecycles cleanly.
- **Connection Pooling**: `httpx.AsyncClient` is used to manage and reuse HTTP connections efficiently.

### Execution Times
Based on testing with the enhanced providers:
- Single ticker (3 providers): ~2-4 seconds
- Multiple tickers (3 providers each): ~3-6 seconds (parallel execution)
- Individual provider calls: ~0.5-2 seconds each

## Future Enhancements

### Caching Implementation
- Redis-based distributed caching
- File-based local caching for development
- Intelligent cache invalidation
- Cache warming strategies

### Additional Providers
- Alpha Vantage API integration
- Interactive Brokers (IBAPI) support
- Cryptocurrency data providers
- Economic indicators and macro data

### Enhanced Analytics
- Data quality scoring
- Provider reliability metrics
- Performance monitoring and alerting
- Usage analytics and optimization

## Migration Guide

### From Old Architecture

1. **Update imports:**
   ```python
   # Old
   from app.models.yahoo import YahooProvider
   
   # New
   from app.models.yahoo import create_yahoo_info_provider
   ```

2. **Update method calls:**
   ```python
   # Old
   provider = YahooProvider()
   data = provider.get_data("AAPL")
   
   # New
   provider = create_yahoo_info_provider()
   result = await provider.get_data("AAPL")
   if result.success:
       data = result.data
   ```

3. **Update error handling:**
   ```python
   # Old
   try:
       data = provider.get_data("AAPL")
   except Exception as e:
       print(f"Error: {e}")
   
   # New
   result = await provider.get_data("AAPL")
   if not result.success:
       print(f"Error: {result.error_message}")
   ```

This improved architecture provides a solid foundation for scaling the financial data infrastructure while maintaining excellent error handling, performance, and developer experience.
