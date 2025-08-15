---
applyTo: "app/providers/**/*.py"
---

# Data Providers Architecture Design

This document outlines the design for data providers in the FinApp project, focusing on collecting data from external sources with consistent output formats using Pydantic models.

## Separation of Concerns

### **Providers**: Data Collection Layer
- **Purpose**: Collect data from external data sources (APIs, web scraping, databases)
- **Scope**: One provider per data source (Yahoo Finance, Zacks, TipRanks, BlackRock, etc.)
- **Output**: Pydantic models ensure consistent, validated output format
- **Responsibilities**:
  - Handle API authentication and rate limiting
  - Manage HTTP requests and error handling
  - Transform raw API responses into standardized Pydantic models
  - Provide caching for expensive data fetching operations

### **Workflows**: Data Assembly Layer
- **Purpose**: Orchestrate multiple providers to assemble comprehensive datasets
- **Scope**: Business logic for combining data from various sources
- **Output**: Structured data ready for Reflex pages
- **Responsibilities**:
  - Coordinate concurrent data fetching from multiple providers
  - Handle cross-provider error scenarios
  - Aggregate and merge data from different sources
  - Feed processed data to Reflex pages for presentation

## Key Features

### 1. **Async-First Architecture**
- All providers support `async`/`await` for seamless integration with llama-index workflows
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

### 4. **Pydantic-Based Data Modeling**
- **Direct Pydantic models**: Use Pydantic models with aliases for field mapping
- **Alias-based transformation**: Use `Field(alias="api_field_name")` for API response mapping
- **Automatic validation**: Pydantic ensures type safety and data validation
- **Flexible field mapping**: Support for both direct field names and API aliases

### 5. **Cache Decorator**
- Transparent file-based caching via the `@cache` decorator in `app/providers/cache.py`
- Caches provider `_fetch_data` outputs under `data/YYYYMMDD/` as Parquet (for DataFrame)
  or JSON (for Pydantic BaseModel)
- Supports optional `cache_date` parameter for read-only historical reads
- Ensures thread safety with per-file `asyncio.Lock`

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Reflex Page   │────│   Workflow       │────│  Data Provider  │
│ (Presentation)  │    │ (Data Assembly)  │    │ (Data Collection)│
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                      │
         │              ┌────────▼────────┐    ┌────────▼────────┐
         │              │ ProviderResult  │    │ Pydantic Model  │
         │              │   - success     │    │   - validated   │
         │              │   - data        │    │   - typed       │
         │              │   - errors      │    │   - aliases     │
         │              │   - metadata    │    │                 │
         │              └─────────────────┘    └─────────────────┘
         │                                              │
         └──────────────── Error Handling ──────────────┘
```

### Data Flow
1. **Reflex Page** triggers background event
2. **Workflow** orchestrates multiple providers concurrently
3. **Providers** fetch and transform data using Pydantic models
4. **Workflow** assembles results and handles errors
5. **Reflex Page** receives structured data for presentation

## Usage Examples

### Basic Provider Usage

```python
from app.providers.yahoo import create_yahoo_info_provider

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
if result.success:
    data = result.data
    print(f"Company: {data.company_name}")

# Read-only historical cache example (async only)
# Reads from cache date 20250805 without writing
cache_result = await provider.get_data("AAPL", cache_date="20250805")
if cache_result.success:
    old_data = cache_result.data  # Data type depends on provider
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
from pydantic import BaseModel, Field
from app.providers.cache import cache
from app.providers.base import BaseProvider, ProviderType

class CustomDataModel(BaseModel):
    """Pydantic model for API response."""
    ticker: str = Field(alias="symbol")
    price: float = Field(alias="last_price")
    volume: int = Field(alias="trade_volume")
    
    model_config = {"populate_by_name": True, "extra": "ignore"}

class CustomProvider(BaseProvider[BaseModel]):
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.CUSTOM

    @cache
    async def _fetch_data(self, query: str | None, **kwargs) -> BaseModel:
        # Fetch raw data from API
        raw_data = await self._make_api_request(query)
        
        # Transform using Pydantic model with aliases
        return CustomDataModel(**raw_data)
```

## Configuration Options

### Provider Configuration

```python
config = ProviderConfig(
    timeout=30.0,           # Request timeout
    retries=3,              # Retry attempts
    retry_delay=1.0,        # Delay between retries
    rate_limit=1.0,         # Requests per second
    extra_config={          # Provider-specific settings
        "period": "1y",
        "interval": "1d"
    }
)
```

### Pydantic Model Configuration

```python
from pydantic import BaseModel, Field

class ApiDataModel(BaseModel):
    """Configure data transformation with Pydantic Field aliases."""
    price: float = Field(alias="regularMarketPrice", default=0.0)
    volume: int = Field(alias="volume", default=0)
    name: str = Field(alias="companyName", default="")
    
    # Configure model behavior
    model_config = {
        "populate_by_name": True,  # Accept both field names and aliases
        "extra": "ignore",         # Ignore extra fields from API
        "validate_default": True   # Validate default values
    }
```

## Error Handling Strategy

### Provider-Level Errors
- Network timeouts and connection errors
- API rate limiting and quota exceeded
- Invalid ticker symbols
- Malformed response data

### Data Modeling Errors
- Invalid JSON/data format
- Pydantic validation failures
- Type conversion errors
- Missing required fields

### Workflow-Level Errors
- Provider initialization failures
- Concurrent execution limits
- Resource exhaustion
- Unexpected exceptions

## Performance Characteristics

### Concurrency
- **Provider-Level Concurrency**: Semaphore-based limiting (default: 10, configurable per provider) to manage the rate of requests to external APIs and prevent rate-limiting errors. This is a form of "API politeness."
- **Workflow-Level Concurrency**: Orchestrated by the workflow engine in `app/flows` (e.g., LlamaIndex's `num_workers`). This layer decides *what* to run in parallel, while the provider layer manages *how* to execute its specific calls safely.
- **Thread Pool for Blocking Operations**: Ensures that synchronous, blocking calls (like the `yfinance` library) do not halt the entire async event loop.
- **Async/await Throughout**: Maximizes I/O efficiency.

### Memory Management
- **Pydantic Model Efficiency**: Efficient object creation and validation with minimal memory overhead
- **Streaming Data Processing**: For large datasets, consider using streaming responses where possible
- **Garbage Collection Friendly**: The design avoids circular references and manages object lifecycles cleanly
- **Connection Pooling**: `httpx.AsyncClient` is used to manage and reuse HTTP connections efficiently

### Execution Times
Based on testing with the enhanced providers:
- Single ticker (3 providers): ~2-4 seconds
- Multiple tickers (3 providers each): ~3-6 seconds (parallel execution)
- Individual provider calls: ~0.5-2 seconds each

## Future Enhancements

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

### Implementation Guidelines

1. **Import patterns:**
   ```python
   from app.providers.yahoo import create_yahoo_info_provider
   from app.providers.base import BaseProvider, ProviderType, ProviderResult
   from pydantic import BaseModel, Field
   ```

2. **Provider usage:**
   ```python
   provider = create_yahoo_info_provider()
   result = await provider.get_data("AAPL")
   if result.success:
       data = result.data
   ```

3. **Error handling:**
   ```python
   result = await provider.get_data("AAPL")
   if not result.success:
       print(f"Error: {result.error_message}")
       print(f"Error code: {result.error_code}")
   ```

This improved architecture provides a solid foundation for scaling the financial data infrastructure while maintaining excellent error handling, performance, and developer experience.
