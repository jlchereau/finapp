# Instructions for Code Agents

This file provides guidance to code agents when working with code in this repository.

## Use Cases

FinApp is a financial market analysis and portfolio management application covering:
    - **Market Analysis**: Real-time and historical market data analysis
    - **Stock/ETF Comparison**: Side-by-side comparison of financial instruments
    - **Stock/ETF Screening**: Filter and search financial instruments by criteria
    - **Portfolio Optimization**: Mathematical optimization using modern portfolio theory
    - **Backtesting**: Historical performance testing of investment strategies

## Development Commands

### Setup
```bash
make install  # Install dependencies from requirements-dev.txt
```
 
 ### Building the Application
 ```bash
 make build   # Compile the Reflex application using '.venv/bin/reflex compile'
 ```

### Resetting the Application
 ```bash
 make reset   # Delete the .web directory
 ```

### Running the Application
```bash
make run      # Run the Reflex application using '.venv/bin/reflex run'
```

### Code Quality
```bash
make format   # Format code with black (app/, tests/, *.py)
make lint     # Run flake8, pylint, and pyrefly on app/, tests/, *.py
make all      # Run install, format, lint, test in sequence
```

**IMPORTANT - Tool Configuration**: 
- Uses `pyproject.toml` for centralized tool configuration (black, pylint, pytest, coverage, pyrefly)
- Pylint configured with `pylint-per-file-ignores` plugin to disable specific checks on test files
- Test files automatically ignore `protected-access`, `redefined-outer-name`, and other pytest-specific violations
- Flake8 uses separate `.flake8` file (v7.3.0 doesn't support pyproject.toml natively)
- **NEVER use `# pylint: disable=*` comments in code** - these should be considered LAST RESORT only when no coding alternative exists
- Always prefer fixing the underlying code issue rather than suppressing the warning
- Per-file ignores in `pyproject.toml` are acceptable for systematic test-specific patterns that cannot be avoided

### Testing
```bash
make test     # Run pytest with coverage on tests/ directory (-vv verbosity)
```

Code agents can consider these make commands as safe to execute without requiring additional confirmation, as they are designed to maintain the integrity of the codebase and ensure a smooth development workflow.

## Project Architecture

This is a **Reflex web application**. Reflex is a Python web framework for building reactive web apps.
Documentation at https://reflex.dev/docs/.
Examples at:
    - https://github.com/reflex-dev/reflex-examples
    - https://github.com/reflex-dev/templates
    - https://github.com/reflex-dev/sidebar-template

## Coding Guidelines

Code is written in Python >=3.12 (configured in pyproject.toml), leveraging Reflex components for the UI and state management. Compilation generates a React single-page application (SPA) with a Python backend. Follow [documented reflex conventions](https://reflex.dev/docs) and recent Python best practices, with particular emphasis on the [modular component architecture](#modular-component-architecture) and [decentralized event handlers](https://reflex.dev/docs/events/decentralized-event-handlers/) patterns. There is no need to support any version of Python <3.12.

Use `make build` to compile the application and check for errors. This will generate a `.web/` directory containing the compiled application.

Use `make format` to format the code with `black`, ensuring consistent style across the codebase. Use `make lint` to run `flake8`, `pylint`, and `pyrefly` for comprehensive static code analysis, focusing on code quality and potential issues. Run `.venv/bin/python -m pytest` on the scope of your changes while iterating. Use `make test` to run all unit tests with coverage reporting ensuring code correctness and functionality (can take several minutes). Do not test code that does not compile.

## Folder Structure

Core structure follows the principles of https://reflex.dev/docs/advanced-onboarding/code-structure/.
- **rxconfig.py**: Reflex configuration with app name "app" and plugins (Sitemap, TailwindV4)
- **app/app.py**: Main application entry point with Reflex components
- **app/flows/**: LlamaIndex workflows for background processing (markets/, compare/, test.py)
- **app/lib/**: Application utilities and business logic
  - **logger.py**: Custom CSV logging system with date-based storage
  - **storage.py**: Unified date-based storage utilities for cache and data management
  - **settings.py**: Application configuration management
  - **finance.py**: Financial calculation utilities
- **app/providers/**: Data provider system for external APIs
  - **base.py**: Base provider class with rate limiting and error handling
  - **cache.py**: Decorator for caching provider responses (JSON/Parquet)
  - **yahoo.py, blackrock.py, tipranks.py, zacks.py**: Active data providers
  - **headers.py**: HTTP request headers and user agents
  - **\*.py.txt**: Inactive/template provider implementations
- **app/pages/**: Application pages (presentation layer)
  - **index.py**: Landing page with markdown content
  - **cache.py**: Cache management and log viewing interface
  - **screen.py**: Stock/ETF screening page
  - **portfolio.py, backtest.py**: Portfolio management pages (single files)
  - **test.py**: Development testing interface
  - **markets/**: Modular markets analysis page
    - **page.py**: Main markets page layout with computed vars
    - **buffet_indicator.py, vix_chart.py, crude_oil.py**: Market component charts
    - **currency_chart.py, crypto_chart.py, precious_metals.py**: Asset component charts
    - **bloomberg_commodity.py, msci_world.py, yield_curve.py**: Index component charts
  - **compare/**: Modular comparison analysis page
    - **page.py**: Main compare page layout with computed vars
    - **returns_chart.py, volatility_chart.py, volume_chart.py**: Time series components
    - **rsi_chart.py**: Technical indicator components
    - **metrics.py**: Combined fundamental metrics component
  - **optimize/**: Modular portfolio optimization page
    - **page.py**: Main optimization page layout
    - **card1.py, card2.py**: Optimization component cards
- **app/templates/**: Application layout templates
- **assets/**: Static assets and markdown content
- **data/**: Date-organized cache storage (YYYYMMDD folders)
  - **YYYYMMDD/log.csv**: Daily application logs
  - **YYYYMMDD/\*.json**: Cached JSON responses from providers
  - **YYYYMMDD/\*.parquet**: Cached DataFrame data
- **temp/**: Temporary workspace for experiments (in .gitignore)
- **tests/**: Comprehensive unit test suite (627 tests)
  - **conftest.py**: Global test configuration with automatic cache isolation

## Application Components
- **State Management**: Uses `rx.State` class for reactive state with page-specific state classes
- **UI Components**: Built with Reflex components (`rx.container`, `rx.vstack`, `rx.data_table`, etc.)
- **Routing**: Multi-page application with routes for each functional area
- **Styling**: TailwindV4 plugin enabled for CSS styling
- **Data Providers**: Modular provider system with caching for external financial APIs
- **Logging System**: Custom CSV logger with structured logging to date-based folders
- **Storage System**: Unified date-based storage for cache management and data persistence
- **Cache Management**: Web interface for viewing logs and managing cached data
- **Workflows**: LlamaIndex-based workflows for complex background processing
- **Modular Components**: Self-contained chart and feature components with individual state management
- **Decentralized Event Handlers**: Component communication using Reflex decentralized event patterns

## Dependencies
- **Core**: reflex 0.8.12, reflex-pyplot
- **Finance**: bt 1.1.2, cvxpy 1.7.3, yfinance 0.2.66, riskfolio-lib 7.0.1, factor_analyzer 0.5.1
- **Data Processing**: numpy 2.3.3, pandas 2.3.2, httpx 0.28.1, pydantic 2.11.9, pydantic-settings 2.10.1
- **Web Scraping**: beautifulsoup4 4.13.5, lxml 6.0.2
- **Serialization**: orjson 3.11.3, pyarrow 21.0.0 (parquet)
- **Templates**: Jinja2 3.1.6
- **Workflows**: llama-index-workflows 2.2.2, llama-index-core 0.14.2
- **Visualization**: matplotlib 3.10.6, plotly 6.3.0
- **Database**: duckdb 1.4.0
- **Build Tools**: pybind11 3.0.1
- **Development**: black 25.9.0, flake8 7.3.0, pylint 3.3.8, pylint-per-file-ignores 2.0.3, pyrefly 0.34.0, pytest 8.4.2, pytest-asyncio 1.2.0, pytest-cov 7.0.0, ipykernel 6.30.1

## Key Architecture Features

### Modular Component Architecture
FinApp uses a sophisticated modular component architecture following Reflex best practices and the pattern established in `markets/`, `compare/`, and `optimize/` pages.

#### **Component Pattern Structure**
Each self-contained component follows this standardized pattern:

```python
# Example: app/pages/compare/returns_chart.py

class ReturnsChartState(rx.State):
    """Component-specific state management."""
    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    @rx.event
    async def update_returns_chart_data(self, tickers: List[str], base_date: datetime):
        """Component logic calling exactly one flow."""
        result = await fetch_returns_data(tickers=tickers, base_date=base_date)
        # Process and update component state

@rx.event
async def update_returns_chart(state: ReturnsChartState, tickers: List[str], base_date: datetime):
    """Decentralized event handler for main page coordination."""
    # IMPORTANT: fix_datetime() is required for datetime parameters in decentralized event handlers
    # due to JSON serialization issues between JavaScript and Python (see reflex-dev/reflex#5811)
    # This pattern may be needed for other complex data types that don't support JSON serialization
    base_date = fix_datetime(base_date)
    await state.update_returns_chart_data(tickers, base_date)

def returns_chart() -> rx.Component:
    """Component function with conditional rendering."""
    return rx.cond(
        ReturnsChartState.loading,
        rx.center(rx.spinner(), height="300px"),
        rx.plotly(data=ReturnsChartState.chart_figure, width="100%", height="300px"),
    )
```

#### **Architectural Principles**
1. **One Component, One Flow**: Each component calls exactly one workflow for data processing
2. **Self-Contained State**: Components manage their own state classes inheriting from `rx.State`
3. **Decentralized Events**: Components expose `@rx.event` handlers for main page coordination
4. **Computed Vars Pattern**: Main pages use `@rx.var` for shared computed properties (e.g., `base_date`)
5. **Component Function Export**: Each module exports both a component function and update handler
6. **Data Type Serialization**: Use helper functions like `fix_datetime()` for complex data types in decentralized event handlers due to JSON serialization limitations between JavaScript and Python (see [reflex-dev/reflex#5811](https://github.com/reflex-dev/reflex/issues/5811))
7. **Event Handler Decoration**: ALL event handlers must be decorated with `@rx.event` to avoid pyrefly type checking issues and ensure proper Reflex event system integration

#### **Page Structure Pattern**
Modular pages follow this organization:

```
app/pages/compare/
├── __init__.py          # Exports: from .page import page
├── page.py              # Main layout with computed vars and coordination logic
├── returns_chart.py     # Self-contained returns component
├── volatility_chart.py  # Self-contained volatility component
├── volume_chart.py      # Self-contained volume component
├── rsi_chart.py         # Self-contained RSI component
└── metrics.py           # Combined metrics component (multiple tables, one flow)
```

#### **Main Page Coordination**
Main page files (`page.py`) coordinate components using computed vars and workflow coordination methods:

```python
class CompareState(rx.State):
    @rx.var
    def base_date(self) -> datetime:
        """Computed var replacing _get_base_date() method."""
        return calculate_base_date(self.period_option)

    @rx.event
    def set_period_option(self, option: str):
        """Set period option and trigger all component updates."""
        self.period_option = option
        yield rx.toast.info(f"Changed time period to {option}")
        yield CompareState.run_workflows

    @rx.event
    def run_workflows(self):
        """
        Coordinate all component updates - NOT async.

        CRITICAL: This method CANNOT be async because it returns a list of event handlers
        that Reflex needs to process. Making it async breaks the event system.
        The individual component updates are async, but this coordination method is not.
        """
        tickers = self.selected_tickers
        base_date = self.base_date
        return [
            # Note: These are NOT awaited - they return event handlers for Reflex to process
            update_returns_chart(tickers=tickers, base_date=base_date),
            update_volatility_chart(tickers=tickers, base_date=base_date),
            update_volume_chart(tickers=tickers, base_date=base_date),
            update_rsi_chart(tickers=tickers, base_date=base_date),
        ]
```

#### **Component Testing Pattern**
Each component includes dedicated unit tests verifying:
- State initialization and methods
- Component function rendering
- Import compatibility with main application
- Event handler functionality

#### **Benefits of Modular Architecture**
- **Maintainability**: Small, focused files (~40-130 lines vs 1000+ line monoliths)
- **Testability**: Individual component testing with clear boundaries
- **Reusability**: Components can be easily moved or reused across pages
- **Parallel Development**: Different components can be developed independently
- **Performance**: Selective component updates without full page re-renders

### Date-Based Storage System (`app/lib/storage.py`)
- **DateBasedStorage class**: Manages data in YYYYMMDD-organized folders
- **Configurable Base Path**: Uses `PROVIDER_CACHE_ROOT` setting for storage location
- **Cache Management**: Automatic file organization with cleanup utilities
- **Storage Methods**: `get_date_folder()`, `get_file_path()`, `list_date_folders()`, `delete_date_folder()`
- **Log Integration**: `get_log_data()` for reading structured CSV logs

### Application Settings (`app/lib/settings.py`)
- **Settings class**: Centralized configuration management using pydantic-settings
- **Environment Priority**: Environment variables > .env file > defaults
- **Storage Configuration**: `PROVIDER_CACHE_ROOT` setting controls cache and data storage location
- **Debug Level Control**: `DEBUG_LEVEL` setting filters log output (debug, info, warning, error)
- **Path Validation**: Automatic project root detection for relative paths
- **CI/CD Support**: Environment variables can override defaults for testing environments

### Custom Logging System (`app/lib/logger.py`)
- **CSVLogger class**: Structured logging to date-based CSV files
- **Log Format**: timestamp, level, message, context, file, function, params
- **Context Detection**: Automatically identifies workflow vs app context
- **Thread-Safe**: Concurrent logging support with file locking
- **Debug Level Filtering**: Respects `DEBUG_LEVEL` setting for selective logging

### Data Provider System (`app/providers/`)
- **Base Provider** (`base.py`): Abstract base with rate limiting, retries, error handling
- **Cache Decorator** (`cache.py`): Automatic caching of API responses (JSON/Parquet)
- **Active Providers**: Yahoo Finance, BlackRock, TipRanks, Zacks
- **Pydantic Models**: Type-safe data validation with field aliases for API mapping
- **HTTP Management**: Custom headers, user agents, timeout handling

### **CRITICAL: Provider Period Limitation Anti-Pattern**
**⚠️ NEVER pass user-selected periods to provider calls - this breaks period selection functionality!**

#### **The Problem**
When providers accept period/date limitation parameters (like `observation_start`, `period`, `start_date`), and workflows pass user-selected periods, the provider caches limited datasets. This creates a critical bug where:

1. **User selects 1Y period** → Provider fetches 1 year of data → Caches 1 year dataset
2. **User later selects 10Y period** → Provider loads cached 1 year dataset → Period selection appears broken

#### **Real Example: Buffet Indicator Bug**
```python
# ❌ WRONG - This broke period selection
provider_result = await self.fred_provider.get_data(
    query="GDP",
    observation_start=base_date.strftime("%Y-%m-%d"),  # Limited to user period!
)

# ✅ CORRECT - Fetch maximum data, filter in workflow
provider_result = await self.fred_provider.get_data(
    query="GDP",  # No period limitations
)
```

#### **Mandatory Provider Pattern**
**ALL providers MUST follow this pattern:**

```python
# ✅ CORRECT PROVIDER PATTERN
class MyProvider(BaseProvider):
    async def _fetch_data(self, query: str, **kwargs) -> DataFrame:
        # CRITICAL: Fetch MAXIMUM historical data by default
        # NEVER pass user periods to external APIs
        return external_api.get_data(
            symbol=query,
            period="max",  # Always maximum
            # observation_start=kwargs.get("observation_start"),  # ❌ NEVER
        )

# ✅ CORRECT WORKFLOW PATTERN
async def my_workflow(base_date: datetime, period: str):
    # 1. Fetch FULL dataset from provider
    provider_result = await provider.get_data(query="SYMBOL")
    full_data = provider_result.data

    # 2. Filter in workflow AFTER getting full data
    filtered_data = ensure_minimum_data_points(
        data=full_data,
        base_date=base_date,  # User period applied HERE
        min_points=2,
    )
```

#### **Provider Implementation Rules**
1. **Default to Maximum Data**: All providers MUST fetch maximum available historical data by default
2. **No User Periods**: NEVER accept user-selected periods in provider calls from workflows
3. **Workflow Filtering**: Period filtering happens in workflows AFTER data collection
4. **Cache Full Datasets**: Cache complete historical data, not limited subsets
5. **Document Pattern**: Include comments explaining why period limitations are dangerous

#### **Correct Examples in Codebase**
- **YahooHistoryProvider**: Defaults to `period="max"`, explicit comments about workflow filtering
- **BuffetIndicatorWorkflow** (after fix): Fetches full GDP + Wilshire data, filters via `ensure_minimum_data_points()`

#### **Anti-Pattern Detection**
Watch for these dangerous patterns in provider calls:
- `observation_start=base_date` (FRED)
- `period=user_selected_period` (Yahoo)
- `start_date=calculated_date` (Generic APIs)
- Any parameter that limits historical data based on user input

#### **Provider Validation Checklist**
Before implementing any provider:
- ✅ Fetches maximum available historical data by default
- ✅ Period parameters (if any) default to maximum range
- ✅ Comments warn against passing user periods from workflows
- ✅ Factory functions don't accept period limitations
- ✅ Tests verify full historical data is cached

### Provider Design Patterns (`app/providers/`)
**CRITICAL**: All providers returning Pydantic models must follow this standardized pattern for API change detection and data validation consistency.

#### **Pydantic Model Provider Pattern**
- **Core Principle**: API change detection through strict validation without default values
- **Design Goal**: If external APIs change structure, Pydantic ValidationError should alert us immediately
- **Implementation**: Use field validators for type conversion, never manual data transformation

#### **Required Model Structure**
```python
class ExampleModel(BaseModel):
    # CRITICAL: Include this comment in every Pydantic model
    # Important: Do not set default values for required fields.
    # This would prevent us from detecting API changes.
    # If the API changes and fields are missing,
    # Pydantic should raise a validation error.
    
    required_field: str = Field(alias="apiFieldName")
    optional_field: str | None = Field(default=None, alias="optionalApi")
    
    @field_validator('required_field', mode='before')
    @classmethod
    def convert_null_strings(cls, v):
        if v == "NULL" or v is None:
            raise ValueError("Required field cannot be NULL")
        return str(v)
```

#### **Provider Implementation Guidelines**
- **Minimal Logic**: Fetch data and pass directly to Pydantic model
- **No Manual Conversion**: Remove manual type conversion blocks (defeats API change detection)
- **Clean Separation**: Data fetching in provider, validation in model
- **Error Propagation**: Let ValidationError bubble up for missing/invalid fields

#### **Examples by Type**
- **Pydantic Models**: `YahooInfoProvider`, `ZacksProvider`, `TipranksDataProvider`
  - Use for: Structured API responses with known fields
  - Pattern: Fetch → Transform keys → Validate with Pydantic
- **DataFrame Returns**: `YahooHistoryProvider`, `BlackrockHoldingsProvider`, `FredSeriesProvider`  
  - Use for: Time series data, tabular data, flexible structures
  - Pattern: Fetch → Process → Return DataFrame

#### **Required Testing Pattern**
```python
def test_model_missing_required_fields_raises_validation_error(self):
    with pytest.raises(ValidationError) as exc_info:
        ExampleModel(**incomplete_data)
    assert "Field required" in str(exc_info.value)

def test_model_null_values_raise_validation_error(self):
    test_data_with_nulls = {"field": None, ...}
    with pytest.raises(ValidationError) as exc_info:
        ExampleModel(**test_data_with_nulls)
    assert "cannot be NULL" in str(exc_info.value)
```

#### **Anti-Patterns to Avoid**
- ❌ Setting `default=` values on required fields (hides API changes)
- ❌ Manual type conversion in provider methods (redundant with Pydantic)
- ❌ Suppressing ValidationError (defeats change detection purpose)
- ❌ Using incomplete mock data in tests (should reflect real API structure)

### Cache Management Interface (`app/pages/cache.py`)
- **Dynamic Directory Loading**: Lists available cache dates from storage
- **Log Viewing**: Interactive data table with search, sort, pagination
- **Cache Deletion**: Confirmation dialog with secure folder removal
- **Toast Notifications**: User feedback for all operations
- **Most Recent First**: Both directories and log entries sorted by recency

## Environment Configuration

### Storage Location
The application uses `PROVIDER_CACHE_ROOT` setting to determine where cache and data files are stored:
- **Default**: `data/` folder in project root (for local development)
- **Override**: Set environment variable `PROVIDER_CACHE_ROOT` to custom path
- **CI/CD**: GitHub Actions sets this to `temp/` to avoid conflicts

## Testing Guidelines

### Cache Isolation for Tests
**CRITICAL**: All tests are automatically isolated to prevent contamination of production cache directory.

Cache isolation is handled globally via `tests/conftest.py` which automatically:
- **Provider Cache Isolation**: Redirects `PROVIDER_CACHE_ROOT` to `temp/pytest/` (instead of `data/`)
- **Workflow Cache Isolation**: Disables `FLOW_CACHE_ENABLED` to prevent workflow caching during tests
- **Complete Protection**: Ensures no test data appears in production `data/` directory
- **Automatic Coverage**: Works for all tests without requiring individual fixtures
- **CI/CD Compatible**: Respects environment overrides for GitHub Actions

This dual isolation protects both file-based provider caching and in-memory workflow caching.

### Running Tests Safely
All tests are automatically isolated via global configuration:

```bash
# Safe: Run specific test modules (automatically isolated)
.venv/bin/python -m pytest tests/flows/compare/ -vv

# Safe: Run all tests (automatically isolated)
make test

# All cache operations are automatically redirected:
# - Provider cache: temp/pytest/ instead of data/
# - Workflow cache: disabled entirely during tests
```

### Cache Contamination Prevention
With global cache isolation, test data contamination is automatically prevented:
- **Provider cache**: Test files stored in `temp/pytest/`, production files in `data/`
- **Workflow cache**: Completely disabled during tests to ensure fresh data
- **Production protection**: `data/` directory remains untouched by test runs
- **Real vs test data**: Real stock data has DatetimeIndex, test data often has RangeIndex
- **Troubleshooting**: If charts show flat lines, check for manual modifications in `data/`

### Global Configuration Details
The `tests/conftest.py` file uses `pytest_configure()` hook to set:
```python
# Redirect provider cache to isolated directory
os.environ["PROVIDER_CACHE_ROOT"] = "temp/pytest/"

# Disable workflow caching entirely during tests  
os.environ["FLOW_CACHE_ENABLED"] = "False"
```

### Debug Level Control  
The `DEBUG_LEVEL` setting controls logging verbosity:
- **Values**: `debug` (all), `info` (info+), `warning` (warning+), `error` (error only)
- **Default**: `debug` for development
- **Override**: Set environment variable `DEBUG_LEVEL` or add to `.env` file

### Testing Considerations
- **Environment isolation**: Tests that verify default behavior should mock environment variables to ensure consistent results
- **CI isolation**: Use `mock.patch.dict(os.environ, {}, clear=True)` to isolate tests from CI environment
- **Settings reload**: Reload settings module when testing configuration changes: `importlib.reload(settings)`
- **No individual fixtures needed**: Global cache isolation eliminates the need for individual test fixtures
- **New test files**: Automatically inherit cache isolation without any special configuration

## Exception Architecture

FinApp uses a clean 3-layer exception architecture with clear separation of concerns:

### **Layer 1: Providers** (`app/providers/`)
- `RetriableProviderException`: Transient provider errors that can be retried (inherit from FinAppException)
- `NonRetriableProviderException`: Permanent provider errors that should not be retried (inherit from FinAppException)
- **Usage**: Raised by data providers for external API failures
- **Features**: Include error IDs, user messages, and context for debugging

### **Layer 2: Workflows** (`app/flows/`)
- `WorkflowException`: High-level workflow execution failures (inherit from FinAppException)
- **Usage**: Catch provider exceptions and internal logic failures, wrap everything in WorkflowException
- **Pattern**: `try/catch` → `raise WorkflowException(workflow="name", step="step", message="...")`
- **Internal failures**: Use generic `Exception` for internal data validation, caught by workflow wrapper

### **Layer 3: Pages** (`app/pages/`)
- `PageInputException`: Page-level input validation errors (inherit from FinAppException)
- `PageOutputException`: Page-level output/rendering failures (inherit from FinAppException)
- **Usage**: Input validation, chart generation, data rendering failures
- **Pattern**: Replace with `output_type` instead of `chart_type` or `operation`

### **Base Exception** (`app/lib/exceptions.py`)
- `FinAppException`: Base class with error_id, user_message, technical message, context
- **Features**: UUID error tracking, structured logging via to_dict(), user-friendly messages

### **Exception Usage Guidelines**
- **Clear boundaries**: Each layer handles its specific concerns
- **No redundancy**: Removed DataFetchException and DataProcessingException
- **Consistent naming**: PageInputException/PageOutputException for page scope
- **Proper inheritance**: All custom exceptions inherit from FinAppException
- **Context tracking**: All exceptions include relevant context for debugging

## Key Notes
- Always maintain unit tests in line with code changes, ensuring all new features and bug fixes are covered, while striking the right balance between good coverage and high maintainability (low complexity).
- Virtual environment is located at `.venv/`. Accordingly the environment should be activated using `source .venv/bin/activate` or commands should be run with the `.venv/bin/` prefix (see Makefile).
- Always format, lint (and fix), build (and fix) and test (and fix) your code. Iterate with commands limited to the scope of your changes (except for format and build which are fast). Once done, run `make format`, `make lint`, `make build` and `make test` to check for regressions across the entire application.
- Avoid hiding lint issues with `# flake8: noqa]`, `# pylint: disable=*`, `# pyright: ignore[*]` or `# pyrefly: ignore[*]` comments as this is not fixing.
- Ask me to run `make run` to start the application, as this is a blocking command that is hard to kill properly.
- Do not use python `tempfile`. Use workspace @temp/ (in .gitignore) for unit test storage and experimentations (it makes it easier to monitor and clear ghost files manually).