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

Code is written in Python >=3.12 (configured in pyproject.toml), leveraging Reflex components for the UI and state management. Compilation generates a React single-page application (SPA) with a python backend. Follow [documented reflex conventions](https://reflex.dev/docs) and recent Python best practices considering there is no need to support any version of Python <3.12.

Use `make build` to compile the application and check for errors. This will generate a `.web/` directory containing the compiled application.

Use `make format` to format the code with `black`, ensuring consistent style across the codebase. Use `make lint` to run `flake8`, `pylint`, and `pyrefly` for comprehensive static code analysis, focusing on code quality and potential issues. Run `.venv/bin/python -m pytest` on the scope of your changes while iterating. Use `make test` to run all unit tests with coverage reporting ensuring code correctness and functionality (can take several minutes). Do not test code that does not compile.

## Folder Structure

Core structure follows the principles of https://reflex.dev/docs/advanced-onboarding/code-structure/.
- **rxconfig.py**: Reflex configuration with app name "app" and plugins (Sitemap, TailwindV4)
- **app/app.py**: Main application entry point with Reflex components
- **app/flows/**: LlamaIndex workflows for background processing (markets.py, test.py)
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
  - **markets.py, compare.py, screen.py**: Financial analysis pages
  - **portfolio.py, optimize.py, backtest.py**: Portfolio management pages
  - **test.py**: Development testing interface
- **app/templates/**: Application layout templates
- **assets/**: Static assets and markdown content
- **data/**: Date-organized cache storage (YYYYMMDD folders)
  - **YYYYMMDD/log.csv**: Daily application logs
  - **YYYYMMDD/\*.json**: Cached JSON responses from providers
  - **YYYYMMDD/\*.parquet**: Cached DataFrame data
- **temp/**: Temporary workspace for experiments (in .gitignore)
- **tests/**: Comprehensive unit test suite (282 tests)
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

## Dependencies
- **Core**: reflex, reflex-pyplot
- **Finance**: bt, cvxpy, yfinance, riskfolio-lib, factor_analyzer
- **Data Processing**: numpy, pandas, httpx, pydantic, pydantic-settings
- **Web Scraping**: beautifulsoup4, lxml
- **Serialization**: orjson, pyarrow (parquet)
- **Templates**: Jinja2
- **Workflows**: llama-index-workflows
- **Visualization**: matplotlib, plotly
- **Database**: duckdb
- **Build Tools**: pybind11
- **Development**: black, flake8, pylint, pylint-per-file-ignores, pyrefly, pytest, pytest-asyncio, pytest-cov, ipykernel

## Key Architecture Features

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
.venv/bin/python -m pytest tests/flows/test_compare.py -vv

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