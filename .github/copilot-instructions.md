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
make lint     # Run pylint on app/, tests/, *.py (disable R,C categories)
make all      # Run install, format, lint, test in sequence
```

### Testing
```bash
make test     # Run pytest on app/ and tests/ directories
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

Code is written in Python >=3.10 (a requirement for Reflex), leveraging Reflex components for the UI and state management. Compilation generates a React single-page application (SPA) with a python backend. Follow [documented reflex conventions](https://reflex.dev/docs) and recent Python best practices considering there is no need to support any version of Python <3.10.

Use `make build` to compile the application and check for errors. This will generate a `.web/` directory containing the compiled application.

Use `make format` to format the code with `black`, ensuring consistent style across the codebase. Use `make lint` to run `pylint` for static code analysis, focusing on code quality and potential issues. Run `.venv/bin/python -m pytest` on the scope of your changes while iterating. Use `make test` to run all unit tests ensuring code correctness and functionality (can take sevaral minutes). Do not test code that does not compile.

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
- **tests/**: Comprehensive unit test suite (196 tests)

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
- **Development**: black, flake8, pylint, pytest, pytest-asyncio, pytest-cov, ipykernel

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

### Debug Level Control  
The `DEBUG_LEVEL` setting controls logging verbosity:
- **Values**: `debug` (all), `info` (info+), `warning` (warning+), `error` (error only)
- **Default**: `debug` for development
- **Override**: Set environment variable `DEBUG_LEVEL` or add to `.env` file

### Testing Considerations
- Tests that verify default behavior should mock environment variables to ensure consistent results
- Use `mock.patch.dict(os.environ, {}, clear=True)` to isolate tests from CI environment
- Reload settings module when testing configuration changes: `importlib.reload(settings)`

## Key Notes
- Always maintain unit tests in line with code changes, ensuring all new features and bug fixes are covered, while striking the right balance between coverage and maintainability.
- Virtual environment is located at `.venv/`. Accordingly the environment should be activated using `source .venv/bin/activate` or commands should be run with the `.venv/bin/` prefix (see Makefile).
- Always format, lint (and fix), build (and fix) and test (and fix) your code. Iterate with commands limited to the scope of your changes (except for format and build which are fast). Once done, run `make format`, `make lint`, `make build` and `make test` to check for regressions across the entire application.
- Ask me to run `make run` to start the application, as this is a blocking command that is hard to kill properly.
- Do not use python `tempfile`. Use workspace @temp/ (in .gitignore) for unit test storage and experimentations (it makes it easier to monitor and clear ghost files manually)