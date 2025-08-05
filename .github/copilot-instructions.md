# Instructions for Code Agents

This file provides guidance to code agents when working with code in this repository.

## Use Cases

FinApp is financial market analysis and portfolio management application covering:
    - Market analysis
    - Stock/ETF comparison
    - Stock/ETF screening
    - Portfolio optimization
    - Backtesting

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
 make reset   # Deletes the .web directory

### Running the Application
```bash
make run      # Run the Reflex application using '.venv/bin/reflex run'
```

### Code Quality
```bash
make format   # Format code with black (app/, tests/, *.py)
make lint     # Run pylint on app/, tests/, *.py (disable R,C categories)
make all      # Run install, lint, test, and format in sequence
```

### Testing
```bash
make test     # Run pytest on test_hello.py (note: test file may not exist yet)
```

## Project Architecture

This is a **Reflex web application**. Reflex is a Python web framework for building reactive web apps.
Documentation at https://reflex.dev/docs/.
Examples at:
    - https://github.com/reflex-dev/reflex-examples
    - https://github.com/reflex-dev/templates
    - https://github.com/reflex-dev/sidebar-template

### Core Structure

Core structure follows the principles of https://reflex.dev/docs/advanced-onboarding/code-structure/.
- **rxconfig.py**: Reflex configuration with app name "app" and plugins (Sitemap, TailwindV4)
- **app/app.py**: Main application entry point with Reflex components
- **app/components/**: Application UI components
- **app/flows/**: Application workflows for background processing
- **app/lib/**: Application logic (business layer)
- **app/models/**: Application models (persistence layer)
- **app/pages/**: Application pages (presentation layer)
- **app/states/**: Application states
- **app/templates/**: Application templates
- **assets/**: Static assets like favicon
- **data/**: Data files (mainly downloaded data cached for analysis)
- **tests/**: Unit tests

### Application Components
- **State Management**: Uses `rx.State` class for reactive state
- **UI Components**: Built with Reflex components (`rx.container`, `rx.vstack`, etc.)
- **Routing**: Single page app with index route
- **Styling**: TailwindV4 plugin enabled for CSS styling

### Dependencies
- **Core**: reflex
- **Financial/Data**: bt, cvxpy, numpy, pandas, yfinance, riskfolio-lib
- **Workflows**: llama-index-workflows
- **Visualization**: matplotlib, plotly
- **Persistence**: duckdb, json, csv, parquet
- **Development**: black, pylint, flake8, pytest, pytest-cov

### Key Notes
- Virtual environment located at `.venv/`
- All commands should be run with `.venv/bin/` prefix (see Makefile)
- Linting and formatting covers `app/`, `tests/`, and root-level Python files
