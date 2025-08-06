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

Note: always run `make build` after making changes to the code to ensure the application is compiling without errors. Sometimes the compilation gets stuck, in which case you can reset the application with `make reset` and then run `make build` again.

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

## Application Components
- **State Management**: Uses `rx.State` class for reactive state
- **UI Components**: Built with Reflex components (`rx.container`, `rx.vstack`, etc.)
- **Routing**: Single page app with index route
- **Styling**: TailwindV4 plugin enabled for CSS styling

## Dependencies
- **Core**: reflex
- **Finance**: bt, cvxpy, yfinance, riskfolio-lib
- **Data**: numpy, pandas, httpx, jmespath, pydantic
- **Templates**: Jinja2
- **Workflows**: llama-index-workflows
- **Visualization**: matplotlib, plotly
- **Persistence**: duckdb, pyarrow (parquet), json, csv
- **Development**: black, flake8, pylint, pytest, pytest-cov

## Key Notes
- Virtual environment located at `.venv/`
- All commands should be run with `.venv/bin/` prefix (see Makefile)
- Linting and formatting covers `app/`, `tests/`, and root-level Python files
