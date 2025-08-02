 # AGENTS.md

 This file provides guidance to coding agents when working with code in this repository.

 ## Development Commands

 ### Setup
 ```bash
 make install  # Install dependencies from requirements-dev.txt
 ```

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
 make test     # Run pytest suite
 ```

 ## Project Architecture

 This is a **Reflex web application**. Reflex is a Python web framework for building reactive web apps.
 See https://reflex.dev/docs/ for reference.

 ### Core Structure
 - **rxconfig.py**: Reflex configuration with app name "app" and plugins (Sitemap, TailwindV4)
 - **app/app.py**: Main application entry point with Reflex components
 - **app/components/**: Application UI components
 - **app/models/**: Application models
 - **app/pages/**: Application pages
 - **app/states/**: Application states
 - **app/templates/**: Application templates
 - **assets/**: Static assets like favicon
 - **data/**: Data files (e.g., Yahoo Finance)
 - **tests/**: Unit and integration tests

 ### Application Components
 - **State Management**: Uses `rx.State` class for reactive state
 - **UI Components**: Built with Reflex components (`rx.container`, `rx.vstack`, etc.)
 - **Routing**: Single page app with index route
 - **Styling**: TailwindV4 plugin enabled for CSS styling

 ### Dependencies
 - **Core**: reflex
 - **Financial/Data**: bt, cvxpy, numpy, pandas, yfinance, riskfolio-lib
 - **Visualization**: matplotlib
 - **Persistence**: duckdb, json, csv, parquet
 - **Development**: black, pylint, flake8, pytest, pytest-cov

 ### UI Kit Components
 - **DO NOT MODIFY** any JSX files in `app/components/uikit/` or `assets/external/app/components/*/uikit/`
 - Only modify the Python Reflex wrappers in `app/components/` to fix integration issues

 ## Suggested Improvements
 - Consolidate development commands under a single `make setup` or `make dev` alias for convenience.
 - Ensure a `pre-commit` hook is configured to run `make format` and `make lint` automatically.
 - Add a quick start section for new contributors including venv activation.
 - Include a troubleshooting section for common errors (e.g., port conflicts, missing dependencies).
 - Document how to run and update UI Kit and external assets when upstream changes.
