---
applyTo: "app/{components|pages|states|templates}/**/*.py"
---

# UI Design and Instructions

UI is split across components and pages, a component being a reusable piece of UI that can be used in different pages, like the [navigation bar](../../app/components/navbar.py) or the [theme toggler](../../app/components/theme_toggle.py).

## General Guidelines

Components and pages are made with the reflex framework, which is a Python framework for building web applications. It uses a declarative style similar to React, allowing you to define UI components and their behavior in Python. The relevant documentation is available at [Reflex documentation](https://reflex.dev/docs/).

## Components

The application currently has the following components:
- [Navigation Bar](../../app/components/navbar.py): The navigation bar at the top of the page, allowing users to navigate between different sections of the application.
- [Theme Toggler](../../app/components/theme_toggle.py): A component that allows users to switch between light and dark themes.

Note that the Reflex framework lacks a combobox component, so we aim to develop one based on a [wrapper](https://reflex.dev/docs/wrapping-react/overview/) around [@headlessui/react](https://headlessui.com/react/combobox) and Tailwind CSS. This is work in progress.

## Page Layout (templates)

Currently all pages use the same layout, which includes a navigation bar at the top.The layout is defined in [app/templates/template.py](../../app/templates/template.py).

## Styling

At this early stage of development, the application uses Reflex's default styling based on Tailwind CSS. Custom styles are low priority.

## Pages

The application currently has the following pages:
- [Root Home](../../app/pages/index.py): The main page of the application, which displays a [welcome message and user instructions](../../assets/index.md).
- [Markets](../../app/pages/markets.py): The page displays the latest market data and trends using a workflow that fetches and aggregates data from various sources.
- [Portfolio](../../app/pages/portfolio.py): The page displays the content of the IBKR portfolio using a workflow that fetches data using the IBKR API.
- [Screen](../../app/pages/screen.py): The page triggers a workflow that fetches data from various sources to build a parquet file which is then filtered using DuckDB.
- [Compare](../../app/pages/compare.py): The page triggers a workflow that fetches data from various sources to compare different assets.
- [Optimize](../../app/pages/optimize.py): The page allows the creation of asset lists which are used to build optimized portfolios using various algorithms available in RiskFolio-Lib.
- [Backtest](../../app/pages/backtest.py): The page allows the user to backtest a portfolio against historical data.
- [Cache](../../app/pages/cache.py): The page allows the user to manage the cache and view logs.

## State Management

At this early stage of development, the application uses Reflex's built-in state management according to documented best practices.