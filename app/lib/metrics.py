"""
Metrics display utility functions for showing financial data in tables
"""

from typing import Callable
import reflex as rx


def integer_formatter(value: float) -> str:
    """Format the metric value for display."""
    return f"{value:.0f}"


def percentage_formatter(value: float) -> str:
    """Format the metric value for display."""
    return f"{value:.2f}%"


def currency_formatter(value: float) -> str:
    """Format the metric value for display."""
    return f"{value:.2f}"


def large_currency_formatter(value: float) -> str:
    """Format the metric value for display."""
    if abs(value) >= 1e12:
        return f"{value/1e12:.2f} T"
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f} B"
    if abs(value) >= 1e6:
        return f"{value/1e6:.2f} M"
    if abs(value) >= 1e3:
        return f"{value/1e3:.2f} K"
    return f"{value:.1f}"


def show_metric_as_badge(
    value: float | None = None,
    low: float | None = None,
    high: float | None = None,
    formatter: Callable = currency_formatter,
    higher_better: bool = True,
) -> rx.Component:
    """Show a metric value as a badge."""
    if higher_better:
        color_high = "green"
        color_low = "tomato"
    else:
        color_high = "tomato"
        color_low = "green"
    if value is None:
        # Some values might not be available, especially for ETFs and other funds
        return rx.badge("n/a", color_scheme="gray", variant="soft")
    if low is not None and value <= low:
        return rx.badge(formatter(value), color_scheme=color_low, variant="solid")
    if high is not None and value >= high:
        return rx.badge(formatter(value), color_scheme=color_high, variant="solid")
    return rx.text(formatter(value))


def show_metric_as_gauge(
    value: float | None = None,
    # min: float | None = None,
    # max: float | None = None,
    # label: str | None = None
) -> rx.Component:
    """Show a metric value as a gauge."""
    if value is None:
        # Some values might not be available, especially for ETFs and other funds
        return rx.badge("n/a", color_scheme="gray", variant="soft")
    return rx.text(
        "TODO use @app/components/gaugechart.py once implemented as required"
    )
