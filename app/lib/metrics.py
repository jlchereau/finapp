"""
Metrics display utility functions for showing financial data in tables
"""

import reflex as rx


def show_metric_as_badge(
    value: float | None = None,
    low: float | None = None,
    high: float | None = None,
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
        return rx.badge(value, color_scheme=color_low, variant="solid")
    if high is not None and value >= high:
        return rx.badge(value, color_scheme=color_high, variant="solid")
    return rx.text(value)


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
