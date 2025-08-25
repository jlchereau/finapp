"""
Metrics display utility functions for showing financial data in tables
"""

import reflex as rx


def show_metrics_as_badge(
    value: float,
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
    if low is not None and value <= low:
        return rx.badge(value, color_scheme=color_low, variant="solid")
    if high is not None and value >= high:
        return rx.badge(value, color_scheme=color_high, variant="solid")
    return rx.text(value)
