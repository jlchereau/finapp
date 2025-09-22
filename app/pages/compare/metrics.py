"""Combined metrics component with all financial metrics tables."""

from datetime import datetime
from typing import List
from random import uniform

import reflex as rx

from app.lib.periods import fix_datetime
from app.lib.metrics import (
    show_metric_as_badge,
    integer_formatter,
    large_currency_formatter,
)
from app.lib.exceptions import PageOutputException


class MetricsState(rx.State):  # pylint: disable=inherit-non-class
    """State for the combined metrics component."""

    loading: rx.Field[bool] = rx.field(False)
    # Future: metrics_data for when the flow is implemented

    @rx.event
    async def update_metrics_data(self, tickers: List[str], base_date: datetime):
        """Update the metrics data using workflow (future implementation)."""
        if not tickers:
            return

        self.loading = True

        try:
            # TODO: Future implementation when metrics flow is ready
            # result = await fetch_metrics_data(tickers=tickers, base_date=base_date)
            # self.metrics_data = result.get("data")
            pass

        except Exception as e:
            # Metrics generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="metrics data",
                message=f"Failed to generate metrics data: {e}",
                user_message=(
                    "Failed to generate metrics data. Please try refreshing the data."
                ),
                context={"tickers": tickers, "error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_metrics(state: MetricsState, tickers: List[str], base_date: datetime):
    """
    Decentralized event handler to update metrics.
    Called from the main page when tickers or period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_metrics_data(tickers, base_date)


def asset_valuation_table() -> rx.Component:
    """Asset valuation section."""
    return rx.table.root(
        rx.table.header(
            # Metrics column + one column per ticker
            rx.table.row(
                rx.table.column_header_cell("Metrics"),
                rx.table.column_header_cell("AMZN"),
                rx.table.column_header_cell("GOOG"),
            ),
        ),
        rx.table.body(
            # One row per metric
            rx.table.row(
                rx.table.row_header_cell("Quote"),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
            ),
            rx.table.row(
                rx.table.row_header_cell("DCF"),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
            ),
        ),
        width="100%",
    )


def graham_indicators_table() -> rx.Component:
    """
    Graham indicators section.

    Metrics can be found in Chapter 14 pages 367 and after of
    The Intelligent Investor (4th revised edition) by Benjamin Graham.
    """
    return rx.table.root(
        # Metrics column + one column per ticker
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Metrics"),
                rx.table.column_header_cell("AMZN"),
                rx.table.column_header_cell("GOOG"),
            ),
        ),
        # One row per metric
        rx.table.body(
            rx.table.row(
                rx.table.row_header_cell("Market Capitalization"),
                # According to Graham, larger companies are generally more stable
                # Market cap should be at least $2 billion
                # and preferably over $5 billion
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(1000000000, 150000000000),
                        2000000000.0,
                        5000000000.0,
                        large_currency_formatter,
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(1000000000, 150000000000),
                        2000000000.0,
                        5000000000.0,
                        large_currency_formatter,
                    )
                ),
            ),
            rx.table.row(
                rx.table.row_header_cell("Annual Revenue"),
                # According to Graham, larger companies are generally more stable
                # Annual revenue should be at least $1 billion
                # and preferably over $3 billion
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(1000000000, 150000000000),
                        1000000000.0,
                        3000000000.0,
                        large_currency_formatter,
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(1000000000, 150000000000),
                        2000000000.0,
                        5000000000.0,
                        large_currency_formatter,
                    )
                ),
            ),
            rx.table.row(
                rx.table.row_header_cell("Current Ratio"),
                # According to Graham, a current ratio of 2 or higher is considered
                # healthy for a strong financial condition
                # (current assets >= 2 * current liabilities)
                rx.table.cell(show_metric_as_badge(uniform(0, 20), 1.5, 2)),
                rx.table.cell(show_metric_as_badge(uniform(0, 20), 1.5, 2)),
            ),
            rx.table.row(
                rx.table.row_header_cell("Years Positive Earnings"),
                # According to Graham, positive earnings for the past 10 years are
                # required to demonstrate earnings stability
                rx.table.cell(
                    show_metric_as_badge(uniform(0, 30), 10, 15, integer_formatter)
                ),
                rx.table.cell(
                    show_metric_as_badge(uniform(0, 30), 10, 15, integer_formatter)
                ),
            ),
        ),
        width="100%",
    )


def analyst_ratings_table() -> rx.Component:
    """Analyst ratings section."""
    return rx.table.root(
        # Metrics column + one column per ticker
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Metrics"),
                rx.table.column_header_cell("AMZN"),
                rx.table.column_header_cell("GOOG"),
            ),
        ),
        # One row per metric
        rx.table.body(
            rx.table.row(
                rx.table.row_header_cell("Tipranks"),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
            ),
            rx.table.row(
                rx.table.row_header_cell("Zacks"),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
            ),
        ),
        width="100%",
    )


def metrics() -> rx.Component:
    """Combined metrics component showing various financial metrics."""
    return rx.cond(
        MetricsState.loading,
        rx.center(rx.spinner(), height="400px"),
        rx.box(
            rx.section(
                rx.heading("Valuation", size="5", margin_bottom="1rem"),
                rx.divider(),
                asset_valuation_table(),
                padding_left="1rem",
                padding_right="1rem",
            ),
            rx.section(
                rx.heading("Graham Indicators", size="5", margin_bottom="1rem"),
                rx.divider(),
                graham_indicators_table(),
                padding_left="1rem",
                padding_right="1rem",
            ),
            rx.section(
                rx.heading("Analyst Ratings", size="5", margin_bottom="1rem"),
                rx.divider(),
                analyst_ratings_table(),
                padding_left="1rem",
                padding_right="1rem",
            ),
            width="100%",
        ),
    )
