"""Optimize page"""

from datetime import datetime, timedelta
import re
from typing import List

import reflex as rx

from app.templates.template import template
from app.pages.optimize.card1 import card1, update_card1
from app.pages.optimize.card2 import card2, update_card2
from app.lib.periods import (
    get_period_default,
    get_period_options,
)


def compute(period_option: str) -> datetime:
    """
    Compute the base date.
    The following is an approximation using 1M=30d and 1Q=90d.
    A more accurate result can be achieved with dateutil.relativedelta.
    """
    match = re.match(r"(\d+)([WMQY])", period_option)
    if match:
        number, unit = int(match.group(1)), match.group(2)
        if unit == "W":
            base_date = datetime.now() - timedelta(days=number * 7)
        elif unit == "M":
            base_date = datetime.now() - timedelta(days=number * 30)
        elif unit == "Q":
            base_date = datetime.now() - timedelta(days=number * 90)
        else:  # elif unit == "Y":
            base_date = datetime.now() - timedelta(days=number * 365)
        return base_date
    else:
        # "YTD" and "MAX" will raise an error, but that's fine for this prototype.
        raise ValueError(f"Invalid period option: {period_option}")


class PageState(rx.State):  # pylint: disable=inherit-non-class
    """
    The page state
    In a separate file to avoid circular imports.
    """

    period_option: rx.Field[str] = rx.field(default_factory=get_period_default)
    period_options: rx.Field[List[str]] = rx.field(default_factory=get_period_options)
    text: rx.Field[str] = rx.field("Optimize Page")

    @rx.event
    def set_period_option(self, option: str):
        """Set period option and update all cards."""
        self.period_option = option
        for w in self.run_workflows():
            yield w

    @rx.event
    def run_workflows(self):
        """Load initial chart data."""
        base_date = compute(self.period_option)
        return [
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_card1(base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_card2(base_date=base_date),
        ]


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[not-callable,bad-argument-type]
@rx.page(route="/optimize", on_load=PageState.run_workflows)
@template
def page():
    """The optimize page."""
    return rx.vstack(
        rx.heading(PageState.text.to_string(), size="6", margin_bottom="1rem"),
        rx.hstack(
            rx.select(
                PageState.period_options,
                value=PageState.period_option,
                on_change=PageState.set_period_option,
            ),
            card1(),
            card2(),
        ),
        # height="100%",
        spacing="0",
    )
