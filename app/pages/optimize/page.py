"""Optimize page"""

from typing import List

import reflex as rx

from app.templates.template import template
from app.pages.optimize.card1 import card1, update_card1
from app.pages.optimize.card2 import card2, update_card2
from app.lib.periods import (
    get_period_default,
    get_period_options,
)


def compute(period_option: str) -> str:
    """Compute the base date."""
    return period_option  # Placeholder for actual computation logic


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
        base_date = compute(option)
        return [
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_card1(base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_card2(base_date=base_date),
            rx.toast.info(f"Changed time period to {option}")
        ]


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[not-callable,bad-argument-type]
@rx.page(route="/optimize")
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
