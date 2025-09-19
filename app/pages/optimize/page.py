"""Optimize page"""

import reflex as rx
from app.templates.template import template
from app.pages.optimize.state import SharedState
from app.pages.optimize.card1 import card1
from app.pages.optimize.card2 import card2


class PageState(SharedState, rx.State):  # pylint: disable=inherit-non-class
    """The optimize page state using the shared state mixin."""

    text: rx.Field[str] = rx.field("Optimize Page")


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
