"""Optimize page"""

import reflex as rx
from app.templates.template import template


class OptimizeState(rx.State):  # pylint: disable=inherit-non-class
    """The optimize page state."""


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[not-callable,bad-argument-type]
@rx.page(route="/optimize")
@template
def page():
    """The optimize page."""
    return rx.vstack(
        rx.heading("Optimize", size="6", margin_bottom="1rem"),
        # rx.hstack(),
        # height="100%",
        spacing="0",
    )
