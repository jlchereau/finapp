"""Test page"""

import reflex as rx
from ..templates.template import template


class TestState(rx.State):  # pylint: disable=inherit-non-class
    """The test page state."""


# pylint: disable=not-callable
@rx.page(route="/test")  # pyright: ignore[reportArgumentType]
@template
def page():
    """The test page."""
    return rx.vstack(
        rx.heading("Test", size="6", margin_bottom="1rem"),
        # rx.hstack(),
        # height="100%",
        spacing="0",
    )
