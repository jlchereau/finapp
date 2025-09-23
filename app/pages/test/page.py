"""Test page"""

import reflex as rx
from app.templates.template import template


# pylint: disable=inherit-non-class
class TestState(rx.State):
    """The test page state."""


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[bad-argument-type]
@rx.page(route="/test")
@template
def page():
    """The test page."""
    return rx.vstack(
        rx.heading("Test", size="6", margin_bottom="1rem"),
        # rx.hstack(),
        # height="100%",
        spacing="0",
    )
