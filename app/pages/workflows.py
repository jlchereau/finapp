"""Workflows page"""

import reflex as rx
from ..templates.template import template


class State(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""

    pass


@rx.page(route="/workflows")  # pyright: ignore[reportArgumentType]
@template
def page():
    """The workflows page."""
    return rx.vstack(
        rx.heading("Workflows", size="9"),
        spacing="5",
        justify="center",
        min_height="85vh",
    )
