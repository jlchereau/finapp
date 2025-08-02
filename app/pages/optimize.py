"""
Optimize page
"""

import reflex as rx
from ..templates.template import template


class State(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""
    pass


@rx.page(route="/optimize")  # pyright: ignore[reportArgumentType]
@template
def portfolio():
    """The optimize page."""
    return rx.vstack(
        rx.heading("Optimize", size="9"),
        spacing="5",
        justify="center",
        min_height="85vh",
    )
