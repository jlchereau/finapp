""" Portfolio page """

import reflex as rx
from ..templates.template import template


class State(rx.State):
    """The app state."""
    pass


@rx.page(route="/portfolio")
@template
def portfolio():
    """The portfolio page."""
    return rx.vstack(
        rx.heading("Portfolio", size="9"),
        spacing="5",
        justify="center",
        min_height="85vh",
    )
