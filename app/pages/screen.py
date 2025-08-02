""" Screen page """

import reflex as rx
from ..templates.template import template


class State(rx.State):
    """The app state."""
    pass


@rx.page(route="/screen")
@template
def portfolio():
    """The screen page."""
    return rx.vstack(
        rx.heading("Screen", size="9"),
        spacing="5",
        justify="center",
        min_height="85vh",
    )
