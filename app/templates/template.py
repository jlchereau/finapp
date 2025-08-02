""" A page template sharing the navbar """

from typing import Callable
import reflex as rx
from ..components.navbar import navbar


def template(
    page: Callable[[], rx.Component],
) -> rx.Component:
    """
    Template for pages.
    """
    return rx.vstack(
        navbar(),
        rx.hstack(
            rx.container(page()),
        ),
        width="100%",
    )
