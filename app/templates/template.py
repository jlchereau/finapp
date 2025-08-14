"""A page template sharing the navbar"""

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
        rx.box(
            page(),
            padding="1rem",
            width="100%",
        ),
        # placeholder for footer
        spacing="0",
        width="100%",
    )
