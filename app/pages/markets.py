"""
Markets page
"""

import asyncio
from typing import Dict, Any, List

import reflex as rx
from ..templates.template import template


class MarketState(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""

    # TODO

    @rx.event(background=True)  # pylint: disable=not-callable
    async def run_workflows(self):
        # TODO
        pass


# pylint: disable=not-callable
@rx.page(route="/markets")  # pyright: ignore[reportArgumentType]
@template
def page():
    """The markets page."""
    return rx.vstack(
        rx.heading("Markets", size="6", margin_bottom="1rem"),
        
        # height="100%",
        spacing="0",
    )
