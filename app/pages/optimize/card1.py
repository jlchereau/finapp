"""Card 1 module."""

import asyncio
from datetime import datetime

import reflex as rx

from app.lib.periods import fix_datetime


class Card1State(rx.State):  # pylint: disable=inherit-non-class
    """
    The card1 state.
    This could be a chart state.
    """

    base_date: rx.Field[datetime] = rx.field(default_factory=datetime.now)

    @rx.event
    def set_base_date(self, base_date: datetime):
        self.base_date = base_date


@rx.event
async def update_card1(state: Card1State, base_date: datetime):
    """
    A decentralized event handler to be called from the page state
    when the period_option, and therefore the base_date changes.
    See https://reflex.dev/docs/events/decentralized-event-handlers/
    It is async because flows are async in our app.
    """
    base_date = fix_datetime(base_date)
    print(f"Card1 date: {base_date}")
    state.set_base_date(base_date)
    await asyncio.sleep(0.5)


def card1() -> rx.Component:
    """
    The card1 component.
    This could be a chart component
    """
    return rx.card(
        rx.vstack(
            rx.heading("Card 1", size="4"),
            rx.text(Card1State.base_date),
        ),
        padding="1rem",
        width="100%",
    )
