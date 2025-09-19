"""Card 1 module."""

import reflex as rx

from app.pages.optimize.state import SharedState


class Card1State(SharedState, rx.State):  # pylint: disable=inherit-non-class
    """The card1 state."""

    text: rx.Field[str] = rx.field("Card1")

    @rx.var
    def text_with_period_option(self) -> str:
        # This will be recomputed with changes
        return self.text + " " + self.period_option


def card1() -> rx.Component:
    return rx.card(Card1State.text_with_period_option, padding="1rem", width="100%")
