"""Card 1 module."""

import reflex as rx


class Card1State(rx.State):  # pylint: disable=inherit-non-class
    """
    The card1 state.
    This could be a chart state.
    """

    base_date: rx.Field[str] = rx.field("")

    @rx.event
    def set_base_date(self, base_date: str):
        self.base_date = base_date


@rx.event
def update_card1(state: Card1State, base_date: str):
    """
    A decentralized event handler to be called from the page state
    when the period_option, and therefore the base_date changes.
    See https://reflex.dev/docs/events/decentralized-event-handlers/
    """
    yield state.set_base_date(f"Period: {base_date}")


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
        width="100%"
    )
