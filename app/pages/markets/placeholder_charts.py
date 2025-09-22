"""Placeholder chart components for future implementation."""

from datetime import datetime

import reflex as rx


class FearAndGreedState(rx.State):  # pylint: disable=inherit-non-class
    """State for the Fear and Greed Index placeholder component."""

    content: rx.Field[str] = rx.field("Fear and Greed index plot")


class ShillerCapeState(rx.State):  # pylint: disable=inherit-non-class
    """State for the Shiller CAPE placeholder component."""

    content: rx.Field[str] = rx.field("Shiller CAPE plot")


@rx.event
def update_fear_and_greed(
    state: FearAndGreedState, base_date: datetime  # pylint: disable=unused-argument
):
    """
    Decentralized event handler to update Fear and Greed Index.
    Called from the main page when period changes.
    """
    # Placeholder - no actual update needed yet
    # In the future, this will call a flow to fetch Fear and Greed data
    # yield state.update_chart_data(base_date)
    _ = state, base_date  # Mark as intentionally unused


@rx.event
def update_shiller_cape(
    state: ShillerCapeState, base_date: datetime  # pylint: disable=unused-argument
):
    """
    Decentralized event handler to update Shiller CAPE.
    Called from the main page when period changes.
    """
    # Placeholder - no actual update needed yet
    # In the future, this will call a flow to fetch Shiller CAPE data
    # yield state.update_chart_data(base_date)
    _ = state, base_date  # Mark as intentionally unused


def fear_and_greed_chart() -> rx.Component:
    """Fear and Greed index placeholder component."""
    return rx.box(rx.text(FearAndGreedState.content))


def shiller_cape_chart() -> rx.Component:
    """Shiller CAPE placeholder component."""
    return rx.box(rx.text(ShillerCapeState.content))
