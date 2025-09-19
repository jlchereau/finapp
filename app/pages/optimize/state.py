"""Shared state module."""

from typing import List

import reflex as rx

from app.lib.periods import (
    get_period_default,
    get_period_options,
)


class SharedState(rx.State, mixin=True):  # pylint: disable=inherit-non-class
    """
    The shared state (mixin).
    More info at https://reflex.dev/docs/state-structure/mixins/
    """

    period_option: rx.Field[str] = rx.field(default_factory=get_period_default)
    period_options: rx.Field[List[str]] = rx.field(default_factory=get_period_options)

    # TODO: consider registering a list of components to update
    #       when changing the period_option

    def update_all_charts(self):
        """Update all charts."""
        yield rx.toast.info(f"Changed time period to {self.period_option}")

    def set_period_option(self, option: str):
        """Set period option and update all charts."""
        self.period_option = option
        yield from self.update_all_charts()
