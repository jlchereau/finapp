"""
Test page
"""

from typing import List
import reflex as rx
from ..templates.template import template
from ..components.combobox import combobox, combobox_input, combobox_button, combobox_options, combobox_option


options: List[str] = ["Option 1", "Option 2", "Option 3"]


class State(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""
    option: str = "No selection yet."
    combobox_value: str = ""
    query: str = ""
    
    def set_combobox_value(self, value: str):
        """Set the combobox selected value."""
        self.combobox_value = value

    def set_query(self, query: str):
        """Set the combobox query."""
        self.query = query

    @rx.var
    def filtered_options(self) -> List[str]:
        """Get filtered options based on query."""
        if not self.query:
            return options
        return [opt for opt in options if self.query.lower() in opt.lower()]


@rx.page(route="/test")  # pyright: ignore[reportArgumentType]
@template
def page():
    """The test page."""
    return rx.vstack(
        rx.heading("Test", size="9"),
        rx.heading(State.option),
        rx.select(
            options,
            placeholder="Select an example.",
            on_change=State.set_option,
        ),
        rx.divider(),
        rx.heading("Combobox Test", size="6"),
        rx.text(f"Selected: {State.combobox_value}"),
        rx.text(f"Query: {State.query}"),
        combobox(
            combobox_input(
                placeholder="Search options...",
                on_change=State.set_query,
            ),
            combobox_options(
                rx.foreach(
                    options,
                    lambda option: combobox_option(
                        option,
                        value=option,
                    )
                )
            ),
            value=State.combobox_value,
            on_change=State.set_combobox_value,
        ),
        spacing="5",
        justify="center",
        min_height="85vh",
    )
