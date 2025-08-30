"""Screen page"""

import reflex as rx
from app.templates.template import template
from app.components.combobox import combobox_wrapper as combobox
from app.components.gaugechart import gaugechart
from app.components.querybuilder import querybuilder, Field


class State(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""

    text: str = ""
    selected: str = ""
    searched: str | None = None

    options = [
        "Durward Reynolds",
        "Kenton Towne",
        "Therese Wunsch",
        "Benedict Kessler",
        "Katelyn Rohan",
    ]

    # pyrefly: ignore[not-callable]
    @rx.event
    def set_text(self, value: str) -> None:
        """Set the text."""
        self.text = value

    # pyrefly: ignore[not-callable]
    @rx.event
    def set_selected(self, value: str) -> None:
        """Set selected value."""
        self.selected = value

    # pyrefly: ignore[not-callable]
    @rx.event
    def set_searched(self, value: str | None) -> None:
        """Set searched value."""
        self.searched = value


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[not-callable]
@rx.page(route="/screen")
@template
def page():
    """The screen page."""
    return rx.vstack(
        rx.heading("Screen", size="6", margin_bottom="1rem"),
        # rx.text(f"Selected: {State.selected}"),
        # rx.text(f"Selected person: {State.selected_person}"),
        # Component comparison
        rx.vstack(
            rx.heading("Component Comparison", size="5", margin_bottom="4"),
            # Reflex Input for comparison
            rx.box(
                rx.text("Reflex Input:", font_weight="bold", margin_bottom="2"),
                rx.input(
                    placeholder="Search people...",
                    value=State.text,
                    on_change=State.set_text,
                    size="2",
                ),
                rx.text(State.text, class_name="text-blue-500", size="1"),
                class_name="mb-6 max-w-xs",
            ),
            # Reflex Select for comparison
            rx.box(
                rx.text("Reflex Select:", font_weight="bold", margin_bottom="2"),
                rx.select(
                    State.options,
                    placeholder="Select a person...",
                    on_change=State.set_selected,
                    value=State.selected,
                    size="2",
                ),
                rx.text(State.selected, class_name="text-blue-500", size="1"),
                class_name="mb-6 max-w-xs",
            ),
            # Our styled Combobox
            rx.box(
                rx.text("HeadlessUI Combobox:", font_weight="bold", margin_bottom="2"),
                combobox(
                    options=State.options,
                    value=State.searched,
                    on_change=State.set_searched,
                ),
                rx.text(State.searched, class_name="text-blue-500", size="1"),
                class_name="mb-6 max-w-xs",
            ),
            rx.box(
                rx.text("Sample Gauge Chart"),
                gaugechart(percent=0.4),
                class_name="mb-6 max-w-xs",
            ),
            rx.box(
                rx.text("Sample Query Builder"),
                querybuilder(
                    fields=[
                        Field(name="field1", label="Field 1"),
                        Field(name="field2", label="Field 2"),
                    ]
                ),
                class_name="mb-6 max-w-xs",
            ),
            class_name="mt-8",
        ),
        spacing="5",
        justify="center",
        min_height="85vh",
    )
