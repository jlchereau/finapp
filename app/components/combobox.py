"""A combobox component."""

from typing import Any, Callable
import reflex as rx
from .headlessui import HeadlessUIComponent


class Combobox(HeadlessUIComponent):
    """The main combobox component."""

    tag = "Combobox"

    # The selected value(s)
    value: rx.Var[Any]

    # Callback when selection changes
    on_change: rx.EventHandler[lambda value: [value]]

    # Whether to allow multiple selections
    multiple: rx.Var[bool]

    # Whether the combobox is disabled
    disabled: rx.Var[bool]

    # Form input name
    name: rx.Var[str]

    # Open options on input focus
    immediate: rx.Var[bool]

    # Enable virtual scrolling for large lists
    virtual: rx.Var[bool]


class ComboboxInput(HeadlessUIComponent):
    """The input field for the combobox."""

    tag = "ComboboxInput"

    # Function to get display value from selected item
    display_value: rx.Var[Callable[[Any], str]]

    # Callback when input value changes
    on_change: rx.EventHandler[lambda event: [event.target.value]]

    # Input placeholder text
    placeholder: rx.Var[str]

    # Whether the input is disabled
    disabled: rx.Var[bool]

    # Input autocomplete behavior
    auto_complete: rx.Var[str]


class ComboboxButton(HeadlessUIComponent):
    """The button that opens the combobox options."""

    tag = "ComboboxButton"

    # Whether the button is disabled
    disabled: rx.Var[bool]


class ComboboxOptions(HeadlessUIComponent):
    """The popover that contains the list of options."""

    tag = "ComboboxOptions"

    # Anchor position for the options popover
    anchor: rx.Var[str]

    # Whether to use a portal for rendering
    portal: rx.Var[bool]

    # Whether to use modal behavior
    modal: rx.Var[bool]

    # Whether to transition when opening/closing
    transition: rx.Var[bool]


class ComboboxOption(HeadlessUIComponent):
    """An option in the list of combobox options."""

    tag = "ComboboxOption"

    # The value of this option
    value: rx.Var[Any]

    # Whether this option is disabled
    disabled: rx.Var[bool]


# Convenience functions for creating components
combobox = Combobox.create
combobox_input = ComboboxInput.create
combobox_button = ComboboxButton.create
combobox_options = ComboboxOptions.create
combobox_option = ComboboxOption.create
