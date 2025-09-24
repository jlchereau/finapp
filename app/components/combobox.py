"""
A combobox component For reflex wrapping Tailwind Headless UI Combobox.

Reflex is a Python framework for building web applications.
The Github repository is at https://github.com/reflex-dev/reflex.

Headless UI is a set of completely unstyled, fully accessible UI components,
designed to integrate beautifully with Tailwind CSS.
The Github repository is at https://github.com/tailwindlabs/headlessui.

The React combobox is located at:
    - https://headlessui.com/react/combobox
    - https://github.com/tailwindlabs/headlessui/tree/main/packages/
    %40headlessui-react/src/components/combobox

Enhancements:
- Improve reflex styles
- Open in portal over other content
- Add option navigation with up down arrows
- Implement filtering
- Add check on selected option in the list
- Add placeholder (if HeadlessUi allows it)
- Add literals for options like ComboboxOptions anchor
- Support for multiple selection
- Support for disabled state
- Support for JSON object values (dict[str, Any])
- Support virtual scrolling
- Support more HeadlessUI Combobox parameters and features
"""

import reflex as rx
from reflex.vars import Var
from reflex.event import (
    EventHandler,
    input_event,
    no_args_event_spec,
    passthrough_event_spec,
)
from .headlessui import HeadlessUIComponent

# from .portal import portal


class Combobox(HeadlessUIComponent):
    """
    The main Combobox reflex no-ssrcomponent.
    A reflex react wrapper for the Headless UI Combobox.
    https://headlessui.com/react/combobox#combobox
    """

    tag = "Combobox"

    # Props
    disabled: Var[bool]
    immediate: Var[bool]
    # multiple: Var[bool]
    value: Var[str | None]  # The selected value
    # class_name is inherited

    # Events
    # see https://github.com/reflex-dev/reflex/blob/main/reflex/
    # components/radix/themes/components/text_field.py
    # pyrefly: ignore=[bad-specialization]
    on_change: EventHandler[passthrough_event_spec(str | None)]
    # pyrefly: ignore=[bad-specialization]
    on_close: EventHandler[no_args_event_spec]  # Fired when the combobox is closed

    @classmethod
    def create(cls, *children, **props):
        props.setdefault("disabled", False)
        props.setdefault("immediate", True)
        return super().create(*children, **props)


class ComboboxInput(HeadlessUIComponent):
    """
    The ComboboxInput reflex no-ssr component.
    A reflex react wrapper for the Headless UI ComboboxInput.
    https://headlessui.com/react/combobox#combobox-input
    """

    tag = "ComboboxInput"

    # Props
    display_value: Var[str | None]  # JavaScript function as string for display value

    # Events
    # pyrefly: ignore=[bad-specialization]
    on_change: EventHandler[input_event]

    @classmethod
    def create(cls, *children, **props):
        # props.setdefault("overflow", "auto")
        return super().create(*children, **props)


class ComboboxButton(HeadlessUIComponent):
    """
    The ComboboxButton reflex no-ssr component.
    A reflex react wrapper for the Headless UI ComboboxButton.
    https://headlessui.com/react/combobox#combobox-button
    """

    tag = "ComboboxButton"

    @classmethod
    def create(cls, *children, **props):
        if len(children) == 0:
            # Consider ChevronDownIcon from heroicons
            children = [rx.icon("chevron_down", size=14)]
        return super().create(*children, **props)


class ComboboxOptions(HeadlessUIComponent):
    """
    The ComboboxOptions reflex no-ssr component.
    A reflex react wrapper for the Headless UI ComboboxOptions.
    https://headlessui.com/react/combobox#combobox-options
    """

    tag = "ComboboxOptions"

    # Props
    # key is inherited
    anchor: Var[str]  # Improve with literal of authorized value

    @classmethod
    def create(cls, *children, **props):
        # Consider a CheckIcon
        return super().create(*children, **props)


class ComboboxOption(HeadlessUIComponent):
    """
    The ComboboxOption reflex no-ssr component.
    A reflex react wrapper for the Headless UI ComboboxOption.
    https://headlessui.com/react/combobox#combobox-option
    """

    tag = "ComboboxOption"

    # Props from HeadlessUI ComboboxOption
    # key is inherited
    value: Var[str | None]
    disabled: Var[bool]

    @classmethod
    def create(cls, *children, **props):
        """Create a ComboboxOption component with hover highlighting."""
        props.setdefault("anchor", "top left")
        # Add React event handlers to toggle data-highlighted attribute
        # which allows to use reflex stylesheets (styles from rx.select dropdown)
        custom_attrs = props.setdefault("custom_attrs", {})
        enter_js = "(e) => e.currentTarget.setAttribute('data-highlighted','true')"
        leave_js = "(e) => e.currentTarget.removeAttribute('data-highlighted')"
        # Use camelCase event names for React
        custom_attrs["onMouseEnter"] = Var(_js_expr=enter_js)
        custom_attrs["onMouseLeave"] = Var(_js_expr=leave_js)
        return super().create(*children, **props)


# Individual convenience functions for creating combobox components
combobox = Combobox.create
combobox_input = ComboboxInput.create
combobox_button = ComboboxButton.create
combobox_options = ComboboxOptions.create
combobox_option = ComboboxOption.create


# Wrapper function for the Combobox component using rx.select styles
def combobox_wrapper(
    value,
    on_change,
    options,
):
    """Wrapper function for the Combobox component."""
    return combobox(
        rx.hstack(
            combobox_input(
                display_value=value,
                on_change=on_change,
                class_name="rt-reset rt-TextFieldInput",
            ),
            combobox_button(),
            align="center",
            class_name="rt-TextFieldRoot rt-r-size-2 rt-variant-surface pr-2",
        ),
        # portal(
        combobox_options(
            rx.foreach(
                options,
                lambda option: combobox_option(
                    rx.text(option),
                    value=option,
                    class_name="rt-SelectItem",
                ),
            ),
            class_name="rt-SelectContent rt-r-size-2 rt-variant-solid p-2",
        ),
        # ),
        value=value,
        on_change=on_change,
    )
