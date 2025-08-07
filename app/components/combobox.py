"""
A combobox component For reflex wrapping Tailwind Headless UI Combobox.

Reflex is a Python framework for building web applications.
The Github repository is at https://github.com/reflex-dev/reflex.

Headless UI is a set of completely unstyled, fully accessible UI components, designed to integrate beautifully with Tailwind CSS.
The Github repository is at https://github.com/tailwindlabs/headlessui.

The React combobox is located at:
    - https://headlessui.com/react/combobox
    - https://github.com/tailwindlabs/headlessui/tree/main/packages/%40headlessui-react/src/components/combobox
"""

from typing import Any, Optional, Union
import reflex as rx
from reflex.vars import Var
from reflex.event import no_args_event_spec
from reflex.constants import EventTriggers
from .headlessui import HeadlessUIComponent


class Combobox(HeadlessUIComponent):
    """The main combobox component."""
    tag = "Combobox"
    
    # Props from HeadlessUI Combobox
    value: Var[Any]
    multiple: Var[bool]
    disabled: Var[bool]
    name: Var[str]
    immediate: Var[bool]
    as_: Var[str]  # HTML element to render as (e.g., 'div', 'span')
    
    @classmethod
    def get_event_triggers(cls) -> dict:
        return {
            "on_change": lambda value: [value],
        }


class ComboboxInput(HeadlessUIComponent):
    """The combobox input element."""
    tag = "ComboboxInput"
    
    # Props from HeadlessUI ComboboxInput
    auto_focus: Var[bool]
    placeholder: Var[str]
    display_value: Var[str]  # JavaScript function as string for display value
    value: Var[str]  # Current input value
    
    @classmethod
    def get_event_triggers(cls) -> dict:
        return {
            "on_change": lambda e0: [e0.target.value],
            "on_blur": no_args_event_spec,
            "on_focus": no_args_event_spec,
        }


class ComboboxButton(HeadlessUIComponent):
    """The combobox toggle button."""
    tag = "ComboboxButton"
    
    # Props from HeadlessUI ComboboxButton
    auto_focus: Var[bool]


class ComboboxOptions(HeadlessUIComponent):
    """The combobox options container."""
    tag = "ComboboxOptions"
    
    # Props from HeadlessUI ComboboxOptions
    anchor: Var[str]
    static: Var[bool]
    portal: Var[bool]
    unmount: Var[bool]


class ComboboxOption(HeadlessUIComponent):
    """Individual combobox option."""
    tag = "ComboboxOption"
    
    # Props from HeadlessUI ComboboxOption
    value: Var[Any]
    disabled: Var[bool]
    
    @classmethod
    def get_event_triggers(cls) -> dict:
        return {
            "on_click": no_args_event_spec,
        }


# Individual convenience functions for creating combobox components
combobox_raw = Combobox.create
combobox_input = ComboboxInput.create
combobox_button = ComboboxButton.create
combobox_options = ComboboxOptions.create
combobox_option = ComboboxOption.create


def combobox(
    options: Union[list, Var],
    value=None,
    on_change=None,
    on_input_change=None,
    input_value=None,
    placeholder: str = "Search...",
    display_field: str = "name",
    value_field: Optional[str] = None,
    show_button: bool = True,
    button_icon: str = "chevron_down",
    input_class: str = "w-full px-3 py-2 pr-10 text-sm bg-white border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 hover:bg-gray-50",
    button_class: str = "absolute inset-y-0 right-0 flex items-center pr-3",
    options_class: str = "rt-SelectContent bg-white border border-gray-200 rounded-md shadow-lg z-50 max-h-60 overflow-auto py-1",
    option_class: str = "rt-SelectItem relative cursor-default select-none px-3 py-2 text-sm text-gray-900 hover:bg-blue-50 hover:text-blue-900 data-[highlighted]:bg-blue-50 data-[highlighted]:text-blue-900",
    container_class: str = "rt-Select relative w-full",
    **props
):
    """Create a complete combobox with input, button, and options.
    
    Args:
        options: List of options (can be strings, dicts, or rx.Var)
        value: Current selected value (passed to Combobox component)
        on_change: Function to call when option is selected
        on_input_change: Function to call when input text changes
        input_value: Current input display value (what shows in the text field)
        placeholder: Placeholder text for input
        display_field: Field name to display for dict options (default: "name")
        value_field: Field to use as value for dict options (default: same as display_field)
        show_button: Whether to show dropdown button (default: True)
        button_icon: Icon for dropdown button (default: "chevrons_up_down")
        input_class: CSS classes for input element
        button_class: CSS classes for button element
        options_class: CSS classes for options container
        option_class: CSS classes for individual options
        container_class: CSS classes for combobox container
        **props: Additional props passed to main Combobox component
    
    Returns:
        A complete Combobox component with all children
    """
    import reflex as rx
    
    # If value_field is not specified, use display_field
    if value_field is None:
        value_field = display_field
    
    # Build the input
    input_component = combobox_input(
        placeholder=placeholder,
        value=input_value,
        on_change=on_input_change,
        class_name=input_class
    )
    
    # Build the button (optional)
    button_component = None
    if show_button:
        button_component = combobox_button(
            rx.icon(tag=button_icon, class_name="h-4 w-4 text-gray-400"),
            class_name=button_class
        )
    
    # Build the options
    option_components = []
    
    if isinstance(options, list):
        # Handle static list of options
        for option in options:
            if isinstance(option, str):
                # String options
                option_components.append(
                    combobox_option(
                        option,
                        value=option,
                        class_name=option_class
                    )
                )
            else:
                # Dict/object options
                display_text = option[display_field] if isinstance(option, dict) else getattr(option, display_field, str(option))
                option_value = option if value_field is None else (option[value_field] if isinstance(option, dict) else getattr(option, value_field, option))
                option_components.append(
                    combobox_option(
                        display_text,
                        value=option_value,
                        class_name=option_class
                    )
                )
    else:
        # Handle rx.Var or dynamic options using rx.foreach
        # For rx.Var, we need to be careful about how we access fields
        # to ensure the entire object is passed as the value
        option_components.append(
            rx.foreach(
                options,
                lambda option: combobox_option(
                    option[display_field],
                    value=option,  # Always pass the entire option object
                    class_name=option_class
                )
            )
        )
    
    options_component = combobox_options(
        *option_components,
        class_name=options_class
    )
    
    # Build the complete combobox (HeadlessUI needs input and button as direct children)
    children = [input_component]
    if button_component:
        children.append(button_component)
    children.append(options_component)
    
    return combobox_raw(
        *children,
        value=value,
        on_change=on_change,
        as_="div",
        class_name=container_class,
        **props
    )
