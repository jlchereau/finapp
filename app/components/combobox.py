"""Reflex wrapper for Tailwind Catalyst UI Kit Combobox component."""

import reflex as rx
from typing import Any, Callable, Dict, List, Optional, Union

# Define the library path using shared assets
# As in https://reflex.dev/docs/wrapping-react/local-packages#local-components
combobox_path = "$/public/" + rx.asset("uikit/combobox.jsx", shared=True)

class Combobox(rx.components.component.NoSSRComponent):
    """A combobox component with search functionality."""
    
    library = combobox_path
    tag = "Combobox"
    lib_dependencies = ["@headlessui/react"]

    options: rx.Var[List[Any]]
    display_value: rx.Var[Any]
    filter: rx.Var[Optional[str]]
    anchor: rx.Var[str]
    placeholder: rx.Var[Optional[str]]
    auto_focus: rx.Var[bool]
    aria_label: rx.Var[Optional[str]]
    as_: rx.Var[str]
    
    def _get_custom_components(self) -> set[str]:
        return {"Combobox"}


class ComboboxOption(rx.components.component.NoSSRComponent):
    """A single option within the combobox."""
    
    library = combobox_path
    tag = "ComboboxOption"
    lib_dependencies = ["@headlessui/react"]
    
    value: rx.Var[Any]
    disabled: rx.Var[bool]
    
    def _get_custom_components(self) -> set[str]:
        return {"ComboboxOption"}


class ComboboxLabel(rx.components.component.NoSSRComponent):
    """A label for content within a combobox option."""
    
    library = combobox_path
    tag = "ComboboxLabel"
    lib_dependencies = ["@headlessui/react"]
    
    def _get_custom_components(self) -> set[str]:
        return {"ComboboxLabel"}


class ComboboxDescription(rx.components.component.NoSSRComponent):
    """A description for content within a combobox option."""
    
    library = combobox_path 
    tag = "ComboboxDescription"
    lib_dependencies = ["@headlessui/react"]
    
    def _get_custom_components(self) -> set[str]:
        return {"ComboboxDescription"}


# Convenience functions
def combobox(
    *children,
    options: Union[rx.Var[List[Any]], List[Any]] = [],
    display_value: Union[rx.Var[str], str] = "",
    filter: Optional[Union[rx.Var[str], str]] = None,
    anchor: Union[rx.Var[str], str] = "bottom",
    placeholder: Optional[Union[rx.Var[str], str]] = None,
    auto_focus: Union[rx.Var[bool], bool] = False,
    aria_label: Optional[Union[rx.Var[str], str]] = None,
    as_: Union[rx.Var[str], str] = "div",
    class_name: Optional[Union[rx.Var[str], str]] = None,
    **props
) -> Combobox:
    """Create a combobox component.
    
    Args:
        *children: Child components to render for each option
        options: List of options to display
        display_value: Display value for the component
        filter: Optional custom filter function
        anchor: Dropdown anchor position ('top', 'bottom', etc.)
        placeholder: Placeholder text for input
        auto_focus: Whether to auto-focus the input
        aria_label: Accessibility label
        class_name: CSS classes to apply
        **props: Additional props to pass to the component
    
    Returns:
        Combobox component
    """
    if not children:
        children = (
            rx.Var.create(
                "(option) => React.createElement(ComboboxOption, { value: option }, option)"
            ),
        )
    return Combobox.create(
        *children,
        options=options,
        display_value=display_value,
        filter=filter,
        anchor=anchor,
        placeholder=placeholder,
        auto_focus=auto_focus,
        aria_label=aria_label,
        as_=as_,
        class_name=class_name,
        **props
    )


def combobox_option(
    *children,
    value: Union[rx.Var[Any], Any],
    disabled: Union[rx.Var[bool], bool] = False,
    class_name: Optional[Union[rx.Var[str], str]] = None,
    **props
) -> ComboboxOption:
    """Create a combobox option.
    
    Args:
        *children: Child components to render inside the option
        value: The value for this option
        disabled: Whether the option is disabled
        class_name: CSS classes to apply
        **props: Additional props to pass to the component
    
    Returns:
        ComboboxOption component
    """
    return ComboboxOption.create(
        *children,
        value=value,
        disabled=disabled,
        class_name=class_name,
        **props
    )


def combobox_label(
    class_name: Optional[Union[rx.Var[str], str]] = None,
    **props
) -> ComboboxLabel:
    """Create a combobox label.
    
    Args:
        class_name: CSS classes to apply
        **props: Additional props to pass to the component
    
    Returns:
        ComboboxLabel component
    """
    return ComboboxLabel.create(
        class_name=class_name,
        **props
    )


def combobox_description(
    class_name: Optional[Union[rx.Var[str], str]] = None,
    **props
) -> ComboboxDescription:
    """Create a combobox description.
    
    Args:
        class_name: CSS classes to apply
        **props: Additional props to pass to the component
    
    Returns:
        ComboboxDescription component
    """
    return ComboboxDescription.create(
        class_name=class_name,
        **props
    )