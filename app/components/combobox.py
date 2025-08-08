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
- Add highlighting when hovering over options (add data-highlighted attribute)
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
from reflex.vars import Var, VarData
from reflex.event import (
    EventHandler,
    input_event,
    no_args_event_spec,
    passthrough_event_spec,
)
from .headlessui import HeadlessUIComponent


class Combobox(HeadlessUIComponent):
    """
    The main Combobox reflex no-ssrcomponent.
    A reflex react wrapper for the Headless UI Combobox.
    https://headlessui.com/react/combobox#combobox
    """

    tag = "Combobox"

    # Props
    disabled: Var[bool]
    # multiple: Var[bool]
    value: Var[str]  # The selected value
    # class_name is inherited

    # Events
    on_change: EventHandler[passthrough_event_spec(str)]
    on_close: EventHandler[no_args_event_spec]  # Fired when the combobox is closed

    @classmethod
    def create(cls, *children, **props):
        props.setdefault("disabled", False)
        return super().create(*children, **props)


class ComboboxInput(HeadlessUIComponent):
    """
    The ComboboxInput reflex no-ssr component.
    A reflex react wrapper for the Headless UI ComboboxInput.
    https://headlessui.com/react/combobox#combobox-input
    """

    tag = "ComboboxInput"

    # Props
    display_value: Var[str]  # JavaScript function as string for display value

    # Events
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
            children = [rx.icon("chevron_down")]
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


ON_MOUSE_ENTER_LEAVE_JS = """
// This is sample code that we need to adapt to
// set/unset data-highlighted on mouse enter/leave 
// It only show the implementation of a JavaScript event handler
// This code should toggle the data-highlighted attribute
// to highlight the currently focused option
const enterKeySubmitOnKeyDown = (e, is_enabled) => {
    if (is_enabled && e.which === 13 && !e.shiftKey) {
        e.preventDefault();
        if (!e.repeat) {
            if (e.target.form) {
                e.target.form.requestSubmit();
            }
        }
    }
}
"""


class ComboboxOption(HeadlessUIComponent):
    """
    The ComboboxOption reflex no-ssr component.
    A reflex react wrapper for the Headless UI ComboboxOption.
    https://headlessui.com/react/combobox#combobox-option
    """

    tag = "ComboboxOption"

    # Props from HeadlessUI ComboboxOption
    # key is inherited
    value: Var[str]
    disabled: Var[bool]

    def _get_all_custom_code(self) -> set[str]:
        """Include the custom code for highlighting on hovering using reflex styles"""
        custom_code = super()._get_all_custom_code()
        custom_code.add(ON_MOUSE_ENTER_LEAVE_JS)
        return custom_code

    # Consider support for highlighted state
    # Adding the attribute data-highlighted allows to use reflex stylesheets
    # We need to find a way to add this attribute when HeadlessUI adds the
    # data-focus attribute, which would save specific styles.
    # data_highlighted: Var[bool]  # Indicates if the option is highlighted
    # Possibly use renaming with _rename_props: dict[str, str] as in
    # https://reflex.dev/docs/wrapping-react/more-wrapping-examples/#react-pdf-renderer
    @classmethod
    def create(cls, *children, **props):
        """Create a ComboboxOption component."""
        custom_attrs = props.setdefault("custom_attrs", {})
        # This is sample code that we need to adapt to
        # set/unset data-highlighted on mouse enter/leave 
        # enter_key_submit = props.get("enter_key_submit")
        # enter_key_submit = Var.create(enter_key_submit)
        # custom_attrs["on_key_down"] = Var(
        #     _js_expr=f"(e) => enterKeySubmitOnKeyDown(e, {enter_key_submit!s})",
        #     _var_data=VarData.merge(enter_key_submit._get_all_var_data()),
        # )
        return super().create(*children, **props)


# Individual convenience functions for creating combobox components
combobox = Combobox.create
combobox_input = ComboboxInput.create
combobox_button = ComboboxButton.create
combobox_options = ComboboxOptions.create
combobox_option = ComboboxOption.create


# Wrapper function for the Combobox component
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
            class_name="rt-TextFieldRoot rt-r-size-2 rt-variant-surface",
        ),
        combobox_options(
            rx.foreach(
                options,
                lambda option: combobox_option(
                    rx.text(option),
                    value=option,
                    class_name="rt-SelectItem",
                ),
            ),
            class_name="rt-SelectContent rt-r-size-2 rt-variant-solid",
        ),
        value=value,
        on_change=on_change,
    )
