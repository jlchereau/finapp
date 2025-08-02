"""
Base class definition for Headless UI components.

See:
    - https://github.com/tailwindlabs/headlessui
    - https://headlessui.com/
"""

from reflex.components.component import Component


class HeadlessUIComponent(Component):
    """The base class for all Headless UI components."""

    library = "@headlessui/react"

    # Any additional libraries needed to use the component.
    lib_dependencies: list[str] = ["@heroicons/react"]
