"""
A base class for Tailwind's Headless UI components.

See:
    - https://github.com/tailwindlabs/headlessui
    - https://headlessui.com/

This base class is mainly responsible for loading the react libraries
according to Reflex documentation about React wrappers for components
that do not render server side:
    - https://reflex.dev/docs/wrapping-react/library-and-tags/
    #wrapping-a-dynamic-component
    - https://reflex.dev/docs/wrapping-react/more-wrapping-examples#react-leaflet
"""

from reflex.components.component import NoSSRComponent


class HeadlessUIComponent(NoSSRComponent):
    """Base class for all Headless UI components."""

    library = "@headlessui/react"

    # Any additional libraries needed to use the component.
    lib_dependencies: list[str] = ["@heroicons/react"]
