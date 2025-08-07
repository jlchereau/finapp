"""
A base class for Tailwind's Headless UI components.

See:
    - https://github.com/tailwindlabs/headlessui
    - https://headlessui.com/

This base class is mainly responsible for loading the react libraries
according to Reflex documentation about React wrappers:
    - https://reflex.dev/docs/wrapping-react/overview/
    - https://reflex.dev/docs/wrapping-react/library-and-tags/
    - https://reflex.dev/docs/wrapping-react/props/
    - https://reflex.dev/docs/wrapping-react/custom-code-and-hooks/
    - https://reflex.dev/docs/wrapping-react/imports-and-styles/
    - https://reflex.dev/docs/wrapping-react/local-packages/
    - https://reflex.dev/docs/wrapping-react/serializers/
    - https://reflex.dev/docs/wrapping-react/example/
    - https://reflex.dev/docs/wrapping-react/more-wrapping-examples/
"""

from reflex.components.component import Component


class HeadlessUIComponent(Component):
    """The base class for all Headless UI components."""

    library = "@headlessui/react"

    # Any additional libraries needed to use the component.
    lib_dependencies: list[str] = ["@heroicons/react"]
